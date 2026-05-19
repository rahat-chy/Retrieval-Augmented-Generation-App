import logging
import uuid
import asyncio

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from graphs.ingest_graph import build_ingest_graph
from graphs.query_graph import build_query_graph
from job_runner import (
    init_db, create_job, set_job_status, get_job,
    reset_job_for_retry, save_chat_message, get_chat_history,
)

load_dotenv()

logger = logging.getLogger("uvicorn")
app = FastAPI()

_checkpointer = MemorySaver()
ingest_graph = build_ingest_graph(_checkpointer)
query_graph = build_query_graph(_checkpointer)


@app.on_event("startup")
async def startup():
    init_db()


# --- Request models ---

class IngestRequest(BaseModel):
    pdf_path: str
    source_id: str | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    history: list[dict] = []


# --- Pipelines ---

async def _run_ingest(job_id: str, pdf_path: str, source_id: str, thread_id: str | None = None):
    try:
        config = {"configurable": {"thread_id": thread_id or job_id}}
        result = await asyncio.to_thread(
            ingest_graph.invoke,
            {"pdf_path": pdf_path, "source_id": source_id, "chunks": [], "ingested": 0},
            config,
        )
        set_job_status(job_id, "completed", {"ingested": result["ingested"]})
    except Exception as e:
        set_job_status(job_id, "failed", error=str(e))
        logger.error(f"Ingest job {job_id} failed: {e}")


async def _run_query(job_id: str, question: str, top_k: int, history: list[dict] | None = None, thread_id: str | None = None):
    try:
        config = {"configurable": {"thread_id": thread_id or job_id}}
        result = await asyncio.to_thread(
            query_graph.invoke,
            {
                "question": question,
                "original_question": question,
                "top_k": top_k,
                "history": history or [],
                "contexts": [],
                "sources": [],
                "relevant_contexts": [],
                "relevant_sources": [],
                "answer": "",
                "rewrite_count": 0,
                "grounded": False,
            },
            config,
        )
        answer_data = {
            "answer": result["answer"],
            "sources": result.get("relevant_sources") or result.get("sources", []),
            "num_contexts": len(result.get("relevant_contexts") or result.get("contexts", [])),
            "grounded": result.get("grounded", True),
            "rewrites": result.get("rewrite_count", 0),
        }
        set_job_status(job_id, "completed", answer_data)
        save_chat_message(question, result["answer"], answer_data["sources"])
    except Exception as e:
        set_job_status(job_id, "failed", error=str(e))
        logger.error(f"Query job {job_id} failed: {e}")


# --- Endpoints ---

@app.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    source_id = req.source_id or req.pdf_path
    create_job(job_id, "ingest", {"pdf_path": req.pdf_path, "source_id": source_id})
    background_tasks.add_task(_run_ingest, job_id, req.pdf_path, source_id)
    return {"job_id": job_id, "status": "running"}


@app.post("/query")
async def query(req: QueryRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    create_job(job_id, "query", {"question": req.question, "top_k": req.top_k})
    background_tasks.add_task(_run_query, job_id, req.question, req.top_k, req.history)
    return {"job_id": job_id, "status": "running"}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/history")
async def history():
    return get_chat_history()


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "failed":
        raise HTTPException(status_code=400, detail=f"Job status is '{job['status']}', only 'failed' jobs can be retried")
    if not job["params"]:
        raise HTTPException(status_code=400, detail="Job has no stored params, cannot retry")

    reset_job_for_retry(job_id)

    # Retry uses a fresh thread_id so LangGraph runs from scratch
    retry_thread_id = str(uuid.uuid4())

    if job["name"] == "ingest":
        p = job["params"]
        background_tasks.add_task(_run_ingest, job_id, p["pdf_path"], p["source_id"], retry_thread_id)
    elif job["name"] == "query":
        p = job["params"]
        background_tasks.add_task(_run_query, job_id, p["question"], p["top_k"], None, retry_thread_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown job type '{job['name']}'")

    return {"job_id": job_id, "status": "running"}
