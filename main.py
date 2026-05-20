import json
import logging
import uuid
import asyncio

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from graphs.ingest_graph import build_ingest_graph
from graphs.query_graph import build_query_graph
from job_runner import (
    init_db, create_job, set_job_status, get_job,
    reset_job_for_retry, save_chat_message, get_chat_history,
    register_document, list_documents, delete_document,
)
from vector_db import QdrantStorage

load_dotenv()

logger = logging.getLogger("uvicorn")
app = FastAPI()

_checkpointer = MemorySaver()
ingest_graph = build_ingest_graph(_checkpointer)
query_graph = build_query_graph(_checkpointer)


@app.on_event("startup")
async def startup():
    """Initialize the SQLite job database on app startup."""
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

async def _run_ingest(job_id: str, pdf_path: str, source_id: str, source_name: str, thread_id: str | None = None):
    """Run the ingest LangGraph in a thread, update job status, and register the document on success."""
    try:
        config = {"configurable": {"thread_id": thread_id or job_id}}
        result = await asyncio.to_thread(
            ingest_graph.invoke,
            {"pdf_path": pdf_path, "source_id": source_id, "chunks": [], "ingested": 0},
            config,
        )
        set_job_status(job_id, "completed", {"ingested": result["ingested"]})
        register_document(source_id, source_name, result["ingested"])
    except Exception as e:
        set_job_status(job_id, "failed", error=str(e))
        logger.error(f"Ingest job {job_id} failed: {e}")


# --- Endpoints ---

@app.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Create an ingest job and start the ingest pipeline in the background."""
    job_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    source_name = req.source_id or req.pdf_path
    create_job(job_id, "ingest", {"pdf_path": req.pdf_path, "source_id": doc_id, "source_name": source_name})
    background_tasks.add_task(_run_ingest, job_id, req.pdf_path, doc_id, source_name)
    return {"job_id": job_id, "status": "running"}



@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Stream query tokens and status events via SSE, then persist the exchange to chat history."""
    job_id = str(uuid.uuid4())
    create_job(job_id, "query", {"question": req.question, "top_k": req.top_k})

    async def event_stream():
        try:
            yield f"data: {json.dumps({'thinking': True, 'job_id': job_id})}\n\n"
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = {
                "question": req.question,
                "original_question": req.question,
                "top_k": req.top_k,
                "history": req.history or [],
                "contexts": [],
                "source_refs": [],
                "relevant_contexts": [],
                "answer": "",
                "rewrite_count": 0,
            }
            full_answer: list[str] = []
            meta: dict = {}
            async for event in query_graph.astream_events(initial_state, config, version="v2"):
                kind = event["event"]
                name = event.get("name", "")
                if kind == "on_custom_event" and name == "token":
                    token = event["data"]
                    full_answer.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
                elif kind == "on_custom_event" and name == "status":
                    yield f"data: {json.dumps({'status': event['data']})}\n\n"
                elif kind == "on_custom_event" and name == "final_meta":
                    meta = event["data"]

            raw_source_refs = meta.get("source_refs", [])
            doc_names = {d["doc_id"]: d["source_name"] for d in list_documents()}
            named_source_refs = [
                {**ref, "source": doc_names.get(ref["source"], ref["source"])}
                for ref in raw_source_refs
            ]
            named_sources = list(dict.fromkeys(ref["source"] for ref in named_source_refs))
            answer_text = "".join(full_answer)
            answer_data = {
                "answer": answer_text,
                "sources": named_sources,
                "source_refs": named_source_refs,
                "rewrites": meta.get("rewrite_count", 0),
            }
            set_job_status(job_id, "completed", answer_data)
            yield f"data: {json.dumps({'done': True, 'sources': named_sources, 'source_refs': named_source_refs, 'rewrites': meta.get('rewrite_count', 0)})}\n\n"
            save_chat_message(req.question, answer_text, named_sources, named_source_refs)
        except Exception as e:
            set_job_status(job_id, "failed", error=str(e))
            logger.error(f"Stream query job {job_id} failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Return current status, params, and result for a job; 404 if not found."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/history")
async def history():
    """Return all stored chat messages in chronological order."""
    return get_chat_history()


@app.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """Reset a failed job and re-run its pipeline with a fresh LangGraph thread ID."""
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
        background_tasks.add_task(_run_ingest, job_id, p["pdf_path"], p["source_id"], p.get("source_name", p["source_id"]), retry_thread_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown job type '{job['name']}'")

    return {"job_id": job_id, "status": "running"}


@app.get("/documents")
async def get_documents():
    """Return all registered documents ordered by most recent ingestion."""
    return list_documents()


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str):
    """Delete a document's vectors from Qdrant and its metadata record from the DB."""
    QdrantStorage().delete_by_source(doc_id)
    delete_document(doc_id)
    return {"deleted": doc_id}
