import logging
import uuid
import os

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import ollama

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult
from job_runner import init_db, create_job, run_step, set_job_status, get_job, reset_job_for_retry, save_chat_message, get_chat_history

load_dotenv()

logger = logging.getLogger("uvicorn")
app = FastAPI()


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

async def _run_ingest(job_id: str, pdf_path: str, source_id: str):
    try:
        def _load():
            chunks = load_and_chunk_pdf(pdf_path)
            return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

        chunks_and_src = await run_step(job_id, "load-and-chunk", _load, output_type=RAGChunkAndSrc)

        def _upsert():
            chunks = chunks_and_src.chunks
            vecs = embed_texts([c["text"] for c in chunks])
            ids = [c["id"] for c in chunks]
            payload = [
                {"source": chunks_and_src.source_id, "text": c["text"], "parent_text": c["parent_text"]}
                for c in chunks
            ]
            QdrantStorage().upsert(ids, vecs, payload)
            return RAGUpsertResult(ingested=len(chunks))

        result = await run_step(job_id, "embed-and-upsert", _upsert, output_type=RAGUpsertResult)
        set_job_status(job_id, "completed", result.model_dump())
    except Exception as e:
        set_job_status(job_id, "failed", error=str(e))
        logger.error(f"Ingest job {job_id} failed: {e}")


async def _run_query(job_id: str, question: str, top_k: int, history: list[dict] | None = None):
    try:
        def _search():
            query_vec = embed_texts([question])[0]
            store = QdrantStorage()
            found = store.search(query_vec, top_k)
            return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

        found = await run_step(job_id, "embed-and-search", _search, output_type=RAGSearchResult)

        def _llm_answer():
            context_block = "\n\n".join(f"- {c}" for c in found.contexts)
            prompt = (
                "Use the following context to answer the question.\n\n"
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n"
                "Answer concisely using the context above."
            )
            messages = [
                {"role": "system", "content": "You answer questions using only the provided context."},
                *(history or []),
                {"role": "user", "content": prompt},
            ]
            res = ollama.chat(model="llama3.2", messages=messages)
            answer = res["message"]["content"].strip()
            return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

        result = await run_step(job_id, "llm-answer", _llm_answer)
        set_job_status(job_id, "completed", result)
        save_chat_message(question, result["answer"], result.get("sources", []))
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

    if job["name"] == "ingest":
        p = job["params"]
        background_tasks.add_task(_run_ingest, job_id, p["pdf_path"], p["source_id"])
    elif job["name"] == "query":
        p = job["params"]
        background_tasks.add_task(_run_query, job_id, p["question"], p["top_k"])
    else:
        raise HTTPException(status_code=400, detail=f"Unknown job type '{job['name']}'")

    return {"job_id": job_id, "status": "running"}
