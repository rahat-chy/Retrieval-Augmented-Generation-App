import sqlite3
import json
import logging
import os
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger("uvicorn")
DB_PATH = os.getenv("DB_PATH", "jobs.db")


def _conn():
    """Return a sqlite3 connection to jobs.db with Row factory enabled."""
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create jobs, chat_history, and documents tables if they don't exist; run additive migrations."""
    logger.info("Initializing SQLite DB at %s", DB_PATH)
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                params_json TEXT,
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT,
                source_refs_json TEXT,
                created_at TEXT NOT NULL
            )
        """)
        try:
            c.execute("ALTER TABLE chat_history ADD COLUMN source_refs_json TEXT")
        except Exception:
            pass
        c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                ingested_at TEXT NOT NULL
            )
        """)


def create_job(job_id: str, name: str, params: Optional[dict] = None):
    """Insert a new job row in 'running' status with serialized params."""
    logger.info("Creating job id=%s name=%s", job_id, name)
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, name, status, params_json, created_at, updated_at) VALUES (?, ?, 'running', ?, ?, ?)",
            (job_id, name, json.dumps(params) if params else None, now, now),
        )


def reset_job_for_retry(job_id: str):
    """Clear error and result on a failed job and reset its status to 'running'."""
    logger.info("Resetting job id=%s for retry", job_id)
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status='running', error=NULL, result_json=NULL, updated_at=? WHERE id=?",
            (now, job_id),
        )


def set_job_status(job_id: str, status: str, result=None, error: Optional[str] = None):
    """Update a job's status, serialized result, and error message."""
    if error:
        logger.error("Job id=%s status=%s error=%s", job_id, status, error)
    else:
        logger.info("Job id=%s status=%s", job_id, status)
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status=?, result_json=?, error=?, updated_at=? WHERE id=?",
            (
                status,
                json.dumps(result) if result is not None else None,
                error,
                datetime.now(timezone.utc).isoformat(),
                job_id,
            ),
        )


def get_job(job_id: str) -> Optional[dict]:
    """Fetch a job by ID and return it as a plain dict, or None if not found."""
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "status": row["status"],
        "params": json.loads(row["params_json"]) if row["params_json"] else None,
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def save_chat_message(question: str, answer: str, sources: list[str], source_refs: list[dict] | None = None):
    """Persist a question/answer exchange with source names and detailed refs to chat_history."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO chat_history (question, answer, sources_json, source_refs_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (question, answer, json.dumps(sources), json.dumps(source_refs) if source_refs is not None else None, now),
        )


def get_chat_history() -> list[dict]:
    """Return all chat history rows ordered by insertion ID, with JSON fields deserialized."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM chat_history ORDER BY id").fetchall()
    return [
        {
            "question": r["question"],
            "answer": r["answer"],
            "sources": json.loads(r["sources_json"]) if r["sources_json"] else [],
            "source_refs": json.loads(r["source_refs_json"]) if r["source_refs_json"] else [],
        }
        for r in rows
    ]


def register_document(doc_id: str, source_name: str, chunk_count: int):
    """Upsert a document record with its display name, chunk count, and ingestion timestamp."""
    logger.info("Registering document doc_id=%s source_name=%s chunks=%d", doc_id, source_name, chunk_count)
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO documents (doc_id, source_name, chunk_count, ingested_at) VALUES (?, ?, ?, ?)",
            (doc_id, source_name, chunk_count, now),
        )


def list_documents() -> list[dict]:
    """Return all registered documents ordered by most recent ingestion first."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
    return [
        {
            "doc_id": r["doc_id"],
            "source_name": r["source_name"],
            "chunk_count": r["chunk_count"],
            "ingested_at": r["ingested_at"],
        }
        for r in rows
    ]


def delete_document(doc_id: str):
    """Remove a document record from the documents table by its UUID."""
    logger.info("Deleting document doc_id=%s", doc_id)
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
