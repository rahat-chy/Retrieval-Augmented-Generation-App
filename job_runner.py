import sqlite3
import json
import logging
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger("uvicorn")
DB_PATH = "jobs.db"


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
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
                created_at TEXT NOT NULL
            )
        """)


def create_job(job_id: str, name: str, params: Optional[dict] = None):
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, name, status, params_json, created_at, updated_at) VALUES (?, ?, 'running', ?, ?, ?)",
            (job_id, name, json.dumps(params) if params else None, now, now),
        )


def reset_job_for_retry(job_id: str):
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status='running', error=NULL, result_json=NULL, updated_at=? WHERE id=?",
            (now, job_id),
        )


def set_job_status(job_id: str, status: str, result=None, error: Optional[str] = None):
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


def save_chat_message(question: str, answer: str, sources: list[str]):
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO chat_history (question, answer, sources_json, created_at) VALUES (?, ?, ?, ?)",
            (question, answer, json.dumps(sources), now),
        )


def get_chat_history() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM chat_history ORDER BY id").fetchall()
    return [
        {
            "question": r["question"],
            "answer": r["answer"],
            "sources": json.loads(r["sources_json"]) if r["sources_json"] else [],
        }
        for r in rows
    ]
