import sqlite3
import json
import asyncio
import logging
from typing import Any, Callable, Optional, Type
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
        try:
            c.execute("ALTER TABLE jobs ADD COLUMN params_json TEXT")
        except sqlite3.OperationalError:
            pass
        c.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                job_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result_json TEXT,
                attempts INTEGER DEFAULT 0,
                error TEXT,
                PRIMARY KEY (job_id, name)
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
        c.execute("DELETE FROM steps WHERE job_id=? AND status='failed'", (job_id,))


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
        steps = c.execute("SELECT * FROM steps WHERE job_id=?", (job_id,)).fetchall()
    return {
        "id": row["id"],
        "name": row["name"],
        "status": row["status"],
        "params": json.loads(row["params_json"]) if row["params_json"] else None,
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "steps": [
            {"name": s["name"], "status": s["status"], "attempts": s["attempts"]}
            for s in steps
        ],
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


async def run_step(
    job_id: str,
    step_name: str,
    fn: Callable,
    output_type: Optional[Type] = None,
    max_retries: int = 3,
) -> Any:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM steps WHERE job_id=? AND name=?", (job_id, step_name)
        ).fetchone()

    # Resume from checkpoint if step already completed
    if row and row["status"] == "completed":
        logger.info(f"job={job_id} step={step_name} resumed from checkpoint")
        data = json.loads(row["result_json"])
        if output_type and isinstance(data, dict):
            return output_type(**data)
        return data

    last_err: Exception = RuntimeError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            result = await asyncio.to_thread(fn)
            result_data = result.model_dump() if hasattr(result, "model_dump") else result
            with _conn() as c:
                c.execute(
                    """INSERT OR REPLACE INTO steps (job_id, name, status, result_json, attempts)
                       VALUES (?, ?, 'completed', ?, ?)""",
                    (job_id, step_name, json.dumps(result_data), attempt),
                )
            return result
        except Exception as e:
            last_err = e
            logger.warning(f"job={job_id} step={step_name} attempt={attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(2 ** (attempt - 1))

    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO steps (job_id, name, status, attempts, error)
               VALUES (?, ?, 'failed', ?, ?)""",
            (job_id, step_name, max_retries, str(last_err)),
        )
    raise last_err
