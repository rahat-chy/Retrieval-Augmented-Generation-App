import os
import tempfile
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG App", layout="centered")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Retrieval-Augmented Generation (RAG)")

# Init session state
_defaults = {
    "ingest_job_id": None,
    "ingest_status": None,
    "ingest_result": None,
    "ingest_last_msg": None,
    "ingest_tmp_path": None,
    "query_job_id": None,
    "query_status": None,
    "pending_question": None,
    "history_loaded": False,
    "chat_history": [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _get_job(job_id: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@st.fragment(run_every=2)
def _poll_ingest():
    if st.session_state.ingest_status != "running":
        return
    job = _get_job(st.session_state.ingest_job_id)
    if job["status"] == "completed":
        ingested = (job.get("result") or {}).get("ingested", "?")
        st.session_state.ingest_last_msg = ("success", f"Ingested {ingested} chunks.")
        st.session_state.ingest_status = None
        st.session_state.ingest_job_id = None
        tmp_path = st.session_state.ingest_tmp_path
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass
            st.session_state.ingest_tmp_path = None
        st.rerun()
    elif job["status"] == "failed":
        err = job.get("error", "unknown error")
        st.session_state.ingest_last_msg = ("error", f"Ingest failed: {err}")
        st.session_state.ingest_status = None
        st.session_state.ingest_job_id = None
        tmp_path = st.session_state.ingest_tmp_path
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass
            st.session_state.ingest_tmp_path = None
        st.rerun()


@st.fragment(run_every=2)
def _poll_query():
    if st.session_state.query_status != "running":
        return
    job = _get_job(st.session_state.query_job_id)
    if job["status"] == "completed":
        result = job.get("result", {})
        st.session_state.chat_history.append({
            "question": st.session_state.pending_question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        })
        st.session_state.query_status = None
        st.session_state.query_job_id = None
        st.session_state.pending_question = None
        st.rerun()
    elif job["status"] == "failed":
        err = job.get("error", "unknown error")
        st.session_state.chat_history.append({
            "question": st.session_state.pending_question,
            "answer": f"Query failed: {err}",
            "sources": [],
        })
        st.session_state.query_status = None
        st.session_state.query_job_id = None
        st.session_state.pending_question = None
        st.rerun()


# Load history from DB once per session
if not st.session_state.history_loaded:
    try:
        resp = requests.get(f"{API_BASE}/history", timeout=5)
        resp.raise_for_status()
        st.session_state.chat_history = resp.json()
    except Exception:
        pass
    st.session_state.history_loaded = True


# ── Ingest ──────────────────────────────────────────────────────────────────

st.header("Ingest PDF")

ingest_disabled = st.session_state.ingest_status == "running" or st.session_state.query_status == "running"
uploaded_file = st.file_uploader(
    "Select PDF",
    type=["pdf"],
    disabled=ingest_disabled,
    label_visibility="collapsed",
)

if st.button("Ingest", disabled=ingest_disabled or uploaded_file is None):
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    st.session_state.ingest_tmp_path = tmp.name

    try:
        resp = requests.post(
            f"{API_BASE}/ingest",
            json={"pdf_path": tmp.name, "source_id": uploaded_file.name},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        st.session_state.ingest_job_id = data["job_id"]
        st.session_state.ingest_status = "running"
    except Exception as e:
        st.error(f"Failed to start ingest: {e}")
        os.unlink(tmp.name)

    st.rerun()

if st.session_state.ingest_status == "running":
    st.info("⏳ Ingesting...")

_poll_ingest()

if st.session_state.ingest_last_msg:
    kind, msg = st.session_state.ingest_last_msg
    if kind == "success":
        st.success(msg)
    else:
        st.error(msg)


# ── Chat ─────────────────────────────────────────────────────────────────────

st.divider()
st.header("Chat")

with st.container(border=True, height=400):
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")

    if st.session_state.pending_question:
        with st.chat_message("user"):
            st.write(st.session_state.pending_question)

    if st.session_state.query_status == "running":
        with st.chat_message("assistant"):
            st.write("⏳ Thinking...")

_poll_query()

question = st.chat_input(
    "Ask something about your documents...",
    disabled=st.session_state.query_status == "running" or st.session_state.ingest_status == "running",
)

if question and question.strip():
    history = [
        msg
        for h in st.session_state.chat_history
        for msg in [
            {"role": "user", "content": h["question"]},
            {"role": "assistant", "content": h["answer"]},
        ]
    ]
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": 5, "history": history},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        st.session_state.query_job_id = data["job_id"]
        st.session_state.query_status = "running"
        st.session_state.pending_question = question
    except Exception as e:
        st.error(f"Failed to start query: {e}")
    st.rerun()
