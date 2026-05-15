import os
import tempfile
import time

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
    "ingest_last_msg": None,  # ("success"|"error", message)
    "ingest_tmp_path": None,
    "query_job_id": None,
    "query_status": None,
    "query_result": None,
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
        st.session_state.ingest_result = None
    except Exception as e:
        st.error(f"Failed to start ingest: {e}")
        os.unlink(tmp.name)

    st.rerun()

# Ingest polling
if st.session_state.ingest_status == "running":
    with st.spinner("Ingesting..."):
        time.sleep(2)
    job = _get_job(st.session_state.ingest_job_id)
    if job["status"] == "completed":
        ingested = (job.get("result") or {}).get("ingested", "?")
        st.session_state.ingest_last_msg = ("success", f"Ingested {ingested} chunks.")
        st.session_state.ingest_status = None
        st.session_state.ingest_job_id = None
        st.session_state.ingest_result = None
    elif job["status"] == "failed":
        err = job.get("error", "unknown error")
        st.session_state.ingest_last_msg = ("error", f"Ingest failed: {err}")
        st.session_state.ingest_status = None
        st.session_state.ingest_job_id = None
        st.session_state.ingest_result = None
    tmp_path = st.session_state.ingest_tmp_path
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass  # backend still holds handle; OS cleans temp on restart
        st.session_state.ingest_tmp_path = None
    st.rerun()

# Ingest last message
if st.session_state.ingest_last_msg:
    kind, msg = st.session_state.ingest_last_msg
    if kind == "success":
        st.success(msg)
    else:
        st.error(msg)


# ── Query ────────────────────────────────────────────────────────────────────

st.divider()
st.header("Query")

query_disabled = st.session_state.query_status == "running"
question = st.text_area(
    "Question",
    placeholder="Ask something about your documents...",
    disabled=query_disabled,
    label_visibility="collapsed",
)

if st.button("Query", disabled=query_disabled or not (question or "").strip()):
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": 5},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        st.session_state.query_job_id = data["job_id"]
        st.session_state.query_status = "running"
        st.session_state.query_result = None
    except Exception as e:
        st.error(f"Failed to start query: {e}")

    st.rerun()

# Query polling
if st.session_state.query_status == "running":
    with st.spinner("Querying..."):
        time.sleep(2)
    job = _get_job(st.session_state.query_job_id)
    if job["status"] in ("completed", "failed"):
        st.session_state.query_status = job["status"]
        st.session_state.query_result = job
    st.rerun()

# Query result
if st.session_state.query_status == "completed":
    result = (st.session_state.query_result or {}).get("result", {})
    st.subheader("Answer")
    st.write(result.get("answer", ""))
    sources = result.get("sources", [])
    if sources:
        st.subheader("Sources")
        for src in sources:
            st.write(f"- {src}")
    if st.button("New query"):
        st.session_state.query_status = None
        st.session_state.query_job_id = None
        st.session_state.query_result = None
        st.rerun()

elif st.session_state.query_status == "failed":
    err = (st.session_state.query_result or {}).get("error", "unknown error")
    st.error(f"Query failed: {err}")
    if st.button("New query"):
        st.session_state.query_status = None
        st.session_state.query_job_id = None
        st.session_state.query_result = None
        st.rerun()
