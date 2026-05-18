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

/* ── Chat history: rounder corners ───────────────────────────── */
.st-key-chat_history_container [data-testid="stVerticalBlockBorderWrapper"],
.st-key-chat_history_container > div:first-child {
    border-radius: 10px !important;
}

/* ── Form: no border, small gap ─────────────────────────────── */
[data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    box-shadow: none !important;
    margin-top: 4px !important;
}

/* ── Horizontal block inside form: relative for abs button ── */
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 0 !important;
    position: relative !important;
}

/* ── Input: pill shape, subtle border ───────────────────────── */
[data-testid="stForm"] [data-testid="stTextInput"] > div {
    border-radius: 15px !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input {
    border-radius: 15px !important;
    padding: 12px 72px 12px 20px !important;
    border: 1.5px solid rgba(128, 128, 128, 0.2) !important;
    font-size: 15px !important;
    background: transparent !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input:focus {
    border-color: rgba(255, 75, 75, 0.45) !important;
    box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.08) !important;
    outline: none !important;
}

/* ── Button column: absolute, overlaps right of input ───────── */
[data-testid="stForm"] [data-testid="column"]:last-child {
    position: absolute !important;
    right: 12px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: auto !important;
    flex: unset !important;
    min-width: 0 !important;
}

/* ── Submit button: circular, ChatGPT-style arrow ───────────── */
[data-testid="stFormSubmitButton"] button {
    border-radius: 25% !important;
    width: 32px !important;
    height: 32px !important;
    min-height: unset !important;
    padding: 0 !important;
    background: #1a1a1a !important;
    color: transparent !important;
    border: none !important;
    font-size: 0 !important;
    line-height: 1 !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease !important;
    position: relative !important;
    overflow: visible !important;
}
[data-testid="stFormSubmitButton"] button::after {
    content: "" !important;
    position: absolute !important;
    inset: 0 !important;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 19V5'/%3E%3Cpath d='M5 12l7-7 7 7'/%3E%3C/svg%3E") no-repeat center / 16px 16px !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: scale(1.08) !important;
    background: #333 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}
[data-testid="stFormSubmitButton"] button:active {
    transform: scale(0.93) !important;
}
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
    "input_key_tracker": 0,
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

with st.container():

    with st.container(height=600, border=True, key="chat_history_container"):
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


    input_key = f"user_query_{st.session_state.input_key_tracker}"

    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([9, 1])

        with cols[0]:
            question = st.text_input(
                "Question",
                disabled=st.session_state.query_status == "running" or st.session_state.ingest_status == "running",
                placeholder="Ask something about your documents...",
                key=input_key,
                label_visibility="collapsed"
            )

        with cols[1]:
            submit_btn = st.form_submit_button("▲", use_container_width=True, disabled=st.session_state.query_status == "running" or st.session_state.ingest_status == "running")


_poll_query()


_is_busy = st.session_state.query_status == "running" or st.session_state.ingest_status == "running"
if submit_btn and question and question.strip() and not _is_busy:
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
        st.session_state.input_key_tracker += 1
    except Exception as e:
        st.error(f"Failed to start query: {e}")
    st.rerun()
