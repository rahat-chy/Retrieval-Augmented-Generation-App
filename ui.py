import json
import os
import tempfile
import uuid
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG App", page_icon="🔍", layout="centered")

st.markdown("""
<style>
/* ── Hide Streamlit chrome ─────────────────────────────── */
#MainMenu, header, footer { visibility: hidden; }

/* ── Background ─────────────────────────────────────────── */
.stApp {
    background: linear-gradient(160deg, #0d0d1a 0%, #12112b 45%, #0f1a2e 100%) !important;
}
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* ── Global text color for dark bg ─────────────────────── */
p, span, label, li, td, th,
.stMarkdown, .stMarkdown p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stText"],
[data-testid="stChatMessage"] p,
[data-testid="stCaptionContainer"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
.stCaption {
    color: #cbd5e1 !important;
}
strong, b {
    color: #e2e8f0 !important;
}

/* ── Page: subtle top accent bar ───────────────────────── */
.stApp::before {
    content: "";
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a78bfa, #c084fc, #60a5fa);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
}

/* ── Gradient title ─────────────────────────────────────── */
.rag-title {
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 45%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    margin: 0.6rem 0 0.2rem !important;
    line-height: 1.15 !important;
}
.rag-subtitle {
    font-size: 1rem;
    opacity: 0.45;
    margin-bottom: 2rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Section headers ────────────────────────────────────── */
.sh {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #e2e8f0;
    opacity: 0.9;
    margin-bottom: 0.9rem;
    margin-top: 0.2rem;
}
.sh-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sh-ingest .sh-dot { background: #818cf8; }
.sh-docs   .sh-dot { background: #34d399; }
.sh-chat   .sh-dot { background: #f472b6; }

/* ── Divider ─────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(148, 163, 184, 0.25) !important;
    margin: 1.6rem 0 !important;
}

/* ── File uploader ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(129, 140, 248, 0.45) !important;
    border-radius: 12px !important;
    padding: 0.2rem 0.8rem !important;
    transition: border-color 0.2s, background 0.2s !important;
    background: #0d0d1a !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(129, 140, 248, 0.75) !important;
    background: #0e0e20 !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploaderDropzone"] {
    background: #0d0d1a !important;
    border: none !important;
    border-radius: 10px !important;
}
/* Browse files button: dark bg, white on hover */
[data-testid="stFileUploader"] button:not(:disabled) {
    background: #0d0d1a !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(148, 163, 184, 0.28) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    font-size: 0.82rem !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
}
[data-testid="stFileUploader"] button:not(:disabled):hover {
    background: white !important;
    color: #0d0d1a !important;
    border-color: white !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Primary buttons ────────────────────────────────────── */
[data-testid="stButton"] > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.01em !important;
    transition: transform 0.12s, box-shadow 0.12s, background 0.15s !important;
    background-color: rgba(255, 255, 255, 0.12) !important;
    color: rgba(255, 255, 255, 0.35) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}
[data-testid="stButton"] > button:not(:disabled) {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(79, 70, 229, 0.45) !important;
}
[data-testid="stButton"] > button:not(:disabled):hover {
    background-color: #6366f1 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(99, 102, 241, 0.55) !important;
}
[data-testid="stButton"] > button:not(:disabled):active {
    transform: scale(0.97) !important;
}

/* ── Alerts ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.86rem !important;
}

/* ── Caption text ───────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] p {
    font-size: 0.78rem !important;
    opacity: 0.55 !important;
}

/* ── Expanders ──────────────────────────────────────────── */
[data-testid="stExpander"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    background-color: #0d0d1a !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > div,
[data-testid="stExpander"] details[open],
[data-testid="stExpander"] details[open] > div,
[data-testid="stExpander"] details[open] > div > div {
    background-color: #0d0d1a !important;
}
[data-testid="stExpander"] details summary p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    opacity: 0.7 !important;
}

/* ── Delete buttons (doc table) ─────────────────────────── */
.st-key-doc_table [data-testid="stButton"] button,
.st-key-doc_table [data-testid="stButton"] button:not(:disabled) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    opacity: 0.08 !important;
    transition: opacity 0.15s, transform 0.12s !important;
}
.st-key-doc_table [data-testid="stButton"] button:not(:disabled):hover {
    background: rgba(248, 113, 113, 0.1) !important;
    opacity: 1 !important;
    transform: scale(1.15) !important;
    box-shadow: none !important;
}

/* ── Chat: uniform text color user + assistant ──────────── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div > p {
    color: #cbd5e1 !important;
}

/* ── Text areas (source excerpts) ───────────────────────── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextArea"] textarea:disabled,
textarea {
    border-radius: 8px !important;
    border: 1px solid rgba(148, 163, 184, 0.28) !important;
    font-size: 0.82rem !important;
    background-color: #0d0d1a !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    resize: none !important;
}
[data-testid="stTextArea"],
[data-testid="stTextArea"] > div,
[data-testid="stTextArea"] > div > div {
    background-color: #0d0d1a !important;
    border-radius: 8px !important;
}

/* ── Chat container ─────────────────────────────────────── */
.st-key-chat_history_container [data-testid="stVerticalBlockBorderWrapper"],
.st-key-chat_history_container > div:first-child {
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
}

/* ── Doc section border ─────────────────────────────────── */
.st-key-doc_section [data-testid="stVerticalBlockBorderWrapper"],
.st-key-doc_section > div:first-child {
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    padding: 0.75rem !important;
}

/* ── Outer white border: chat + doc sections ────────────── */
.st-key-chat_history_container,
.st-key-doc_section {
    border: 1.5px solid rgba(255, 255, 255, 0.75) !important;
    border-radius: 14px !important;
}

/* ── Chat messages ──────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    margin-bottom: 2px !important;
}

/* ── Document table header ──────────────────────────────── */
.doc-header {
    display: grid;
    grid-template-columns: 5fr 2fr 2fr 1fr;
    gap: 0.5rem;
    padding: 0.4rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.4;
    border-bottom: 1px solid rgba(148, 163, 184, 0.25);
    margin-bottom: 0.25rem;
}

/* ── Form: no border ─────────────────────────────────────── */
[data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    box-shadow: none !important;
    margin-top: 6px !important;
}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 0 !important;
    position: relative !important;
}

/* ── Input pill ─────────────────────────────────────────── */
[data-testid="stForm"] [data-testid="stTextInput"] > div {
    border-radius: 16px !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input {
    border-radius: 16px !important;
    padding: 13px 72px 13px 20px !important;
    border: 1.5px solid rgba(148, 163, 184, 0.2) !important;
    font-size: 15px !important;
    background: transparent !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stForm"] [data-testid="stTextInput"] input:focus {
    border-color: rgba(129, 140, 248, 0.55) !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.1) !important;
    outline: none !important;
}

/* ── Submit button column: absolute overlay ─────────────── */
[data-testid="stForm"] [data-testid="column"]:last-child {
    position: absolute !important;
    right: 12px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: auto !important;
    flex: unset !important;
    min-width: 0 !important;
}

/* ── Submit button: circle with arrow ──────────────────── */
[data-testid="stFormSubmitButton"] button {
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    min-height: unset !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: transparent !important;
    border: none !important;
    font-size: 0 !important;
    line-height: 1 !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    position: relative !important;
}
[data-testid="stFormSubmitButton"] button::after {
    content: "" !important;
    position: absolute !important;
    inset: 0 !important;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 19V5'/%3E%3Cpath d='M5 12l7-7 7 7'/%3E%3C/svg%3E") no-repeat center / 16px 16px !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.55) !important;
}
[data-testid="stFormSubmitButton"] button:active {
    transform: scale(0.92) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="rag-title">🔍 RAG App</div>', unsafe_allow_html=True)
st.markdown('<div class="rag-subtitle">Ask questions across your PDF documents using retrieval-augmented generation.</div>', unsafe_allow_html=True)

# Init session state
_defaults = {
    "ingest_job_id": None,
    "ingest_status": None,
    "ingest_result": None,
    "ingest_last_msg": None,
    "ingest_tmp_path": None,
    "ingest_failed_job_id": None,
    "ingest_failed_source_id": None,
    "pending_question": None,
    "streaming": False,
    "stream_history": [],
    "stream_result": {},
    "history_loaded": False,
    "chat_history": [],
    "input_key_tracker": 0,
    "uploader_key": 0,
    "documents": [],
    "documents_loaded": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _render_source_refs(refs: list[dict]):
    """Render source reference cards with matched excerpt and full context expanders."""
    all_chunks = []
    for ref in refs:
        all_chunks.append(ref)
        all_chunks.extend(ref.get("siblings", []))

    for chunk in all_chunks:
        st.markdown(
            f'<div style="font-size:0.83rem;font-weight:600;margin-bottom:0.25rem;">'
            f'📄 {chunk["source"]} &nbsp;<span style="opacity:0.45;font-weight:400;">page {chunk["page_num"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        excerpt = chunk.get("excerpt", "")
        if excerpt:
            st.caption("Matched chunk:")
            st.text_area("", value=excerpt, height=120, disabled=True, key=uuid.uuid4().hex)
        full_ctx = chunk.get("context_preview", "")
        if full_ctx and full_ctx != excerpt:
            with st.expander("Full context"):
                st.text_area("", value=full_ctx, height=200, disabled=True, key=uuid.uuid4().hex)


def _get_job(job_id: str) -> dict:
    """Fetch job status from the API; return a failed-status dict on network or HTTP error."""
    try:
        resp = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@st.fragment(run_every=2)
def _poll_ingest():
    """Streamlit fragment that polls the ingest job every 2 seconds and updates session state on completion or failure."""
    if st.session_state.ingest_status != "running":
        return
    job = _get_job(st.session_state.ingest_job_id)
    if job["status"] == "completed":
        ingested = (job.get("result") or {}).get("ingested", "?")
        st.session_state.ingest_last_msg = ("success", f"Ingested {ingested} chunks.")
        st.session_state.ingest_status = None
        st.session_state.ingest_job_id = None
        st.session_state.uploader_key += 1
        st.session_state.documents_loaded = False
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
        st.session_state.ingest_failed_job_id = st.session_state.ingest_job_id
        st.session_state.ingest_failed_source_id = (job.get("params") or {}).get("source_id")
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


# Load history from DB once per session
if not st.session_state.history_loaded:
    try:
        resp = requests.get(f"{API_BASE}/history", timeout=5)
        resp.raise_for_status()
        st.session_state.chat_history = resp.json()
    except Exception:
        pass
    st.session_state.history_loaded = True


# ── Ingest ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="sh sh-ingest"><span class="sh-dot"></span>Ingest PDF</div>', unsafe_allow_html=True)

ingest_disabled = st.session_state.ingest_status == "running" or st.session_state.streaming
uploaded_file = st.file_uploader(
    "Select PDF",
    type=["pdf"],
    disabled=ingest_disabled,
    label_visibility="collapsed",
    key=f"file_uploader_{st.session_state.uploader_key}",
)

same_file = (
    uploaded_file is not None
    and uploaded_file.name == st.session_state.ingest_failed_source_id
)
retry_disabled = ingest_disabled or not st.session_state.ingest_failed_job_id or not same_file

col_ingest, col_retry = st.columns(2)

with col_ingest:
    if st.button("⬆ Ingest", disabled=ingest_disabled or uploaded_file is None, use_container_width=True):
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

with col_retry:
    if st.button("↺ Retry", disabled=retry_disabled, use_container_width=True):
        try:
            resp = requests.post(
                f"{API_BASE}/jobs/{st.session_state.ingest_failed_job_id}/retry",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.ingest_job_id = data["job_id"]
            st.session_state.ingest_status = "running"
            st.session_state.ingest_last_msg = None
        except Exception as e:
            st.error(f"Retry failed: {e}")
        st.rerun()

if st.session_state.ingest_status == "running":
    st.info("⏳ Ingesting PDF — this may take a moment...")

_poll_ingest()

if st.session_state.ingest_last_msg:
    kind, msg = st.session_state.ingest_last_msg
    if kind == "success":
        st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")


# ── Documents ──────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="sh sh-docs"><span class="sh-dot"></span>Ingested Documents</div>', unsafe_allow_html=True)

if not st.session_state.documents_loaded:
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=5)
        resp.raise_for_status()
        st.session_state.documents = resp.json()
        st.session_state.documents_loaded = True
    except Exception:
        pass

with st.container(key="doc_section", border=True):
    if not st.session_state.documents:
        st.caption("No documents ingested yet.")
    else:
        st.markdown(
            '<div class="doc-header"><span>Name</span><span>Chunks</span><span>Ingested</span><span></span></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="doc_table"):
            for doc in st.session_state.documents:
                cols = st.columns([5, 2, 2, 1])
                cols[0].write(doc["source_name"])
                cols[1].write(str(doc["chunk_count"]))
                ingested_date = doc["ingested_at"][:10] if doc.get("ingested_at") else "—"
                cols[2].write(ingested_date)
                with cols[3]:
                    if st.button("🗑", key=f"del_{doc['doc_id']}", help="Delete document", disabled=ingest_disabled):
                        try:
                            del_resp = requests.delete(f"{API_BASE}/documents/{doc['doc_id']}", timeout=10)
                            del_resp.raise_for_status()
                            st.session_state.documents_loaded = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")


# ── Chat ───────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="sh sh-chat"><span class="sh-dot"></span>Chat</div>', unsafe_allow_html=True)

with st.container():

    with st.container(height=600, border=True, key="chat_history_container"):
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg["question"])
            with st.chat_message("assistant"):
                st.write(msg["answer"])
                meta_parts = []
                if msg.get("rewrites", 0) > 0:
                    meta_parts.append(f"🔄 Query rewritten {msg['rewrites']}x")
                if meta_parts:
                    st.caption(" · ".join(meta_parts))
                source_refs = msg.get("source_refs", [])
                if source_refs:
                    with st.expander("📎 Sources"):
                        _render_source_refs(source_refs)
                elif msg.get("sources"):
                    with st.expander("📎 Sources"):
                        for src in msg["sources"]:
                            st.write(f"- {src}")

        if st.session_state.pending_question:
            with st.chat_message("user"):
                st.write(st.session_state.pending_question)

        if st.session_state.streaming:
            with st.chat_message("assistant"):
                _q = st.session_state.pending_question
                _hist = st.session_state.stream_history
                _placeholder = st.empty()
                _placeholder.markdown("⏳ _Thinking..._")
                _parts: list[str] = []

                with requests.post(
                    f"{API_BASE}/query/stream",
                    json={"question": _q, "top_k": 5, "history": _hist},
                    stream=True,
                    timeout=600,
                ) as _resp:
                    for _line in _resp.iter_lines():
                        if not _line:
                            continue
                        if _line.startswith(b"data: "):
                            _data = json.loads(_line[6:])
                            if "status" in _data:
                                _placeholder.markdown(f"⏳ _{_data['status']}_")
                            elif "token" in _data:
                                _parts.append(_data["token"])
                                _placeholder.markdown("".join(_parts) + " ▌")
                            elif _data.get("done"):
                                st.session_state.stream_result = {
                                    "sources": _data.get("sources", []),
                                    "source_refs": _data.get("source_refs", []),
                                    "rewrites": _data.get("rewrites", 0),
                                }

                _full = "".join(_parts)
                _placeholder.markdown(_full)
                _res = st.session_state.get("stream_result", {})
                if _res.get("rewrites", 0) > 0:
                    st.caption(f"🔄 Query rewritten {_res['rewrites']}x")
                _source_refs = _res.get("source_refs", [])
                if _source_refs:
                    with st.expander("📎 Sources"):
                        _render_source_refs(_source_refs)
                elif _res.get("sources"):
                    with st.expander("📎 Sources"):
                        for _src in _res["sources"]:
                            st.write(f"- {_src}")

            st.session_state.chat_history.append({
                "question": st.session_state.pending_question,
                "answer": _full,
                "sources": st.session_state.stream_result.get("sources", []),
                "source_refs": st.session_state.stream_result.get("source_refs", []),
                "rewrites": st.session_state.stream_result.get("rewrites", 0),
            })
            st.session_state.pending_question = None
            st.session_state.streaming = False
            st.session_state.stream_result = {}
            st.rerun()


    input_key = f"user_query_{st.session_state.input_key_tracker}"

    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([9, 1])

        with cols[0]:
            question = st.text_input(
                "Question",
                disabled=st.session_state.streaming or st.session_state.ingest_status == "running",
                placeholder="Ask something about your documents...",
                key=input_key,
                label_visibility="collapsed"
            )

        with cols[1]:
            submit_btn = st.form_submit_button(
                "▲",
                use_container_width=True,
                disabled=st.session_state.streaming or st.session_state.ingest_status == "running",
            )


_is_busy = st.session_state.streaming or st.session_state.ingest_status == "running"
if submit_btn and question and question.strip() and not _is_busy:
    history = [
        msg
        for h in st.session_state.chat_history
        for msg in [
            {"role": "user", "content": h["question"]},
            {"role": "assistant", "content": h["answer"]},
        ]
    ]
    st.session_state.pending_question = question
    st.session_state.streaming = True
    st.session_state.stream_history = history
    st.session_state.input_key_tracker += 1
    st.rerun()
