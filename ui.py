import json
import os
import tempfile
import uuid
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
    all_chunks = []
    for ref in refs:
        all_chunks.append(ref)
        all_chunks.extend(ref.get("siblings", []))

    for chunk in all_chunks:
        st.markdown(f"**{chunk['source']}** · page {chunk['page_num']}")
        excerpt = chunk.get("excerpt", "")
        if excerpt:
            st.caption("Matched chunk:")
            st.text_area("", value=excerpt, height=120, disabled=True, key=uuid.uuid4().hex)
        full_ctx = chunk.get("context_preview", "")
        if full_ctx and full_ctx != excerpt:
            with st.expander("Full context"):
                st.text_area("", value=full_ctx, height=200, disabled=True, key=uuid.uuid4().hex)


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


# ── Ingest ──────────────────────────────────────────────────────────────────

st.header("Ingest PDF")

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
    if st.button("Ingest", disabled=ingest_disabled or uploaded_file is None, use_container_width=True):
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
    if st.button("Retry Ingest", disabled=retry_disabled, use_container_width=True):
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
    st.info("⏳ Ingesting...")

_poll_ingest()

if st.session_state.ingest_last_msg:
    kind, msg = st.session_state.ingest_last_msg
    if kind == "success":
        st.success(msg)
    else:
        st.error(msg)


# ── Documents ────────────────────────────────────────────────────────────────

st.divider()
st.header("Ingested Documents")

if not st.session_state.documents_loaded:
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=5)
        resp.raise_for_status()
        st.session_state.documents = resp.json()
        st.session_state.documents_loaded = True
    except Exception:
        pass

if not st.session_state.documents:
    st.caption("No documents ingested yet.")
else:
    header_cols = st.columns([5, 2, 2, 1])
    header_cols[0].caption("**Name**")
    header_cols[1].caption("**Chunks**")
    header_cols[2].caption("**Ingested**")
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
                meta_parts = []
                if msg.get("rewrites", 0) > 0:
                    meta_parts.append(f"🔄 Query rewritten {msg['rewrites']}x")
                if meta_parts:
                    st.caption(" · ".join(meta_parts))
                source_refs = msg.get("source_refs", [])
                if source_refs:
                    with st.expander("Sources"):
                        _render_source_refs(source_refs)
                elif msg.get("sources"):
                    with st.expander("Sources"):
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
                    with st.expander("Sources"):
                        _render_source_refs(_source_refs)
                elif _res.get("sources"):
                    with st.expander("Sources"):
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
            submit_btn = st.form_submit_button("▲", use_container_width=True, disabled=st.session_state.streaming or st.session_state.ingest_status == "running")


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
