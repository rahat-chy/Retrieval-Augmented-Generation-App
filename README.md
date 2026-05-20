# RAG App

A fully local Retrieval-Augmented Generation (RAG) pipeline. Ingest PDFs, ask questions, get grounded answers — no cloud APIs, no data leaves your machine.

---

## Features

- **Hybrid retrieval** — dense cosine (sentence-transformers) + sparse BM25 (fastembed) fused with Reciprocal Rank Fusion (RRF)
- **Semantic chunking** — `SemanticSplitterNodeParser` from llama-index produces context-aware chunks instead of fixed-size splits
- **Parent-document retrieval** — child chunks indexed for precision; parent context returned to the LLM for richer answers
- **Image understanding** — non-decorative PDF images described by `llava` and injected as text context
- **Adaptive query pipeline** — intent classification, relevance grading, automatic query rewriting (up to 2 rounds), streaming SSE output
- **Job runner** — SQLite-backed async jobs with status polling and one-click retry for failed ingests
- **Streamlit UI** — dark-themed chat interface with live streaming, source citations, and document management
- **100% local** — Qdrant + Ollama + sentence-transformers, zero external API calls

---

## Architecture

### Ingest Pipeline

```
PDF file
  │
  ▼
PDFReader (llama-index)          ← extract text per page
  │
  ├──► pymupdf image extraction
  │         │
  │         ▼
  │    llava (ollama)             ← describe non-decorative images (≥100×100 px)
  │         │
  │         └──► image text appended to corpus
  │
  ▼
SemanticSplitterNodeParser        ← semantic chunk boundaries (95th pct threshold)
  │
  ▼
Parent grouping (every 4 child chunks → 1 parent)
  │
  ├──► all-MiniLM-L6-v2           ← dense vectors (dim 384)
  └──► Qdrant/bm25 (fastembed)    ← sparse BM25 vectors
            │
            ▼
       Qdrant upsert               ← collection "docs", COSINE + sparse index
            │
            ▼
    SQLite: job → completed, document registered
```

### Query Pipeline (LangGraph)

```
User question
      │
      ▼
 classify_intent ──── llama3.2 ──► "chitchat" ──► chitchat_node ──► stream tokens ──► END
      │
     "rag"
      │
      ▼
  retrieve_node
      ├── all-MiniLM-L6-v2  ← dense query embed
      └── Qdrant/bm25        ← sparse query embed
                │
                ▼
      Qdrant hybrid search (RRF fusion, top_k×2 prefetch each)
                │
                ▼
       grade_docs_node  ──── llama3.2 grades each chunk in parallel
                │
         ┌──────┴──────────────────────────┐
    relevant?                         no relevant docs
         │                            + rewrites < 2
         ▼                                 │
   generate_node                   rewrite_query_node
   (streaming SSE)                   (llama3.2 rewrites)
         │                                 │
         ▼                                 └──► retrieve_node  (loop, max 2×)
  answer + source refs
         │
         ▼
   SQLite: chat_history saved
```

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  :8501                       │
│   File upload ──► POST /ingest          Chat ──► POST /query/stream │
└────────────────────────┬───────────────────────────────────────-┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────────────┐
│                   FastAPI + Uvicorn  :8000                       │
│  /ingest  /query/stream  /jobs/{id}  /documents  /history       │
└──────┬──────────────────────┬───────────────────────────────────┘
       │                      │
       ▼                      ▼
  ingest_graph           query_graph           ← LangGraph state machines
  (LangGraph)            (LangGraph)
       │                      │
       ▼                      ▼
  data_loader.py         vector_db.py          ← shared helpers
       │                      │
       ▼                      ▼
  Qdrant  :6333          SQLite  jobs.db        ← persistence
  ollama  :11434         (jobs · docs · chat)
```

---

## File Structure

```
RAGApp/
├── main.py                FastAPI app — endpoints + background job wiring
├── data_loader.py         PDF load, semantic chunk, image description, embed
├── vector_db.py           Qdrant wrapper — hybrid upsert + RRF search
├── job_runner.py          SQLite CRUD for jobs, documents, chat history
├── ui.py                  Streamlit frontend
├── graphs/
│   ├── ingest_graph.py    LangGraph: load_and_chunk → embed_and_upsert
│   └── query_graph.py     LangGraph: classify → retrieve → grade → [rewrite] → generate
├── lib/
│   └── state.py           TypedDict state definitions for LangGraph nodes
└── pyproject.toml         uv-managed dependencies
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Runtime |
| [uv](https://docs.astral.sh/uv/) | latest | Package manager |
| [Docker](https://docs.docker.com/get-docker/) | any | Run Qdrant |
| [Ollama](https://ollama.com/download) | latest | Local LLM inference |

---

## Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/RAGApp.git
cd RAGApp
```

### 2. Install uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install Python dependencies

```bash
uv sync
```

Creates `.venv` and installs all packages from `pyproject.toml`. No manual `pip install` needed.

### 4. Start Qdrant via Docker

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

Qdrant dashboard: `http://localhost:6333/dashboard`

### 5. Pull Ollama models

Download Ollama from [https://ollama.com/download](https://ollama.com/download), then:

```bash
ollama pull llama3.2   # LLM — answer generation, intent classification, grading, rewriting
ollama pull llava      # Vision — PDF image descriptions (skip if PDFs are text-only)
```

---

## Running the App

You need **two terminals** (three if Ollama isn't running as a system service).

### Terminal 1 — FastAPI backend

```bash
uv run uvicorn main:app --reload
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

### Terminal 2 — Streamlit UI

```bash
uv run streamlit run ui.py
```

- UI: `http://localhost:8501`

### Terminal 3 — Ollama (if not running as a service)

```bash
ollama serve
```

---

## Usage

### Streamlit UI

1. Open `http://localhost:8501`
2. **Ingest** — drag-and-drop a `.pdf`, click **Ingest**, wait for the completion banner
3. **Chat** — type a question, press Enter, watch the answer stream in real time
4. **Sources** — expand the "Sources" section under any answer to see matched pages and excerpts
5. **Delete** — click the trash icon in the Ingested Documents table to remove a document and all its vectors

### REST API

**Ingest a PDF:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "/absolute/path/to/document.pdf", "source_id": "my-doc"}'
# → {"job_id": "...", "status": "running"}
```

**Poll job status:**
```bash
curl http://localhost:8000/jobs/<job_id>
# → {"status": "completed", "result": {"ingested": 42}, ...}
```

**Ask a question (streaming SSE):**
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main finding?", "top_k": 5}'
```

**List documents:**
```bash
curl http://localhost:8000/documents
```

**Delete a document:**
```bash
curl -X DELETE http://localhost:8000/documents/<doc_id>
```

**Retry a failed ingest:**
```bash
curl -X POST http://localhost:8000/jobs/<job_id>/retry
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Start async ingest job for a PDF |
| `POST` | `/query/stream` | Stream answer tokens via SSE |
| `GET` | `/jobs/{job_id}` | Get job status, params, result |
| `POST` | `/jobs/{job_id}/retry` | Retry a failed job |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{doc_id}` | Delete document and its Qdrant vectors |
| `GET` | `/history` | Get full chat history |

---

## Configuration

| Setting | Env var / location | Default |
|---------|-------------------|---------|
| Qdrant URL | `QDRANT_URL` | `http://localhost:6333` |
| SQLite DB path | `DB_PATH` | `jobs.db` |
| API base (UI) | `API_BASE` | `http://localhost:8000` |
| Embed model | `data_loader.py` → `EMBED_MODEL` | `all-MiniLM-L6-v2` |
| Vector dim | `data_loader.py` → `EMBED_DIM` | `384` |
| Parent group size | `data_loader.py` → `PARENT_GROUP_SIZE` | `4` |
| Max query rewrites | `graphs/query_graph.py` → `MAX_REWRITES` | `2` |

Override via `.env` in the project root:

```env
QDRANT_URL=http://localhost:6333
DB_PATH=jobs.db
API_BASE=http://localhost:8000
```

---

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| API server | FastAPI + Uvicorn |
| UI | Streamlit |
| Pipeline orchestration | LangGraph |
| PDF parsing | llama-index `PDFReader` |
| Semantic chunking | llama-index `SemanticSplitterNodeParser` |
| Image extraction | PyMuPDF (fitz) |
| Dense embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Sparse embeddings | fastembed `Qdrant/bm25` |
| Vector store | Qdrant (local Docker) |
| LLM + vision | Ollama (`llama3.2`, `llava`) |
| Job persistence | SQLite |

---

## Troubleshooting

**Qdrant connection refused**
```bash
docker ps | grep qdrant     # check container is running
docker start qdrant         # restart if stopped
```

**Ollama model not found**
```bash
ollama list                 # see installed models
ollama pull llama3.2
ollama pull llava
```

**Ingest job fails — PDF not found**
The `pdf_path` must be an absolute path accessible by the server process. The Streamlit UI handles this automatically by writing uploads to a temp file.

**llava slow on image-heavy PDFs**
Images smaller than 100×100 px and images classified as "decorative" are automatically skipped, keeping ingestion fast.

**Port already in use**
```bash
# Change API port
uv run uvicorn main:app --reload --port 8001

# Change UI port (update API_BASE env var too)
uv run streamlit run ui.py --server.port 8502
```

---

## License

MIT
