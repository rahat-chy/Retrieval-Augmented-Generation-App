import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from lib.state import IngestState
from data_loader import load_and_chunk_pdf, embed_texts, bm25_embed_texts
from vector_db import QdrantStorage

logger = logging.getLogger(__name__)


def load_and_chunk_node(state: IngestState) -> dict:
    """LangGraph node: load the PDF and split it into semantic child chunks."""
    logger.info("Node load_and_chunk: pdf_path=%s", state["pdf_path"])
    chunks = load_and_chunk_pdf(state["pdf_path"])
    logger.info("Node load_and_chunk complete: %d chunks", len(chunks))
    return {"chunks": chunks}


def embed_and_upsert_node(state: IngestState) -> dict:
    """LangGraph node: embed all chunks (dense+sparse) and upsert into Qdrant."""
    chunks = state["chunks"]
    texts = [c["text"] for c in chunks]
    logger.info("Node embed_and_upsert: embedding %d chunks", len(chunks))
    dense_vecs = embed_texts(texts)
    sparse_vecs = bm25_embed_texts(texts)
    ids = [c["id"] for c in chunks]
    payload = [
        {
            "source": state["source_id"],
            "text": c["text"],
            "parent_text": c["parent_text"],
            "page_num": c.get("page_num", 1),
        }
        for c in chunks
    ]
    QdrantStorage().upsert(ids, dense_vecs, sparse_vecs, payload)
    logger.info("Node embed_and_upsert complete: %d vectors stored", len(chunks))
    return {"ingested": len(chunks)}


def build_ingest_graph(checkpointer: MemorySaver):
    """Build and compile the two-node ingest graph (load+chunk → embed+upsert) with checkpointing."""
    g = StateGraph(IngestState)
    g.add_node("load_and_chunk", load_and_chunk_node)
    g.add_node("embed_and_upsert", embed_and_upsert_node)
    g.add_edge(START, "load_and_chunk")
    g.add_edge("load_and_chunk", "embed_and_upsert")
    g.add_edge("embed_and_upsert", END)
    return g.compile(checkpointer=checkpointer)
