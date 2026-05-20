from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from lib.state import IngestState
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage


def load_and_chunk_node(state: IngestState) -> dict:
    chunks = load_and_chunk_pdf(state["pdf_path"])
    return {"chunks": chunks}


def embed_and_upsert_node(state: IngestState) -> dict:
    chunks = state["chunks"]
    vecs = embed_texts([c["text"] for c in chunks])
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
    QdrantStorage().upsert(ids, vecs, payload)
    return {"ingested": len(chunks)}


def build_ingest_graph(checkpointer: MemorySaver):
    g = StateGraph(IngestState)
    g.add_node("load_and_chunk", load_and_chunk_node)
    g.add_node("embed_and_upsert", embed_and_upsert_node)
    g.add_edge(START, "load_and_chunk")
    g.add_edge("load_and_chunk", "embed_and_upsert")
    g.add_edge("embed_and_upsert", END)
    return g.compile(checkpointer=checkpointer)
