from typing import TypedDict


class IngestState(TypedDict):
    pdf_path: str
    source_id: str
    chunks: list[dict]
    ingested: int


class QueryState(TypedDict):
    question: str
    original_question: str
    top_k: int
    history: list[dict]
    intent: str  # "rag" or "chitchat"
    contexts: list[str]
    sources: list[str]
    relevant_contexts: list[str]
    relevant_sources: list[str]
    answer: str
    rewrite_count: int
