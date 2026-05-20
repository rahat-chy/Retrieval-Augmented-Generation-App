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
    source_refs: list[dict]          # [{source, page_num, excerpt}], parallel to contexts
    relevant_contexts: list[str]
    answer: str
    rewrite_count: int
