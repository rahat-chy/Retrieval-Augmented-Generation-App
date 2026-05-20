import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    """Request/response model carrying chunked document data with an optional source ID."""
    chunks: list[dict]  # each: {id, text, parent_text}
    source_id: str | None = None

class RAGUpsertResult(pydantic.BaseModel):
    """Result model reporting how many chunks were upserted into Qdrant."""
    ingested: int

class RAGSearchResult(pydantic.BaseModel):
    """Result model for a Qdrant similarity search containing contexts and source names."""
    contexts: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    """Result model for a full RAG query containing the answer, sources, and context count."""
    answer: str
    sources: list[str]
    num_contexts: int
