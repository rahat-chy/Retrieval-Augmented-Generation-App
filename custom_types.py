import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[dict]  # each: {id, text, parent_text}
    source_id: str | None = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int