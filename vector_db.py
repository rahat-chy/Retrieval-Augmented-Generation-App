import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, FilterSelector,
)

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Thin wrapper around QdrantClient for the RAG pipeline's vector store."""

    def __init__(self, url=None, collection="docs", dim=384):
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        """Connect to Qdrant and create the COSINE collection if it doesn't exist."""
        logger.debug("Connecting to Qdrant at %s", url)
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            logger.info("Creating Qdrant collection '%s' (dim=%d, COSINE)", collection, dim)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            logger.debug("Qdrant collection '%s' already exists", collection)

    def upsert(self, ids, vectors, payloads):
        """Insert or update points with their vectors and payloads in the collection."""
        logger.info("Upserting %d points into '%s'", len(ids), self.collection)
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)
        logger.info("Upsert complete: %d points", len(ids))


    def search(self, query_vector, top_k: int = 5):
        """Search top-k similar vectors and return deduplicated parent contexts with child source refs."""
        logger.info("Searching '%s' top_k=%d", self.collection, top_k)
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points
        contexts = []
        source_refs: list[dict] = []
        seen_parents: dict[str, int] = {}  # parent_text -> index in source_refs

        for r in results:
            payload = getattr(r, "payload", None) or {}
            context = payload.get("parent_text") or payload.get("text", "")
            child_ref = {
                "source": payload.get("source", ""),
                "page_num": payload.get("page_num", 1),
                "excerpt": payload.get("text", ""),
            }
            if not context:
                continue
            if context not in seen_parents:
                contexts.append(context)
                seen_parents[context] = len(source_refs)
                source_refs.append({
                    **child_ref,
                    "context_preview": context,
                    "siblings": [],
                })
            else:
                source_refs[seen_parents[context]]["siblings"].append({
                    **child_ref,
                    "context_preview": context,
                })

        logger.info("Search returned %d unique parent contexts", len(contexts))
        return {"contexts": contexts, "source_refs": source_refs}

    def delete_by_source(self, source_id: str):
        """Delete all Qdrant points whose payload source field equals source_id."""
        logger.info("Deleting all points with source='%s' from '%s'", source_id, self.collection)
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_id))])
            ),
        )