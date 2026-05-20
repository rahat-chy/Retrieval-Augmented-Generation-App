import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams, SparseVector,
    Filter, FieldCondition, MatchValue, FilterSelector,
    Prefetch, FusionQuery, Fusion,
)

logger = logging.getLogger(__name__)


class QdrantStorage:
    """Thin wrapper around QdrantClient for the RAG pipeline's vector store."""

    def __init__(self, url=None, collection="docs", dim=384):
        """Connect to Qdrant and ensure the collection exists with hybrid (dense+sparse) config."""
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        logger.debug("Connecting to Qdrant at %s", url)
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self._ensure_collection(dim)

    def _ensure_collection(self, dim: int):
        """Create the collection with named dense+sparse vectors if it doesn't exist."""
        if self.client.collection_exists(self.collection):
            logger.debug("Qdrant collection '%s' already exists", self.collection)
            return
        logger.info("Creating Qdrant collection '%s' (dim=%d, COSINE + BM25 sparse)", self.collection, dim)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
        )

    def upsert(self, ids, dense_vectors, sparse_vectors, payloads):
        """Insert or update points with dense+sparse vectors and payloads in the collection."""
        logger.info("Upserting %d points into '%s'", len(ids), self.collection)
        points = [
            PointStruct(
                id=ids[i],
                vector={
                    "dense": dense_vectors[i],
                    "sparse": SparseVector(
                        indices=sparse_vectors[i]["indices"],
                        values=sparse_vectors[i]["values"],
                    ),
                },
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)
        logger.info("Upsert complete: %d points", len(ids))

    def search(self, query_dense, query_sparse: dict, top_k: int = 5):
        """Hybrid search using RRF fusion of dense cosine and sparse BM25 results."""
        logger.info("Hybrid searching '%s' top_k=%d", self.collection, top_k)
        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=query_dense, using="dense", limit=top_k * 2),
                Prefetch(
                    query=SparseVector(
                        indices=query_sparse["indices"],
                        values=query_sparse["values"],
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points

        contexts = []
        source_refs: list[dict] = []
        seen_parents: dict[str, int] = {}

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

        logger.info("Hybrid search returned %d unique parent contexts", len(contexts))
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
