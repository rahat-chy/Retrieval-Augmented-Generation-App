from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, FilterSelector,
)

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=384):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)


    def search(self, query_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points
        contexts = []
        sources = set()
        seen_parents: set[str] = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            # return parent_text for richer LLM context; dedup siblings sharing same parent
            context = payload.get("parent_text") or payload.get("text", "")
            source = payload.get("source", "")
            if context and context not in seen_parents:
                contexts.append(context)
                seen_parents.add(context)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

    def delete_by_source(self, source_id: str):
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_id))])
            ),
        )