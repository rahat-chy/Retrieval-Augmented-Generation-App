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

        return {"contexts": contexts, "source_refs": source_refs}

    def delete_by_source(self, source_id: str):
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_id))])
            ),
        )