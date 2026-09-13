from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from config.settings import settings


class VectorStoreHandler:
    def __init__(self):
        self.embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        # Auto-provision serverless index if it doesn't exist
        active_indexes = [idx.name for idx in self.pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in active_indexes:
            self.pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1024,  # Matches bge-large-en-v1.5
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Embeds text chunks and upserts them into Pinecone with metadata."""
        vectors_to_upsert = []
        for item in chunks:
            vector_embedding = self.embed_model.encode(item["text"]).tolist()
            metadata = item.get("metadata", {})
            metadata["text"] = item["text"]

            vectors_to_upsert.append({
                "id": item["id"],
                "values": vector_embedding,
                "metadata": metadata
            })

        self.index.upsert(vectors=vectors_to_upsert)

    def query(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Generates dense vector representation and performs top-K lookup in Pinecone."""
        query_vector = self.embed_model.encode(query_text).tolist()
        response = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return [{"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})} for m in
                response["matches"]]
        # return response.get("matches", [])