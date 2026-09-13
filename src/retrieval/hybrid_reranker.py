from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.retrieval.vector_store import VectorStoreHandler
from src.retrieval.sparse_store import SparseStoreHandler
from config.settings import settings


class HybridRetriever:
    def __init__(self, doc_corpus: List[Dict[str, Any]] = None):
        self.vector_store = VectorStoreHandler()
        self.doc_corpus = doc_corpus or []
        self.sparse_store = SparseStoreHandler(corpus=self.doc_corpus) if self.doc_corpus else None
        self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)

    @staticmethod
    def _reciprocal_rank_fusion(dense_ranks: List[str], sparse_ranks: List[str],
                                k: int = 60) -> List[str]:
        rrf_scores = {}
        for rank, doc_id in enumerate(dense_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Fetch Dense Vector Candidates from Pinecone
        dense_matches = self.vector_store.query(query, top_k=20)
        dense_ids = [match["id"] for match in dense_matches]

        # Extract text from local corpus first; fall back to Pinecone metadata
        id_to_text = {doc['id']: doc['text'] for doc in self.doc_corpus}
        for match in dense_matches:
            if match["id"] not in id_to_text:
                text = match.get("metadata", {}).get("text", "")
                if text:
                    id_to_text[match["id"]] = text

        # 2. Fetch Candidates from Sparse BM25 Index (if corpus initialized)
        sparse_ids = self.sparse_store.query(query, top_k=20) if self.sparse_store else []

        # 3. Reciprocal Rank Fusion (RRF)
        fused_ids = self._reciprocal_rank_fusion(dense_ids, sparse_ids)[:20]

        # 4. Filter IDs to only those with valid text payloads to preserve 1:1 array alignment
        eval_ids = [doc_id for doc_id in fused_ids if doc_id in id_to_text]

        if not eval_ids:
            return []

        # 5. Cross-Encoder Fine Reranking
        pairs = [[query, id_to_text[doc_id]] for doc_id in eval_ids]
        scores = self.reranker.predict(pairs)

        # Pair scores strictly with eval_ids
        ranked_results = sorted(zip(scores, eval_ids), key=lambda x: x[0], reverse=True)

        final_docs = []
        for score, doc_id in ranked_results[:top_k]:
            final_docs.append({
                "id": doc_id,
                "text": id_to_text[doc_id],
                "score": float(score)
            })

        return final_docs