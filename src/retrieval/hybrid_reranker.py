import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from src.retrieval.vector_store import VectorStoreHandler
from src.retrieval.sparse_store import SparseStoreHandler, load_corpus
from src.retrieval.query_filters import matches
from config.settings import settings

logger = logging.getLogger(__name__)

# Global singleton to prevent reloading model weights on every query/node
_reranker_instance = None

# BM25 indexes for filtered sub-corpora (one 10-K is ~1,000 chunks, built in well under a second)
_MAX_CACHED_SUBSETS = 64


def _get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(settings.RERANKER_MODEL_NAME)
    return _reranker_instance


def _filter_key(filters: Dict[str, Any]) -> tuple:
    return tuple(sorted((field, tuple(sorted(cond["$in"]))) for field, cond in filters.items()))


class HybridRetriever:
    def __init__(self, doc_corpus: List[Dict[str, Any]] = None, vector_store=None):
        self.vector_store = vector_store or VectorStoreHandler()
        # Default to the on-disk corpus written at ingestion time, so BM25 is live in the app
        self.doc_corpus = doc_corpus if doc_corpus is not None else load_corpus(settings.CORPUS_PATH)
        self.id_to_text = {doc["id"]: doc["text"] for doc in self.doc_corpus}
        # Built on first use: the full corpus is ~100k chunks and most questions are filtered
        self.sparse_store: Optional[SparseStoreHandler] = None
        self._subset_stores: Dict[tuple, SparseStoreHandler] = {}
        if not self.doc_corpus:
            logger.warning(
                "No BM25 corpus at %s - running dense-only. Run scripts/export_corpus.py.",
                settings.CORPUS_PATH,
            )

    @property
    def reranker(self) -> CrossEncoder:
        return _get_reranker()

    def _sparse_for(self, filters: Dict[str, Any]) -> Optional[SparseStoreHandler]:
        if not self.doc_corpus:
            return None
        if not filters:
            if self.sparse_store is None:
                self.sparse_store = SparseStoreHandler(corpus=self.doc_corpus)
            return self.sparse_store

        key = _filter_key(filters)
        if key not in self._subset_stores:
            subset = [doc for doc in self.doc_corpus if matches(doc.get("metadata", {}), filters)]
            if not subset:
                return None
            if len(self._subset_stores) >= _MAX_CACHED_SUBSETS:
                self._subset_stores.pop(next(iter(self._subset_stores)))
            self._subset_stores[key] = SparseStoreHandler(corpus=subset)
        return self._subset_stores[key]

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

    def retrieve(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # 1. Fetch Dense Vector Candidates from Pinecone, inside the filtered filings if any
        dense_matches = self.vector_store.query(query, top_k=20, filter=filters) if filters \
            else self.vector_store.query(query, top_k=20)
        if filters and not dense_matches:
            # A filter that matches nothing (e.g. an index without metadata) must not blank the answer
            logger.warning("Filter %s matched no vectors - searching unfiltered", filters)
            return self.retrieve(query, top_k=top_k)
        dense_ids = [match["id"] for match in dense_matches]

        # Extract text from local corpus first; fall back to Pinecone metadata
        pinecone_text = {m["id"]: m.get("metadata", {}).get("text", "") for m in dense_matches
                         if m["id"] not in self.id_to_text}
        id_to_text = {**pinecone_text, **{i: self.id_to_text[i] for i in dense_ids if i in self.id_to_text}}

        # 2. Fetch Candidates from Sparse BM25 Index over the same filings
        sparse_store = self._sparse_for(filters or {})
        sparse_ids = sparse_store.query(query, top_k=20) if sparse_store else []
        id_to_text.update({i: self.id_to_text[i] for i in sparse_ids})

        # 3. Reciprocal Rank Fusion (RRF)
        fused_ids = self._reciprocal_rank_fusion(dense_ids, sparse_ids)[:20]

        # 4. Filter IDs to only those with valid text payloads to preserve 1:1 array alignment
        eval_ids = [doc_id for doc_id in fused_ids if id_to_text.get(doc_id)]

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
