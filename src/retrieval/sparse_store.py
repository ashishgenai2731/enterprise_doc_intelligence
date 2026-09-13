from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class SparseStoreHandler:
    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus  # Expected format: [{'id': str, 'text': str}]
        tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query(self, query_text: str, top_k: int = 20) -> List[str]:
        """Executes exact keyword matching search using standard BM25 ranking."""
        tokenized_query = query_text.lower().split()
        top_indices = self.bm25.get_top_n(tokenized_query, range(len(self.corpus)), n=top_k)
        return [self.corpus[idx]["id"] for idx in top_indices]