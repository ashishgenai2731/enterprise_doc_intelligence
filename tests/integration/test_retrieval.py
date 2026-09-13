import pytest
from src.retrieval.hybrid_reranker import HybridRetriever


def test_retrieval_flow():
    corpus = [
        {"id": "doc_1", "text": "Q3 gross revenue totaled 10 million dollars."},
        {"id": "doc_2", "text": "Company hiring expanded by 15 percent."}
    ]

    retriever = HybridRetriever(doc_corpus=corpus)
    results = retriever.retrieve("What was the gross revenue?", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "doc_1"