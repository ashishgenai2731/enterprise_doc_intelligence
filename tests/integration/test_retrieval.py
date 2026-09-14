import pytest
from src.retrieval.hybrid_reranker import HybridRetriever


def test_retrieval_flow():
    """
    Integration test validating that HybridRetriever queries Pinecone + Cross-Encoder
    and returns properly structured chunks.
    """
    retriever = HybridRetriever()
    results = retriever.retrieve(query="What were the total operating expenses?", top_k=2)

    # 1. Assert results are returned from the index
    assert isinstance(results, list)
    assert len(results) > 0

    # 2. Validate structure and fields of the top result
    top_doc = results[0]
    assert "id" in top_doc
    assert "text" in top_doc
    assert ("score" in top_doc or "rerank_score" in top_doc)

    # 3. Assert content validity
    assert isinstance(top_doc["id"], str)
    assert len(top_doc["text"].strip()) > 0