import pytest
from src.ingestion.chunker import TextChunker


def test_chunker_splitting():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    sample_text = "Data point " * 40
    chunks = chunker.split(sample_text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
