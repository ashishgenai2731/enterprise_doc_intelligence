import pytest
import sys
from pathlib import Path

from src.ingestion.chunker import TextChunker


# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_chunker_splitting():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    sample_text = "Data point " * 40
    chunks = chunker.split(sample_text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
