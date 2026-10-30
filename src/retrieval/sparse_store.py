import json
import re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens; strips punctuation so 'expenses?' matches 'expenses'."""
    return re.findall(r"\w+", text.lower())


def load_corpus(path: str) -> List[Dict[str, Any]]:
    """Reads the local chunk corpus (one {'id', 'text'} JSON object per line)."""
    corpus_file = Path(path)
    if not corpus_file.exists():
        return []
    with corpus_file.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_to_corpus(path: str, chunks: List[Dict[str, Any]]):
    """Appends chunks to the local corpus, skipping IDs that are already present."""
    corpus_file = Path(path)
    corpus_file.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = {doc["id"] for doc in load_corpus(path)}
    with corpus_file.open("a") as f:
        for chunk in chunks:
            if chunk["id"] not in existing_ids:
                f.write(json.dumps({"id": chunk["id"], "text": chunk["text"]}) + "\n")


class SparseStoreHandler:
    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus  # Expected format: [{'id': str, 'text': str}]
        tokenized_corpus = [tokenize(doc["text"]) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query(self, query_text: str, top_k: int = 20) -> List[str]:
        """Executes exact keyword matching search using standard BM25 ranking."""
        tokenized_query = tokenize(query_text)
        top_indices = self.bm25.get_top_n(tokenized_query, range(len(self.corpus)), n=top_k)
        return [self.corpus[idx]["id"] for idx in top_indices]
