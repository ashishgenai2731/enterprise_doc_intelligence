"""
Parses, chunks, embeds and indexes every filing in data/edgar/filings.jsonl.

Each chunk starts with a '[Company (TICKER) 10-K, fiscal YEAR]' line, so the reranker and the
LLM can see which filing a table row came from, and carries ticker / fiscal_year metadata for
filtered search. Filings already in the BM25 corpus are skipped, so an interrupted run resumes.

Usage:
    python -m scripts.ingest_edgar                       # all filings in the manifest
    python -m scripts.ingest_edgar --tickers AAPL MSFT
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from config.settings import settings
from src.ingestion.html_parser import html_to_text
from src.ingestion.chunker import TextChunker
from src.retrieval.sparse_store import append_to_corpus, corpus_ids
from src.retrieval.vector_store import VectorStoreHandler


def build_chunks(filing: Dict[str, Any], text: str, chunker: TextChunker) -> List[Dict[str, Any]]:
    header = f"[{filing['company']} ({filing['ticker']}) 10-K, fiscal {filing['fiscal_year']}]"
    return [
        {
            "id": f"{filing['ticker']}-FY{filing['fiscal_year']}-chunk-{i}",
            "text": f"{header}\n{chunk}",
            "metadata": {
                "ticker": filing["ticker"],
                "company": filing["company"],
                "fiscal_year": filing["fiscal_year"],
                "accession": filing["accession"],
                "source": filing["url"],
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(chunker.split(text))
    ]


def ingest(filings: List[Dict[str, Any]]):
    chunker = TextChunker()
    vector_store = VectorStoreHandler()
    indexed = corpus_ids(settings.CORPUS_PATH)
    total = 0
    for n, filing in enumerate(filings, 1):
        first_id = f"{filing['ticker']}-FY{filing['fiscal_year']}-chunk-0"
        if first_id in indexed:
            print(f"[{n}/{len(filings)}] {filing['ticker']} FY{filing['fiscal_year']}: already indexed", flush=True)
            continue
        start = time.perf_counter()
        text = html_to_text(Path(filing["path"]).read_bytes())
        chunks = build_chunks(filing, text, chunker)
        vector_store.upsert_chunks(chunks)
        # A re-parse can produce fewer chunks; old IDs past the new end would otherwise linger
        prefix = f"{filing['ticker']}-FY{filing['fiscal_year']}-chunk-"
        new_ids = {c["id"] for c in chunks}
        stale = [item.id for page in vector_store.index.list(prefix=prefix) for item in page
                 if item.id not in new_ids]
        if stale:
            vector_store.index.delete(ids=stale)
        # Corpus is written after the upsert succeeds, so it doubles as the resume checkpoint
        append_to_corpus(settings.CORPUS_PATH, chunks, existing_ids=indexed)
        total += len(chunks)
        print(f"[{n}/{len(filings)}] {filing['ticker']} FY{filing['fiscal_year']}: "
              f"{len(chunks)} chunks in {time.perf_counter() - start:.0f}s", flush=True)
    print(f"\nIndexed {total} new chunks into '{settings.PINECONE_INDEX_NAME}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(settings.FILINGS_MANIFEST).read_text().splitlines() if line.strip()]
    ingest([r for r in rows if not args.tickers or r["ticker"] in args.tickers])
