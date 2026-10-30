"""
Rebuilds the local BM25 corpus (data/corpus.jsonl) from the chunks already in Pinecone.

Use this when the index was filled before ingest_doc.py started saving the corpus.
Reading from Pinecone keeps the chunk IDs identical on both sides, which Reciprocal
Rank Fusion needs to merge dense and sparse hits for the same chunk.
Read-only against Pinecone.
"""
from pinecone import Pinecone
from config.settings import settings
from src.retrieval.sparse_store import append_to_corpus


def export_corpus():
    index = Pinecone(api_key=settings.PINECONE_API_KEY).Index(settings.PINECONE_INDEX_NAME)

    chunks = []
    for page in index.list():  # paginated: each page holds up to 100 ListItem(id=...)
        fetched = index.fetch(ids=[item.id for item in page])
        for vec_id, vector in fetched.vectors.items():
            text = (vector.metadata or {}).get("text", "")
            if text:
                chunks.append({"id": vec_id, "text": text})

    append_to_corpus(settings.CORPUS_PATH, chunks)
    print(f"Exported {len(chunks)} chunks to {settings.CORPUS_PATH}")


if __name__ == "__main__":
    export_corpus()
