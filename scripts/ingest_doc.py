import uuid
from pathlib import Path
from src.ingestion.parser import DocumentIngestor
from src.retrieval.vector_store import VectorStoreHandler


def ingest_pdf(pdf_path: str):
    print(f"[1/4] Starting PDF extraction: {pdf_path}")
    ingestor = DocumentIngestor()
    raw_text = ingestor.parse_pdf(pdf_path)
    chunks = ingestor.create_chunks(raw_text)
    print(f"[2/4] Successfully chunked document into {len(chunks)} segments.")

    # Format payload with unique IDs and metadata for Pinecone
    formatted_chunks = []
    doc_name = Path(pdf_path).name
    for i, chunk_text in enumerate(chunks):
        formatted_chunks.append({
            "id": f"{doc_name}-chunk-{i}-{uuid.uuid4().hex[:6]}",
            "text": chunk_text,
            "metadata": {
                "source": doc_name,
                "chunk_index": i
            }
        })

    print(f"[3/4] Generating embeddings and upserting to Pinecone...")
    vector_store = VectorStoreHandler()
    vector_store.upsert_chunks(formatted_chunks)
    print(f"Successfully indexed {len(formatted_chunks)} vectors in Pinecone!")

    # Verification query
    print("\n[4/4] Running sanity check query against index...")
    sample_results = vector_store.query("What are the total operating expenses?", top_k=2)
    print(f"Retrieved {len(sample_results)} top matches from Pinecone:")
    for idx, match in enumerate(sample_results):
        print(
            f"  Match #{idx + 1} [Score: {match['score']:.4f}]: {match['metadata'].get('text', '')[:120]}...")


if __name__ == "__main__":
    sample_pdf_path = "/Users/ashishkumar/Downloads/Form10K.pdf"
    ingest_pdf(sample_pdf_path)