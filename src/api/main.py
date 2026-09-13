import json
from fastapi import FastAPI, HTTPException
import instructor
from openai import OpenAI
import redis
from src.api.schema import QueryRequest, DocumentInsightResponse
from src.retrieval.hybrid_reranker import HybridRetriever
from config.settings import settings

app = FastAPI(title=settings.PROJECT_NAME)

redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)

client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON
)

# Initialize retriever using dynamic Pinecone metadata lookup
retriever = HybridRetriever()


@app.post("/api/v1/query", response_model=DocumentInsightResponse)
async def query_document(request: QueryRequest):
    cache_key = f"query:{request.query}"
    cached_data = redis_client.get(cache_key)
    if cached_data:
        # Validate JSON directly into response model
        return DocumentInsightResponse.model_validate_json(cached_data)

    try:
        context_docs = retriever.retrieve(request.query, top_k=5)
        context_text = "\n---\n".join([doc["text"] for doc in context_docs])
        context_ids = [doc["id"] for doc in context_docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval system failure: {str(e)}")

    system_prompt = (
        "You are an enterprise financial specialist. Parse the retrieved context "
        "and return structured JSON matching the declared response schema precisely."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {request.query}"

    response: DocumentInsightResponse = client.chat.completions.create(
        model="mistral",
        response_model=DocumentInsightResponse,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_retries=3
    )

    response.source_chunks = context_ids
    redis_client.setex(cache_key, 3600, response.model_dump_json())

    return response