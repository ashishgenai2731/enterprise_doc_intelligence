import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import instructor
from openai import OpenAI
import redis
from src.api.schema import QueryRequest, DocumentInsightResponse
from src.retrieval.hybrid_reranker import HybridRetriever
from config.settings import settings
from src.agents.graph import build_financial_agent_graph

logger = logging.getLogger("uvicorn")

app = FastAPI(title=settings.PROJECT_NAME)

# Initialize Redis client
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    socket_timeout=2
)

# Initialize Instructor with local Ollama client
client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON
)

# Initialize retriever & agent state graph
retriever = HybridRetriever()
agent_graph = build_financial_agent_graph()


class AgentAnalysisRequest(BaseModel):
    query: str


class AgentAnalysisResponse(BaseModel):
    query: str
    report: str
    audit_passed: bool
    iterations: int


@app.post("/api/v1/query", response_model=DocumentInsightResponse)
def query_document(request: QueryRequest):
    """Hybrid RAG endpoint with Redis caching and Instructor JSON enforcement."""
    cache_key = f"query:{request.query}"

    # 1. Graceful Redis cache read
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return DocumentInsightResponse.model_validate_json(cached_data)
    except Exception as err:
        logger.warning(f"Redis cache lookup failed: {err}. Proceeding to RAG execution.")

    # 2. Hybrid Retrieval
    try:
        context_docs = retriever.retrieve(request.query, top_k=5)
        context_text = "\n---\n".join([doc["text"] for doc in context_docs])
        context_ids = [doc["id"] for doc in context_docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval system failure: {str(e)}")

    # 3. LLM Generation
    system_prompt = (
        "You are an enterprise financial specialist. Parse the retrieved context "
        "and return structured JSON matching the declared response schema precisely."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {request.query}"

    try:
        response: DocumentInsightResponse = client.chat.completions.create(
            model="mistral",
            response_model=DocumentInsightResponse,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_retries=3
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")

    response.source_chunks = context_ids

    # 4. Graceful Redis cache write
    try:
        redis_client.setex(cache_key, 3600, response.model_dump_json())
    except Exception as err:
        logger.warning(f"Redis cache write failed: {err}")

    return response


@app.post("/api/v1/agent/analyze", response_model=AgentAnalysisResponse)
def run_agent_analysis(request: AgentAnalysisRequest):
    """Triggers the LangGraph Financial Analyst & Auditor state machine."""
    try:
        initial_state = {
            "user_query": request.query,
            "next_step": "",
            "rag_context": [],
            "calculated_metrics": {},
            "draft_report": "",
            "audit_passed": False,
            "audit_feedback": None,
            "iteration_count": 0
        }

        # Synchronous invocation running on threadpool
        final_state = agent_graph.invoke(initial_state)

        return AgentAnalysisResponse(
            query=request.query,
            report=final_state.get("draft_report", ""),
            audit_passed=final_state.get("audit_passed", False),
            iterations=final_state.get("iteration_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failure: {str(e)}")