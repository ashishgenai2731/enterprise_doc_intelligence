import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
import instructor
from openai import OpenAI
import redis
from src.api.schema import QueryRequest, DocumentInsightResponse
from src.retrieval.hybrid_reranker import HybridRetriever
from config.settings import settings
from src.agents.graph import build_financial_agent_graph
from src.agents.state import initial_agent_state
from src.cache.semantic_cache import SemanticCache
from src.retrieval.query_filters import detect_filters

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

# Semantic cache for agent answers: same Redis, same embedding model as retrieval (already loaded).
# Threshold 0.80 and the scope rules come from scripts/calibrate_semantic_cache.py.
semantic_cache = SemanticCache(
    redis_client=redis_client,
    embed=lambda text: retriever.vector_store.embed_model.encode(text, normalize_embeddings=True),
    threshold=settings.SEMANTIC_CACHE_THRESHOLD,
    ttl_sec=settings.SEMANTIC_CACHE_TTL_SEC,
)


def _cache_lookup(query: str):
    """(cached hit or None, detected filters). Never raises: a cache problem only costs time."""
    filters = detect_filters(query)
    try:
        return semantic_cache.lookup(query, filters), filters
    except Exception as err:
        logger.warning(f"Semantic cache lookup failed: {err}")
        return None, filters


def _cache_store(query: str, filters, payload: dict):
    # Only audited answers are reused; a failed audit would be served again for days
    if payload.get("audit_passed"):
        try:
            semantic_cache.store(query, filters, payload)
        except Exception as err:
            logger.warning(f"Semantic cache write failed: {err}")


class AgentAnalysisRequest(BaseModel):
    query: str


class AgentAnalysisResponse(BaseModel):
    query: str
    report: str
    audit_passed: bool
    iterations: int
    sources: List[str] = []
    unsupported_claims: List[str] = []
    cached: bool = False
    cache_similarity: Optional[float] = None


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
            model=settings.LLM_MODEL,
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
        redis_client.set(cache_key, response.model_dump_json(), ex=3600)
    except Exception as err:
        logger.warning(f"Redis cache write failed: {err}")

    return response


@app.post("/api/v1/agent/analyze", response_model=AgentAnalysisResponse)
async def run_agent_analysis(request: AgentAnalysisRequest):
    """Triggers the LangGraph Financial Analyst & Auditor state machine asynchronously."""
    hit, filters = _cache_lookup(request.query)
    if hit:
        return AgentAnalysisResponse(**hit["payload"], cached=True, cache_similarity=hit["similarity"])

    try:
        initial_state = initial_agent_state(request.query)

        # Asynchronous invocation required for async MCP execution nodes
        final_state = await agent_graph.ainvoke(initial_state)

        response = AgentAnalysisResponse(
            query=request.query,
            report=final_state.get("draft_report", ""),
            audit_passed=final_state.get("audit_passed", False),
            iterations=final_state.get("iteration_count", 0),
            sources=final_state.get("sources", []),
            unsupported_claims=final_state.get("unsupported_claims", []),
        )
        _cache_store(request.query, filters, response.model_dump(exclude={"cached", "cache_similarity"}))
        return response
    except Exception as e:
        logger.error(f"Agent execution failure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failure: {str(e)}")


@app.post("/api/v1/agent/analyze/stream")
async def stream_agent_analysis(request: AgentAnalysisRequest):
    """Streams real-time state machine node transitions and final payload over SSE."""
    initial_state = initial_agent_state(request.query)
    hit, filters = _cache_lookup(request.query)

    async def event_generator():
        if hit:
            payload = hit["payload"]
            yield f"data: {json.dumps({'status': 'cache_hit', 'similarity': hit['similarity'], 'cached_question': hit['cached_question']})}\n\n"
            yield f"data: {json.dumps({'status': 'final_result', 'report': payload['report'], 'audit_passed': payload['audit_passed'], 'sources': payload['sources'], 'unsupported_claims': payload['unsupported_claims'], 'cached': True})}\n\n"
            yield "data: [DONE]\n\n"
            return

        final_report = ""
        iterations = 0
        audit_passed = False
        sources, unsupported_claims = [], []

        # Stream events from LangGraph state execution
        async for event in agent_graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")

            # Catch when a graph node starts or finishes
            if kind == "on_chain_start" and name in ["rag_analyst", "code_executor", "auditor"]:
                yield f"data: {json.dumps({'status': 'node_start', 'node': name})}\n\n"
            elif kind == "on_chain_end" and name in ["rag_analyst", "code_executor", "auditor"]:
                yield f"data: {json.dumps({'status': 'node_complete', 'node': name})}\n\n"
                if name == "rag_analyst":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        iterations = output.get("iteration_count", iterations)

                # Capture state outputs from auditor node completion
                if name == "auditor":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        final_report = output.get("draft_report", final_report)
                        audit_passed = output.get("audit_passed", audit_passed)
                        sources = output.get("sources", sources)
                        unsupported_claims = output.get("unsupported_claims", unsupported_claims)

        # Emit final structured payload before closing stream
        final_payload = {
            'status': 'final_result',
            'report': final_report,
            'audit_passed': audit_passed,
            'sources': sources,
            'unsupported_claims': unsupported_claims,
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        yield "data: [DONE]\n\n"

        _cache_store(request.query, filters, {
            "query": request.query, "report": final_report, "audit_passed": audit_passed,
            "iterations": iterations, "sources": sources, "unsupported_claims": unsupported_claims,
        })

    return StreamingResponse(event_generator(), media_type="text/event-stream")