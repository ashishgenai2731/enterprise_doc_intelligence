import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
        redis_client.set(cache_key, response.model_dump_json(), ex=3600)
    except Exception as err:
        logger.warning(f"Redis cache write failed: {err}")

    return response


@app.post("/api/v1/agent/analyze", response_model=AgentAnalysisResponse)
async def run_agent_analysis(request: AgentAnalysisRequest):
    """Triggers the LangGraph Financial Analyst & Auditor state machine asynchronously."""
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

        # Asynchronous invocation required for async MCP execution nodes
        final_state = await agent_graph.ainvoke(initial_state)

        return AgentAnalysisResponse(
            query=request.query,
            report=final_state.get("draft_report", ""),
            audit_passed=final_state.get("audit_passed", False),
            iterations=final_state.get("iteration_count", 0)
        )
    except Exception as e:
        logger.error(f"Agent execution failure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failure: {str(e)}")


@app.post("/api/v1/agent/analyze/stream")
async def stream_agent_analysis(request: AgentAnalysisRequest):
    """Streams real-time state machine node transitions and final payload over SSE."""
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

    async def event_generator():
        final_report = ""
        audit_passed = False

        # Stream events from LangGraph state execution
        async for event in agent_graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")

            # Catch when a graph node starts or finishes
            if kind == "on_chain_start" and name in ["rag_analyst", "code_executor", "auditor"]:
                yield f"data: {json.dumps({'status': 'node_start', 'node': name})}\n\n"
            elif kind == "on_chain_end" and name in ["rag_analyst", "code_executor", "auditor"]:
                yield f"data: {json.dumps({'status': 'node_complete', 'node': name})}\n\n"

                # Capture state outputs from auditor node completion
                if name == "auditor":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        final_report = output.get("draft_report", final_report)
                        audit_passed = output.get("audit_passed", audit_passed)

        # Emit final structured payload before closing stream
        yield f"data: {json.dumps({'status': 'final_result', 'report': final_report, 'audit_passed': audit_passed})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")