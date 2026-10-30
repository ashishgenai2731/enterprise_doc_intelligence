from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    user_query: str
    next_step: str
    rag_context: List[Dict[str, Any]]  # retrieved chunks: {'id', 'text', 'score'}
    calculated_metrics: Dict[str, Any]
    draft_report: str
    audit_passed: bool
    audit_feedback: Optional[str]
    retry_query: Optional[str]  # auditor's targeted search query for the next round
    unsupported_claims: List[str]
    sources: List[str]  # chunk IDs cited in the final report
    iteration_count: int


def initial_agent_state(query: str) -> AgentState:
    return {
        "user_query": query,
        "next_step": "",
        "rag_context": [],
        "calculated_metrics": {},
        "draft_report": "",
        "audit_passed": False,
        "audit_feedback": None,
        "retry_query": None,
        "unsupported_claims": [],
        "sources": [],
        "iteration_count": 0,
    }
