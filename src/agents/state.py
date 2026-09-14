from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    user_query: str
    next_step: str
    rag_context: List[Dict[str, Any]]
    calculated_metrics: Dict[str, Any]
    draft_report: str
    audit_passed: bool
    audit_feedback: Optional[str]
    iteration_count: int