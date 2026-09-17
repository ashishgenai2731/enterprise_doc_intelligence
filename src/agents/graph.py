from langgraph.graph import StateGraph, START, END
from src.agents.state import AgentState
from src.agents.nodes import rag_analyst_node, code_executor_node, auditor_node


def route_audit(state: AgentState) -> str:
    """Conditional edge routing: terminates on success or max iteration threshold."""
    if state.get("audit_passed", False):
        return END
    if state.get("iteration_count", 0) >= 3:
        return END  # Safety guard preventing infinite retry loops
    return "rag_analyst"


def build_financial_agent_graph():
    """Constructs and compiles the multi-agent state graph."""
    builder = StateGraph(AgentState)

    # Register graph nodes
    builder.add_node("rag_analyst", rag_analyst_node)
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("auditor", auditor_node)

    # Define deterministic directed execution sequence
    builder.add_edge(START, "rag_analyst")
    builder.add_edge("rag_analyst", "code_executor")
    builder.add_edge("code_executor", "auditor")

    # Conditional edge routing based on audit verification
    builder.add_conditional_edges("auditor", route_audit)

    return builder.compile()


app = build_financial_agent_graph()