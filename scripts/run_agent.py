import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.graph import build_financial_agent_graph


def main():
    print("=== Launching Autonomous Financial Analyst & Auditor Agent ===")

    # User query requiring hybrid retrieval + quantitative computation
    test_query = "What were total operating expenses for 2015 and what is the calculated percentage change compared to 2014?"

    initial_state = {
        "user_query": test_query,
        "next_step": "",
        "rag_context": [],
        "calculated_metrics": {},
        "draft_report": "",
        "audit_passed": False,
        "audit_feedback": None,
        "iteration_count": 0
    }

    graph = build_financial_agent_graph()

    print(f"\n[Query]: {test_query}\n")
    print("Executing state graph nodes...")

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 50)
    print("FINAL AUDITED REPORT")
    print("=" * 50)
    print(final_state.get("draft_report", "No report generated."))
    print("\n" + "=" * 50)
    print(f"Audit Status: {'PASSED' if final_state.get('audit_passed') else 'FAILED'}")
    print(f"Total Iterations: {final_state.get('iteration_count')}")


if __name__ == "__main__":
    main()