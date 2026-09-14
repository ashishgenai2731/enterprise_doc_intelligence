import json
from typing import Dict, Any
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from src.agents.state import AgentState
from src.agents.tools import query_financial_docs, execute_python_calc

# Initialize local LLM for agent reasoning steps
llm = ChatOllama(model="mistral", temperature=0.0)


def rag_analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: RAG Analyst Node
    Queries Pinecone vector store via HybridRetriever tool using user query or refined search.
    """
    query = state["user_query"]
    feedback = state.get("audit_feedback")

    # Refine search query if audit previously failed
    search_query = f"{query} {feedback}" if (
                feedback and not state.get("audit_passed", False)) else query

    retrieved_text = query_financial_docs.invoke({"query": search_query, "top_k": 5})

    return {
        "rag_context": [{"text": retrieved_text}],
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def code_executor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Code Execution Node
    Prompts LLM to generate Python code based on context numbers, then executes via sandboxed Python REPL tool.
    """
    query = state["user_query"]
    context_items = state.get("rag_context", [])
    context_text = "\n".join([item.get("text", "") for item in context_items])

    prompt = (
        "You are a quantitative financial analyst.\n"
        f"User Question: {query}\n"
        f"Financial Context:\n{context_text}\n\n"
        "Write a clean Python code snippet to compute the required financial metrics.\n"
        "Rules:\n"
        "1. Extract numerical values strictly from context.\n"
        "2. Assign the final output to a variable named `result`.\n"
        "3. Wrap code inside ```python ``` block."
    )

    response = llm.invoke(prompt)
    raw_response = str(response.content)

    # Extract clean Python snippet from Markdown block
    if "```python" in raw_response:
        code_snippet = raw_response.split("```python")[1].split("```")[0].strip()
    elif "```" in raw_response:
        code_snippet = raw_response.split("```")[1].split("```")[0].strip()
    else:
        code_snippet = raw_response.strip()

    execution_output = execute_python_calc.invoke({"code_snippet": code_snippet})

    return {
        "calculated_metrics": {
            "generated_code": code_snippet,
            "execution_result": execution_output
        }
    }


def auditor_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Auditor Node
    Evaluates groundedness of calculations against source context and produces final report draft.
    """
    query = state["user_query"]
    context_items = state.get("rag_context", [])
    context_text = "\n".join([item.get("text", "") for item in context_items])
    metrics = state.get("calculated_metrics", {})

    prompt = (
        "You are a strict financial compliance auditor.\n"
        f"User Question: {query}\n"
        f"Source Context:\n{context_text}\n"
        f"Calculated Metrics:\n{json.dumps(metrics, indent=2)}\n\n"
        "Tasks:\n"
        "1. Synthesize an executive summary answering the user query.\n"
        "2. Verify if all values are strictly grounded in source context.\n"
        "3. Conclude your response with 'AUDIT_STATUS: PASSED' if accurate, or 'AUDIT_STATUS: FAILED' if hallucinated."
    )

    response = llm.invoke(prompt)
    report = str(response.content)
    is_passed = "AUDIT_STATUS: PASSED" in report.upper()

    return {
        "draft_report": report,
        "audit_passed": is_passed,
        "audit_feedback": report if not is_passed else None
    }