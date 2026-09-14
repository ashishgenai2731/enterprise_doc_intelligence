import sys
import io
import math
from typing import Dict, Any
from langchain_core.tools import tool
from src.retrieval.hybrid_reranker import HybridRetriever

# Singleton retriever instance to avoid re-initializing models across agent calls
_retriever_instance = None


def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


@tool
def query_financial_docs(query: str, top_k: int = 5) -> str:
    """
    Queries the 10-K document vector index using hybrid search and cross-encoder reranking.
    Use this tool to find verified financial facts, figures, and historical textual context.
    """
    try:
        retriever = get_retriever()
        results = retriever.retrieve(query=query, top_k=top_k)

        if not results:
            return "No relevant financial contexts found."

        formatted_chunks = []
        for idx, res in enumerate(results, 1):
            chunk_text = res.get("text", "")
            score = res.get("rerank_score", res.get("score", 0.0))
            doc_id = res.get("id", f"chunk_{idx}")
            formatted_chunks.append(
                f"[Source ID: {doc_id} | Relevance Score: {score:.4f}]\n{chunk_text}"
            )

        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        return f"Error executing retrieval tool: {str(e)}"


@tool
def execute_python_calc(code_snippet: str) -> str:
    """
    Executes a Python code snippet to compute quantitative financial formulas
    (e.g., operating margin, YoY percentage growth, CAGR, interest coverage ratio).

    The code must assign its final answer to a variable named `result` or print it.
    """
    buffer = io.StringIO()
    sys.stdout = buffer

    # Restrict execution scope for safe local operations
    safe_globals = {
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "len": len,
    }
    local_scope: Dict[str, Any] = {}

    try:
        exec(code_snippet, safe_globals, local_scope)
        sys.stdout = sys.__stdout__

        printed_output = buffer.getvalue().strip()

        if "result" in local_scope:
            return f"Calculation Result: {local_scope['result']}"
        elif printed_output:
            return f"Execution Output:\n{printed_output}"
        else:
            return "Execution successful, but no 'result' variable or print statement was produced."

    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Execution Error: {type(e).__name__} - {str(e)}"