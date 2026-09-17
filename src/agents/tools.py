import asyncio
from typing import List
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.retrieval.hybrid_reranker import HybridRetriever

# Singleton retriever instance
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


async def _fetch_mcp_tools() -> List[BaseTool]:
    """Connects to the MCP server via stdio and dynamically converts tools."""
    client = MultiServerMCPClient(
        {
            "python_repl": {
                "command": "python",
                "args": ["-m", "src.mcp.server"],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()


def load_all_agent_tools() -> List[BaseTool]:
    """
    Combines in-process LangChain tools with dynamically loaded MCP server tools.
    """
    mcp_tools = asyncio.run(_fetch_mcp_tools())
    return [query_financial_docs] + mcp_tools