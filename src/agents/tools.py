import sys
import asyncio
from typing import Dict, Any
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.retrieval.hybrid_reranker import HybridRetriever

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


async def call_mcp_python_repl(code_snippet: str) -> str:
    """Invokes the standalone MCP Server tool over JSON-RPC stdio on the active event loop."""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp.server"],
        env=None
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
                "execute_python_calc",
                arguments={"code": code_snippet}
            )

            if isinstance(result.content, list) and len(result.content) > 0:
                return result.content[0].text
            return str(result.content)