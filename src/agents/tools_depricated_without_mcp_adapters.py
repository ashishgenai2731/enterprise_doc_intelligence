import asyncio
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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


async def call_mcp_python_repl(code_snippet: str) -> str:
    """Invokes the standalone MCP Server tool over JSON-RPC stdio."""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp.server"],
        env=None
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Dynamically call tool exposed by MCP Server
            result = await session.call_tool(
                "execute_python_calc",
                arguments={"code": code_snippet}
            )
            return result.content[0].text


@tool
def execute_python_calc(code_snippet: str) -> str:
    """
    Executes a Python code snippet to compute quantitative financial formulas
    (e.g., operating margin, YoY percentage growth, CAGR, interest coverage ratio)
    via an external Model Context Protocol (MCP) server sandbox.

    The code must assign its final answer to a variable named `result` or print it.
    """
    try:
        # Route execution to standalone MCP server process
        return asyncio.run(call_mcp_python_repl(code_snippet))
    except Exception as e:
        return f"MCP Execution Bridge Error: {str(e)}"