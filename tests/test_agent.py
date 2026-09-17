import sys
from pathlib import Path
import pytest

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.tools import call_mcp_python_repl
from src.agents.graph import build_financial_agent_graph


@pytest.mark.asyncio
async def test_mcp_python_executor_tool_success():
    """Validates math computation and output capture via MCP Python REPL tool."""
    code = "result = round(((1800 - 1200) / 1200) * 100, 2)"
    output = await call_mcp_python_repl(code)

    assert "SUCCESS" in output
    assert "50.0" in output


@pytest.mark.asyncio
async def test_mcp_python_executor_tool_error_handling():
    """Ensures division by zero produces an execution error without crashing process."""
    code = "result = 100 / 0"
    output = await call_mcp_python_repl(code)

    assert "EXECUTION_ERROR" in output
    assert "division by zero" in output.lower()


def test_agent_graph_compilation():
    """Verifies all registered state nodes and conditional edges compile successfully."""
    graph = build_financial_agent_graph()
    assert graph is not None

    node_keys = graph.nodes.keys()
    assert "rag_analyst" in node_keys
    assert "code_executor" in node_keys
    assert "auditor" in node_keys