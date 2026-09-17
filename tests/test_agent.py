import sys
from pathlib import Path
import pytest

# Ensure project root directory is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.tools import load_all_agent_tools
from src.agents.graph import build_financial_agent_graph


@pytest.fixture(scope="module")
def agent_tools():
    """Module-scoped fixture to resolve local and MCP server tools once."""
    tools_list = load_all_agent_tools()
    return {tool.name: tool for tool in tools_list}


def test_mcp_python_executor_tool_success(agent_tools):
    """Validates math computation and output capture via MCP Python REPL tool."""
    repl_tool = agent_tools["execute_python_calc"]
    code = "result = round(((1800 - 1200) / 1200) * 100, 2)"

    # MCP tool uses parameter name 'code'
    output = repl_tool.invoke({"code": code})

    assert "SUCCESS" in output
    assert "50.0" in output


def test_mcp_python_executor_tool_error_handling(agent_tools):
    """Ensures division by zero produces an execution error without crashing process."""
    repl_tool = agent_tools["execute_python_calc"]
    code = "result = 100 / 0"

    output = repl_tool.invoke({"code": code})

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