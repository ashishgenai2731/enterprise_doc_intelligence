import pytest
import sys
from pathlib import Path

from src.agents.tools import execute_python_calc
from src.agents.graph import build_financial_agent_graph

# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_python_executor_tool_success():
    """Validates math computation and variable capture in execution scope."""
    code = "result = round(((1800 - 1200) / 1200) * 100, 2)"
    output = execute_python_calc.invoke({"code_snippet": code})
    assert "Calculation Result: 50.0" in output


def test_python_executor_tool_error_handling():
    """Ensures division by zero doesn't crash the agent node."""
    code = "result = 100 / 0"
    output = execute_python_calc.invoke({"code_snippet": code})
    assert "Execution Error: ZeroDivisionError" in output


def test_agent_graph_compilation():
    """Verifies all registered nodes and conditional edges compile successfully."""
    graph = build_financial_agent_graph()
    assert graph is not None

    node_keys = graph.nodes.keys()
    assert "rag_analyst" in node_keys
    assert "code_executor" in node_keys
    assert "auditor" in node_keys