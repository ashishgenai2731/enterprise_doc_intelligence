import pytest
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from src.api.main import app

# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

client = TestClient(app)


def test_health_and_query_endpoint():
    """Validates the standard Hybrid RAG structured insight endpoint."""
    response = client.post(
        "/api/v1/query",
        json={"query": "What were total operating expenses for 2015?"}
    )
    assert response.status_code == 200
    data = response.json()

    assert "query" in data
    assert "summary" in data
    assert "metrics" in data
    assert isinstance(data["source_chunks"], list)


def test_agent_analyze_endpoint():
    """Validates full execution of the synchronous agent state machine."""
    response = client.post(
        "/api/v1/agent/analyze",
        json={"query": "Calculate percentage change in operating expenses between 2014 and 2015."}
    )
    assert response.status_code == 200
    data = response.json()

    assert "report" in data
    assert "audit_passed" in data
    assert isinstance(data["iterations"], int)
    assert data["iterations"] > 0


def test_agent_stream_endpoint():
    """Validates real-time SSE event streaming from node transitions."""
    with client.stream(
            "POST",
            "/api/v1/agent/analyze/stream",
            json={"query": "What were 2015 operating expenses?"}
    ) as response:
        assert response.status_code == 200
        events = [line for line in response.iter_lines() if line]

        # Verify node status events and closure tag
        assert any("node_start" in event for event in events)
        assert any("node_complete" in event for event in events)
        assert any("[DONE]" in event for event in events)