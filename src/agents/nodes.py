import json
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from src.agents.state import AgentState
from src.agents.tools import get_retriever, call_mcp_python_repl

MAX_CONTEXT_CHUNKS = 10


class AuditVerdict(BaseModel):
    """Structured auditor output - replaces string-matching on 'AUDIT_STATUS: PASSED'."""
    summary: str = Field(
        description="Executive summary answering the question. Cite every figure as [Source: <chunk id>]."
    )
    passed: bool = Field(description="True only if every figure is supported by the source context.")
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Statements or figures in the summary that the source context does not support.",
    )
    retry_query: Optional[str] = Field(
        default=None,
        description="If not passed: a short search query (under 15 words) for the missing evidence.",
    )


def get_llm() -> ChatOllama:
    """Instantiates a ChatOllama client bound to the active event loop."""
    return ChatOllama(model="mistral", temperature=0.0)


def format_context(chunks: List[Dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(f"[Source: {c['id']}]\n{c['text']}" for c in chunks)


_CITATION_RE = re.compile(r"\[Source:[^\]]*\]")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _extract_numbers(text: str) -> List[float]:
    values = []
    for raw in _NUMBER_RE.findall(_CITATION_RE.sub("", text)):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def find_ungrounded_numbers(summary: str, evidence: str) -> List[str]:
    """
    Deterministic grounding check: every figure in the summary must appear in the
    retrieved context or the sandbox output (0.5% tolerance for rounding).
    Integers below 10 are skipped - they are usually counts or list markers, not figures.
    """
    evidence_values = _extract_numbers(evidence)
    ungrounded = []
    for value in _extract_numbers(summary):
        if value < 10 and value.is_integer():
            continue
        tolerance = 0.005 * max(1.0, abs(value))
        if not any(abs(value - ev) <= tolerance for ev in evidence_values):
            ungrounded.append(f"{value:g}")
    return ungrounded


def rag_analyst_node(state: AgentState) -> Dict[str, Any]:
    query = state["user_query"]
    retry_query = state.get("retry_query")

    # On a retry, search for the specific evidence the auditor said was missing,
    # instead of gluing the whole failed report onto the query
    search_query = f"{query} {retry_query}" if retry_query else query
    new_chunks = get_retriever().retrieve(query=search_query, top_k=5)

    # Keep earlier evidence and add the new chunks, so a retry widens the context
    merged = {c["id"]: c for c in state.get("rag_context", [])}
    for chunk in new_chunks:
        merged.setdefault(chunk["id"], chunk)

    return {
        "rag_context": list(merged.values())[:MAX_CONTEXT_CHUNKS],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def code_executor_node(state: AgentState) -> Dict[str, Any]:
    query = state["user_query"]
    context_text = format_context(state.get("rag_context", []))

    prompt = (
        "You are a quantitative financial analyst.\n"
        f"User Question: {query}\n"
        f"Financial Context:\n{context_text}\n\n"
        "Write a clean Python code snippet to compute the required financial metrics.\n"
        "Rules:\n"
        "1. Read the numbers from the context yourself and write them into the code as literals, "
        "e.g. `revenue_2015 = 233715`. The code cannot see the context - there is no `context` variable.\n"
        "2. No imports. Only these built-ins exist: abs, min, max, sum, len, round, float, int.\n"
        "3. Assign the final output to a variable named `result`.\n"
        "4. Wrap code inside ```python ``` block."
    )

    llm = get_llm()
    response = await llm.ainvoke(prompt)
    raw_response = str(response.content)

    if "```python" in raw_response:
        code_snippet = raw_response.split("```python")[1].split("```")[0].strip()
    elif "```" in raw_response:
        code_snippet = raw_response.split("```")[1].split("```")[0].strip()
    else:
        code_snippet = raw_response.strip()

    execution_output = await call_mcp_python_repl(code_snippet)

    return {
        "calculated_metrics": {
            "generated_code": code_snippet,
            "execution_result": execution_output,
        }
    }


async def auditor_node(state: AgentState) -> Dict[str, Any]:
    query = state["user_query"]
    chunks = state.get("rag_context", [])
    context_text = format_context(chunks)
    metrics = state.get("calculated_metrics", {})

    prompt = (
        "You are a strict financial compliance auditor.\n"
        f"User Question: {query}\n"
        f"Source Context:\n{context_text}\n"
        f"Calculated Metrics:\n{json.dumps(metrics, indent=2)}\n\n"
        "Tasks:\n"
        "1. Write an executive summary answering the user question.\n"
        "2. After every figure, cite the chunk it came from as [Source: <chunk id>]. "
        "Computed figures cite the chunks their inputs came from.\n"
        "3. Set passed=false and list unsupported_claims if any figure is not in the source context "
        "or the calculated metrics.\n"
        "4. If passed=false, give a short retry_query naming the evidence that is missing."
    )

    try:
        verdict: AuditVerdict = await get_llm().with_structured_output(AuditVerdict).ainvoke(prompt)
    except Exception as err:
        # Malformed JSON from the model counts as a failed audit, not a crash
        verdict = AuditVerdict(summary="", passed=False, unsupported_claims=[f"Auditor output invalid: {err}"])

    # Second, non-LLM opinion: the model should not be the only judge of its own numbers
    evidence = context_text + "\n" + str(metrics.get("execution_result", ""))
    ungrounded = find_ungrounded_numbers(verdict.summary, evidence)
    unsupported = verdict.unsupported_claims + [f"Figure not found in sources: {n}" for n in ungrounded]
    passed = verdict.passed and not unsupported

    # A failed audit with no retry_query would just repeat the same search; fall back
    # to the first claim the auditor could not verify as the next search
    retry_query = verdict.retry_query or (verdict.unsupported_claims[0] if verdict.unsupported_claims else None)

    # Keep only citations that point at chunks we actually retrieved ("[Source: a, b]" or "a and b")
    known_ids = {c["id"] for c in chunks}
    cited = [
        cid.strip()
        for group in re.findall(r"\[Source:\s*([^\]]+)\]", verdict.summary)
        for cid in re.split(r",|\band\b", group)
        if cid.strip() in known_ids
    ]

    return {
        "draft_report": verdict.summary,
        "audit_passed": passed,
        "audit_feedback": None if passed else "; ".join(unsupported),
        "retry_query": None if passed else retry_query,
        "unsupported_claims": unsupported,
        "sources": list(dict.fromkeys(cited)),
    }
