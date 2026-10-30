"""
Fast unit tests for retrieval fusion, grounding checks and agent routing.
No Pinecone, Ollama or model downloads: the retriever, reranker and LLM are faked.
"""
import pytest

from src.retrieval import hybrid_reranker
from src.retrieval.hybrid_reranker import HybridRetriever
from src.retrieval.sparse_store import tokenize, load_corpus, append_to_corpus
from src.agents import nodes
from src.agents.nodes import AuditVerdict, find_ungrounded_numbers
from src.agents.graph import route_audit
from src.agents.state import initial_agent_state

CORPUS = [
    {"id": "c1", "text": "Total operating expenses were $38,000 million in 2015."},
    {"id": "c2", "text": "Research and development expense grew in fiscal 2015."},
    {"id": "c3", "text": "The Company sells iPhone, iPad and Mac products."},
]


class FakeVectorStore:
    def __init__(self, ids):
        self.ids = ids

    def query(self, query_text, top_k=20):
        return [{"id": i, "score": 0.5, "metadata": {}} for i in self.ids[:top_k]]


class FakeReranker:
    def predict(self, pairs):
        # Score by keyword overlap so the test stays deterministic
        return [len(set(tokenize(q)) & set(tokenize(t))) for q, t in pairs]


# ---------- sparse store / corpus ----------

def test_tokenize_strips_punctuation():
    assert tokenize("Operating Expenses?") == ["operating", "expenses"]


def test_corpus_roundtrip_skips_duplicate_ids(tmp_path):
    path = str(tmp_path / "corpus.jsonl")
    append_to_corpus(path, CORPUS[:2])
    append_to_corpus(path, CORPUS)  # c1, c2 already there
    assert [d["id"] for d in load_corpus(path)] == ["c1", "c2", "c3"]


def test_load_corpus_missing_file_returns_empty(tmp_path):
    assert load_corpus(str(tmp_path / "nope.jsonl")) == []


# ---------- hybrid retrieval ----------

def test_rrf_rewards_chunks_found_by_both_retrievers():
    fused = HybridRetriever._reciprocal_rank_fusion(["a", "b", "c"], ["c", "d"])
    assert fused[0] == "c"


def test_bm25_only_hit_reaches_results(monkeypatch):
    """c1 is not in the dense results at all; only BM25 can surface it."""
    monkeypatch.setattr(hybrid_reranker, "_get_reranker", lambda: FakeReranker())
    retriever = HybridRetriever(doc_corpus=CORPUS, vector_store=FakeVectorStore(["c3"]))

    results = retriever.retrieve("total operating expenses 2015", top_k=2)

    assert retriever.sparse_store is not None
    assert results[0]["id"] == "c1"


# ---------- grounding check ----------

def test_grounded_figures_pass_with_rounding_tolerance():
    evidence = "Operating expenses 38,000 in 2015. SUCCESS: 12.3456"
    assert find_ungrounded_numbers("Expenses were 38,000 [Source: c1], up 12.35%.", evidence) == []


def test_invented_figure_is_flagged():
    assert find_ungrounded_numbers("Revenue was 91,250 in 2015.", "Revenue 2015: 233,715") == ["91250"]


def test_citation_ids_and_small_counts_are_ignored():
    summary = "Across 3 segments [Source: Form10K.pdf-chunk-251-e5d228] costs rose."
    assert find_ungrounded_numbers(summary, "") == []


# ---------- graph routing ----------

def test_route_audit_ends_on_pass_or_max_iterations():
    assert route_audit({"audit_passed": True, "iteration_count": 1}) == "__end__"
    assert route_audit({"audit_passed": False, "iteration_count": 3}) == "__end__"
    assert route_audit({"audit_passed": False, "iteration_count": 1}) == "rag_analyst"


# ---------- nodes ----------

class RecordingRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def retrieve(self, query, top_k=5):
        self.queries.append(query)
        return self.chunks


def test_retry_searches_with_auditor_query_and_keeps_old_context(monkeypatch):
    fake = RecordingRetriever([CORPUS[1]])
    monkeypatch.setattr(nodes, "get_retriever", lambda: fake)

    state = initial_agent_state("operating expenses 2015")
    state.update(rag_context=[CORPUS[0]], retry_query="R&D expense 2015", iteration_count=1)
    out = nodes.rag_analyst_node(state)

    assert fake.queries == ["operating expenses 2015 R&D expense 2015"]
    assert [c["id"] for c in out["rag_context"]] == ["c1", "c2"]
    assert out["iteration_count"] == 2


class FakeLLM:
    def __init__(self, verdict):
        self.verdict = verdict

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt):
        return self.verdict


@pytest.mark.asyncio
async def test_auditor_overrides_llm_pass_when_figure_is_ungrounded(monkeypatch):
    verdict = AuditVerdict(summary="Expenses were 41,000 [Source: c1].", passed=True)
    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM(verdict))

    state = initial_agent_state("operating expenses 2015")
    state["rag_context"] = [CORPUS[0]]
    out = await nodes.auditor_node(state)

    assert out["audit_passed"] is False
    assert any("41000" in claim for claim in out["unsupported_claims"])


@pytest.mark.asyncio
async def test_auditor_passes_grounded_report_and_keeps_only_real_citations(monkeypatch):
    verdict = AuditVerdict(
        summary="Expenses were 38,000 million in 2015 [Source: c1 and made-up-id].", passed=True
    )
    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM(verdict))

    state = initial_agent_state("operating expenses 2015")
    state["rag_context"] = [CORPUS[0]]
    out = await nodes.auditor_node(state)

    assert out["audit_passed"] is True
    assert out["retry_query"] is None
    assert out["sources"] == ["c1"]
