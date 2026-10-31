"""
Unit tests for the multi-filing EDGAR pipeline: HTML table parsing, company/year filters,
XBRL golden-question building and filtered hybrid retrieval. No network, Pinecone or models.
"""
from src.ingestion.html_parser import html_to_text
from src.retrieval import hybrid_reranker
from src.retrieval.hybrid_reranker import HybridRetriever
from src.retrieval.query_filters import detect_filters, detect_tickers
from src.retrieval.sparse_store import tokenize
from src.eval.edgar_golden import appears_in, as_filed, build_questions, filing_values

# ---------- HTML parsing ----------

INCOME_TABLE = b"""
<html><body>
<div style="display:none"><ix:header>hidden xbrl junk 999,999</ix:header></div>
<p>Results of <span>operations</span> for the year.</p>
<table>
  <tr><td colspan="3"></td><td colspan="3">September 28, 2024</td><td colspan="3">September 30, 2023</td></tr>
  <tr><td colspan="3">Net sales:</td></tr>
  <tr><td colspan="3">Products</td><td>$</td><td>294,866</td><td></td><td>$</td><td>298,085</td><td></td></tr>
  <tr><td colspan="3">Total net sales</td><td colspan="2">391,035</td><td></td><td colspan="2">383,285</td><td></td></tr>
  <tr><td colspan="3">Other income</td><td colspan="2">( 269 )</td><td></td><td>(565</td><td>)</td><td></td></tr>
</table>
</body></html>
"""


def test_table_values_land_in_their_year_column():
    text = html_to_text(INCOME_TABLE)
    assert "Net sales: Products — 2024: 294,866; 2023: 298,085" in text
    assert "Total net sales — 2024: 391,035; 2023: 383,285" in text


def test_negative_figures_are_rejoined():
    assert "Other income — 2024: (269); 2023: (565)" in html_to_text(INCOME_TABLE)


def test_hidden_xbrl_header_dropped_and_inline_spans_kept_in_sentence():
    text = html_to_text(INCOME_TABLE)
    assert "999,999" not in text
    assert "Results of operations for the year." in text


def test_two_row_header_keeps_change_columns():
    html = b"""<table>
      <tr><td colspan="3"></td><td colspan="6">Year Ended December 31,</td><td colspan="3">2025 vs. 2024 Change</td></tr>
      <tr><td colspan="3">(Dollars in millions)</td><td colspan="3">2025</td><td colspan="3">2024</td><td colspan="2"></td><td>%</td></tr>
      <tr><td colspan="3">Total revenues</td><td>94,827</td><td colspan="2"></td><td>97,690</td><td colspan="2"></td>
          <td>(2,863)</td><td></td><td>(3)%</td></tr>
    </table>"""
    assert html_to_text(html) == ("Total revenues — 2025: 94,827; 2024: 97,690; "
                                  "2025 vs. 2024 Change: (2,863); 2025 vs. 2024 Change: (3)%")


# ---------- company / year filters ----------

YEARS = {"AAPL": [2023, 2024, 2025], "MSFT": [2024, 2025, 2026], "INTC": [2023, 2024, 2025]}


def test_detects_company_names_and_upper_case_tickers_only():
    assert detect_tickers("Compare Apple and MSFT revenue") == ["AAPL", "MSFT"]
    assert detect_tickers("intellectual property and microsoftware") == []


def test_year_filter_picks_earliest_filing_that_reports_it():
    assert detect_filters("Apple net income in fiscal 2024", YEARS) == \
        {"ticker": {"$in": ["AAPL"]}, "fiscal_year": {"$in": [2024]}}
    # 2022 is not indexed for Apple, but the FY2023 10-K reports it as the prior year
    assert detect_filters("Apple revenue change from 2022 to 2023", YEARS)["fiscal_year"] == {"$in": [2023]}


def test_no_company_or_unindexed_year_degrades_gracefully():
    assert detect_filters("What is a 10-K?", YEARS) == {}
    assert detect_filters("Apple revenue in 2031", YEARS) == {"ticker": {"$in": ["AAPL"]}}


# ---------- XBRL golden questions ----------

FACTS = {"facts": {"us-gaap": {
    "Revenues": {"units": {"USD": [
        {"start": "2023-10-01", "end": "2024-09-28", "val": 391_035_000_000, "accn": "A1"},
        {"start": "2022-09-25", "end": "2023-09-30", "val": 383_285_000_000, "accn": "A1"},
        {"start": "2024-06-30", "end": "2024-09-28", "val": 94_930_000_000, "accn": "A1"},  # a quarter
    ]}},
    "NetIncomeLoss": {"units": {"USD": [
        {"start": "2023-10-01", "end": "2024-09-28", "val": 93_736_000_000, "accn": "A1"},
        {"start": "2022-09-25", "end": "2023-09-30", "val": 96_995_000_000, "accn": "A1"},
        {"start": "2023-10-01", "end": "2024-09-28", "val": 1, "accn": "OTHER"},  # another filing
    ]}},
    "Assets": {"units": {"USD": [{"end": "2024-09-28", "val": 364_980_000_000, "accn": "A1"}]}},
}}}
FILING = {"ticker": "AAPL", "company": "Apple", "fiscal_year": 2024, "accession": "A1", "report_date": "2024-09-28"}


def test_filing_values_use_full_years_from_the_same_filing():
    values = filing_values(FACTS, "A1", "2024-09-28")
    assert values["revenue"] == 391_035_000_000 and values["revenue_prior"] == 383_285_000_000
    assert values["net_income"] == 93_736_000_000
    assert values["total_assets"] == 364_980_000_000


def test_rd_uses_the_largest_tagged_concept():
    # J&J: a small item under ResearchAndDevelopmentExpense, the real total under the IPR&D-excluding concept
    facts = {"facts": {"us-gaap": {
        "ResearchAndDevelopmentExpense": {"units": {"USD": [
            {"start": "2023-01-02", "end": "2023-12-31", "val": 483_000_000, "accn": "J1"}]}},
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost": {"units": {"USD": [
            {"start": "2023-01-02", "end": "2023-12-31", "val": 15_085_000_000, "accn": "J1"}]}},
    }}}
    assert filing_values(facts, "J1", "2023-12-31")["rd"] == 15_085_000_000


def test_figures_match_filing_format_exactly():
    assert as_filed("revenue", 391_035_000_000) == "391,035"
    assert appears_in("Total net sales — 2024: 391,035", "391,035")
    assert not appears_in("1,391,035.5", "391,035")


def test_questions_only_kept_when_inputs_are_in_the_text():
    values = filing_values(FACTS, "A1", "2024-09-28")
    text = "Total net sales — 2024: 391,035; 2023: 383,285\nNet income — 2024: 93,736; 2023: 96,995"
    by_id = {q["id"]: q for q in build_questions(FILING, values, text)}

    assert by_id["AAPL-2024-net_income_growth"]["answer"] == -3.36
    assert by_id["AAPL-2024-net_margin"]["answer"] == 23.97
    assert "AAPL-2024-total_assets" not in by_id  # 364,980 is not in the text


# ---------- filtered hybrid retrieval ----------

class FilterRecordingStore:
    def __init__(self, ids):
        self.ids, self.filters = ids, []

    def query(self, query_text, top_k=20, filter=None):
        self.filters.append(filter)
        return [{"id": i, "score": 0.5, "metadata": {}} for i in self.ids]


class OverlapReranker:
    def predict(self, pairs):
        return [len(set(tokenize(q)) & set(tokenize(t))) for q, t in pairs]


CORPUS = [
    {"id": "aapl", "text": "Apple net income 93,736", "metadata": {"ticker": "AAPL", "fiscal_year": 2024}},
    {"id": "msft", "text": "Microsoft net income 88,136", "metadata": {"ticker": "MSFT", "fiscal_year": 2024}},
]


def test_filters_reach_pinecone_and_restrict_bm25(monkeypatch):
    monkeypatch.setattr(hybrid_reranker, "_get_reranker", lambda: OverlapReranker())
    store = FilterRecordingStore(["aapl"])
    retriever = HybridRetriever(doc_corpus=CORPUS, vector_store=store)
    filters = {"ticker": {"$in": ["AAPL"]}}

    results = retriever.retrieve("net income", top_k=5, filters=filters)

    assert store.filters == [filters]
    assert [r["id"] for r in results] == ["aapl"]  # BM25 never saw the Microsoft chunk


def test_filter_matching_nothing_falls_back_to_unfiltered(monkeypatch):
    monkeypatch.setattr(hybrid_reranker, "_get_reranker", lambda: OverlapReranker())

    class EmptyWhenFiltered(FilterRecordingStore):
        def query(self, query_text, top_k=20, filter=None):
            return [] if filter else super().query(query_text, top_k)

    retriever = HybridRetriever(doc_corpus=CORPUS, vector_store=EmptyWhenFiltered(["msft"]))
    assert retriever.retrieve("net income", filters={"ticker": {"$in": ["AAPL"]}})
