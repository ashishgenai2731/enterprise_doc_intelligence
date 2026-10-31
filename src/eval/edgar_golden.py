"""
Builds golden Q&A pairs for the EDGAR filings from the companies' own XBRL data.

Every 10-K is filed with machine-readable XBRL facts (NetIncomeLoss = 93,736,000,000 for the
year ended 2024-09-28, ...). Those facts give the right answer without anyone reading the
filing by hand, so the test set can grow with the corpus. A question is kept only if every
input figure also appears in the parsed filing text: otherwise it tests the parser, not the agent.

Usage:
    python -m src.eval.edgar_golden        # writes data/eval/edgar_golden_qa.jsonl
"""
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings

OUT_PATH = Path("data/eval/edgar_golden_qa.jsonl")

# Metric -> XBRL concepts that can carry it, in order of preference
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "rd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "total_assets": ["Assets"],
    "equity": ["StockholdersEquity"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}
INSTANT = {"total_assets", "equity", "current_assets", "current_liabilities"}

LOOKUPS = {
    "revenue": "total revenue",
    "net_income": "net income",
    "operating_income": "operating income",
    "rd": "research and development expense",
    "cfo": "net cash provided by operating activities",
    "total_assets": "total assets",
    "equity": "total shareholders' equity",
    "eps_diluted": "diluted earnings per share",
}

# id -> (question template, inputs, formula, unit); templates get {name}, {year}, {period}, {prev}
COMPUTED = {
    "revenue_growth": ("By what percentage did {name}'s total revenue change from fiscal {prev} to fiscal {year}",
                       ["revenue", "revenue_prior"], lambda v: (v["revenue"] / v["revenue_prior"] - 1) * 100, "percent"),
    "net_income_growth": ("By what percentage did {name}'s net income change from fiscal {prev} to fiscal {year}",
                          ["net_income", "net_income_prior"],
                          lambda v: (v["net_income"] / v["net_income_prior"] - 1) * 100, "percent"),
    "gross_margin": ("What was {name}'s gross margin (gross profit / total revenue) in fiscal {year}",
                     ["gross_profit", "revenue"], lambda v: v["gross_profit"] / v["revenue"] * 100, "percent"),
    "operating_margin": ("What was {name}'s operating margin (operating income / total revenue) in fiscal {year}",
                         ["operating_income", "revenue"], lambda v: v["operating_income"] / v["revenue"] * 100, "percent"),
    "net_margin": ("What was {name}'s net margin (net income / total revenue) in fiscal {year}",
                   ["net_income", "revenue"], lambda v: v["net_income"] / v["revenue"] * 100, "percent"),
    "rd_intensity": ("What percentage of total revenue did {name} spend on research and development in fiscal {year}",
                     ["rd", "revenue"], lambda v: v["rd"] / v["revenue"] * 100, "percent"),
    "current_ratio": ("What was {name}'s current ratio (total current assets / total current liabilities) "
                      "at the end of fiscal {year}", ["current_assets", "current_liabilities"],
                      lambda v: v["current_assets"] / v["current_liabilities"], "ratio"),
}


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _pick(rows: List[Dict[str, Any]], end: str, instant: bool) -> Optional[float]:
    for row in rows:
        if row["end"] != end:
            continue
        if instant and "start" not in row:
            return row["val"]
        if not instant and "start" in row and 350 <= _days(row["start"], row["end"]) <= 380:
            return row["val"]
    return None


def filing_values(facts: Dict[str, Any], accession: str, report_date: str) -> Dict[str, float]:
    """Current- and prior-year values of each metric, all taken from this one filing."""
    gaap = facts["facts"].get("us-gaap", {})
    values: Dict[str, float] = {}
    for metric, concepts in CONCEPTS.items():
        unit = "USD/shares" if metric == "eps_diluted" else "USD"
        candidates, priors = [], []
        for concept in concepts:
            rows = [r for r in gaap.get(concept, {}).get("units", {}).get(unit, []) if r.get("accn") == accession]
            current = _pick(rows, report_date, metric in INSTANT)
            if current is not None:
                candidates.append(current)
                # Prior fiscal year: the latest full-year period ending before this one
                prior_ends = sorted({r["end"] for r in rows if r["end"] < report_date and "start" in r
                                     and 350 <= _days(r["start"], r["end"]) <= 380})
                prior = _pick(rows, prior_ends[-1], False) if prior_ends and metric not in INSTANT else None
                if prior is not None:
                    priors.append(prior)
                # Several revenue or R&D concepts can be tagged at once; the total is the largest
                # (J&J tags a $483M item as ResearchAndDevelopmentExpense and its $15,085M total
                # R&D as ...ExcludingAcquiredInProcessCost)
                if metric not in ("revenue", "rd"):
                    break
        if candidates:
            best = max(range(len(candidates)), key=lambda i: candidates[i])
            values[metric] = candidates[best]
            if len(priors) == len(candidates):
                values[f"{metric}_prior"] = priors[best]
    return values


def as_filed(metric: str, value: float) -> str:
    """How the figure is printed in the filing: $ millions with commas, or EPS in dollars."""
    return f"{value:.2f}" if metric.startswith("eps") else f"{round(value / 1e6):,}"


def appears_in(text: str, figure: str) -> bool:
    return re.search(rf"(?<![\d,.]){re.escape(figure)}(?![\d,]|\.\d)", text) is not None


def _period(report_date: str) -> str:
    d = date.fromisoformat(report_date)
    return f"{d:%B} {d.day}, {d.year}"


def build_questions(filing: Dict[str, Any], values: Dict[str, float], text: str) -> List[Dict[str, Any]]:
    year, name = filing["fiscal_year"], filing["company"]
    period = _period(filing["report_date"])
    base = {"ticker": filing["ticker"], "fiscal_year": year}
    questions = []

    def verified(inputs: List[str]) -> Optional[List[str]]:
        if not all(m in values for m in inputs):
            return None
        figures = [as_filed(m.replace("_prior", ""), values[m]) for m in inputs]
        return figures if all(appears_in(text, f) for f in figures) else None

    for metric, label in LOOKUPS.items():
        figures = verified([metric])
        if not figures:
            continue
        when = f"as of {period}, the end of fiscal {year}" if metric in INSTANT else \
            f"in fiscal {year} (the year ended {period})"
        unit, answer = ("usd", round(values[metric], 2)) if metric == "eps_diluted" \
            else ("usd_millions", round(values[metric] / 1e6))
        suffix = "in dollars" if unit == "usd" else "in $ millions"
        questions.append({**base, "id": f"{filing['ticker']}-{year}-{metric}", "type": "lookup",
                          "question": f"What was {name}'s {label} {when}, {suffix}?",
                          "answer": answer, "unit": unit, "evidence": figures})

    for qid, (template, inputs, formula, unit) in COMPUTED.items():
        figures = verified(inputs)
        if not figures:
            continue
        question = template.format(name=name, year=year, prev=year - 1) + f" (fiscal {year} ended {period})?"
        question += " Answer in percent." if unit == "percent" else ""
        questions.append({**base, "id": f"{filing['ticker']}-{year}-{qid}", "type": "computed",
                          "question": question, "answer": round(formula(values), 3 if unit == "ratio" else 2),
                          "unit": unit, "evidence": figures})
    return questions


def main():
    from src.ingestion.html_parser import html_to_text

    rows = [json.loads(line) for line in Path(settings.FILINGS_MANIFEST).read_text().splitlines() if line.strip()]
    # Local PDFs have no XBRL; the hand-built golden_qa.jsonl covers Apple 2015
    filings = [f for f in rows if not f["path"].endswith(".pdf")]
    questions, dropped = [], 0
    for filing in filings:
        facts = json.loads((Path(settings.EDGAR_DIR) / "facts" / f"{filing['ticker']}.json").read_text())
        values = filing_values(facts, filing["accession"], filing["report_date"])
        text = html_to_text(Path(filing["path"]).read_bytes())
        built = build_questions(filing, values, text)
        dropped += len(LOOKUPS) + len(COMPUTED) - len(built)
        questions.extend(built)
        print(f"{filing['ticker']} FY{filing['fiscal_year']}: {len(built)} questions", flush=True)

    OUT_PATH.write_text("".join(json.dumps(q) + "\n" for q in questions))
    lookups = sum(q["type"] == "lookup" for q in questions)
    print(f"\n{len(questions)} questions ({lookups} lookup, {len(questions) - lookups} computed) "
          f"from {len(filings)} filings; {dropped} skipped (no XBRL value or figure not found in text)")


if __name__ == "__main__":
    main()
