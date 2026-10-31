"""
Picks the company and filing a question is about, so search runs inside the right 10-K.

With 30 companies x 3 years indexed, an unfiltered search for "net income in fiscal 2024"
returns near-identical income-statement chunks from many companies; the reranker cannot tell
Apple's from Microsoft's. Matching names and years is deterministic and cheap, so it is done
before retrieval instead of asking the LLM.
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from config.companies import COMPANIES
from config.settings import settings

_YEAR_RE = re.compile(r"\b(?:FY\s?)?((?:19|20)\d{2})\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _available_years() -> Dict[str, List[int]]:
    """ticker -> fiscal years indexed for it, from the filings manifest."""
    path = Path(settings.FILINGS_MANIFEST)
    years: Dict[str, List[int]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                years.setdefault(row["ticker"], []).append(int(row["fiscal_year"]))
    return {ticker: sorted(set(ys)) for ticker, ys in years.items()}


def detect_tickers(query: str) -> List[str]:
    lowered = query.lower()
    found = []
    for company in COMPANIES:
        names = [re.escape(alias) for alias in company["aliases"]]
        # Tickers only match in upper case, so 'KO' or 'HD' inside ordinary words never count
        if re.search(rf"(?<![\w&])(?:{'|'.join(names)})(?![\w&])", lowered) or \
                re.search(rf"\b{company['ticker']}\b", query):
            found.append(company["ticker"])
    return found


def detect_filters(query: str, available_years: Dict[str, List[int]] = None) -> Dict[str, Any]:
    """
    Pinecone-style metadata filter for the question, or {} to search everything.

    A 10-K also reports the two prior years, so a question about fiscal 2023 is answered by
    the earliest indexed filing for fiscal 2023 or later; for a comparison ('2023 to 2024')
    the latest year mentioned decides, because that filing contains both columns.
    """
    available_years = available_years if available_years is not None else _available_years()
    tickers = [t for t in detect_tickers(query) if t in available_years]
    if not tickers:
        return {}

    filters: Dict[str, Any] = {"ticker": {"$in": tickers}}
    mentioned = [int(y) for y in _YEAR_RE.findall(query)]
    if mentioned:
        target = max(mentioned)
        chosen = {min((y for y in available_years[t] if y >= target), default=None) for t in tickers}
        chosen.discard(None)
        if chosen:
            filters["fiscal_year"] = {"$in": sorted(chosen)}
    return filters


def matches(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Local equivalent of the Pinecone filter, used to pick the BM25 sub-corpus."""
    return all(metadata.get(field) in condition["$in"] for field, condition in filters.items())
