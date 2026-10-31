"""
Minimal SEC EDGAR client: company lookup, 10-K filing lists, filing documents and XBRL facts.

EDGAR rules: a User-Agent with a contact email on every request, at most 10 requests/second.
"""
import time
from typing import Any, Dict, List

import requests

from config.settings import settings

_MIN_INTERVAL_SEC = 0.12  # stays under EDGAR's 10 requests/second limit
_last_request = 0.0


def sec_get(url: str, retries: int = 3) -> requests.Response:
    global _last_request
    if not settings.SEC_USER_AGENT:
        raise RuntimeError("Set SEC_USER_AGENT in .env, e.g. 'Your Name you@example.com' - EDGAR returns 403 without it")
    for attempt in range(retries):
        wait = _MIN_INTERVAL_SEC - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        response = requests.get(url, headers={"User-Agent": settings.SEC_USER_AGENT}, timeout=60)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()
    return response


def cik_by_ticker() -> Dict[str, int]:
    rows = sec_get("https://www.sec.gov/files/company_tickers.json").json().values()
    return {row["ticker"]: row["cik_str"] for row in rows}


def recent_10k_filings(cik: int, count: int) -> List[Dict[str, Any]]:
    """The `count` most recent original 10-K filings (amendments excluded), newest first."""
    recent = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    filings = []
    for form, accession, document, report_date, filing_date in zip(
            recent["form"], recent["accessionNumber"], recent["primaryDocument"],
            recent["reportDate"], recent["filingDate"]):
        if form == "10-K":
            filings.append({
                "accession": accession,
                "report_date": report_date,
                "filing_date": filing_date,
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{document}",
            })
    return filings[:count]


def company_facts(cik: int) -> Dict[str, Any]:
    """All XBRL facts the company has ever filed (us-gaap and dei taxonomies)."""
    return sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()


def fiscal_year_of(facts: Dict[str, Any], accession: str) -> int:
    """The fiscal year a filing reports on, as the company itself labels it (DEI 'fy').

    Taken from the filing's own XBRL facts, because companies with non-calendar years
    (Walmart ends in January, Microsoft in June) do not match the calendar year of the end date.
    """
    for taxonomy in facts["facts"].values():
        for concept in taxonomy.values():
            for unit_rows in concept["units"].values():
                for row in unit_rows:
                    if row.get("accn") == accession and row.get("form") == "10-K" and row.get("fy"):
                        return int(row["fy"])
    raise ValueError(f"No XBRL facts found for filing {accession}")
