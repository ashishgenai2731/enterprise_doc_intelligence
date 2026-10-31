"""
Downloads the most recent 10-K filings and XBRL facts for every company in config/companies.py.

Writes:
    data/edgar/raw/<TICKER>/<accession>.htm   the filing document (gitignored)
    data/edgar/facts/<TICKER>.json            XBRL company facts, used to build golden questions (gitignored)
    data/edgar/filings.jsonl                  manifest: one row per filing

Already-downloaded files are skipped, so the script can be re-run after an interruption.

Usage:
    python -m scripts.edgar_download                    # 3 latest 10-Ks per company
    python -m scripts.edgar_download --years 2 --tickers AAPL MSFT
"""
import argparse
import json
from pathlib import Path

from config.companies import COMPANIES
from config.settings import settings
from src.ingestion.edgar import cik_by_ticker, recent_10k_filings, company_facts, fiscal_year_of, sec_get


def main(years: int, tickers=None):
    edgar_dir = Path(settings.EDGAR_DIR)
    companies = [c for c in COMPANIES if not tickers or c["ticker"] in tickers]
    ciks = cik_by_ticker()
    manifest = []

    for company in companies:
        ticker = company["ticker"]
        cik = ciks[ticker]
        facts_path = edgar_dir / "facts" / f"{ticker}.json"
        if not facts_path.exists():
            facts_path.parent.mkdir(parents=True, exist_ok=True)
            facts_path.write_text(json.dumps(company_facts(cik)))
        facts = json.loads(facts_path.read_text())

        previous_year = 0
        for filing in reversed(recent_10k_filings(cik, years)):  # oldest first
            raw_path = edgar_dir / "raw" / ticker / f"{filing['accession']}.htm"
            if not raw_path.exists():
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(sec_get(filing["url"]).content)
            # Filers occasionally tag two consecutive 10-Ks with the same fiscal year
            # (Salesforce FY2026 is tagged 2025); one company's years must be unique
            fiscal_year = max(fiscal_year_of(facts, filing["accession"]), previous_year + 1)
            previous_year = fiscal_year
            row = {
                "ticker": ticker,
                "company": company["name"],
                "cik": cik,
                "fiscal_year": fiscal_year,
                **filing,
                "path": str(raw_path),
            }
            manifest.append(row)
            print(f"{ticker} FY{row['fiscal_year']} (period end {row['report_date']}): "
                  f"{raw_path.stat().st_size / 1e6:.1f} MB", flush=True)

    # Merge by accession, so rows added another way (a local PDF) survive a re-download
    manifest_path = Path(settings.FILINGS_MANIFEST)
    rows = {}
    if manifest_path.exists():
        rows = {r["accession"]: r for r in map(json.loads, manifest_path.read_text().splitlines()) if r}
    rows.update({r["accession"]: r for r in manifest})
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows.values()))
    print(f"\n{len(manifest)} filings from {len(companies)} companies -> {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3, help="Most recent 10-Ks per company")
    parser.add_argument("--tickers", nargs="*", help="Only these tickers")
    args = parser.parse_args()
    main(args.years, args.tickers)
