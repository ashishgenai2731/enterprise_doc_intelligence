"""
Companies whose 10-K filings are indexed from SEC EDGAR.

Non-financial companies only: banks and insurers have no gross profit line and use
industry-specific statements, so the generated golden questions would not apply to them.
`aliases` are the names a user might type; query_filters.py uses them to pick the right filing.
"""
COMPANIES = [
    {"ticker": "AAPL", "name": "Apple", "aliases": ["apple"]},
    {"ticker": "MSFT", "name": "Microsoft", "aliases": ["microsoft"]},
    {"ticker": "GOOGL", "name": "Alphabet", "aliases": ["alphabet", "google"]},
    {"ticker": "AMZN", "name": "Amazon", "aliases": ["amazon"]},
    {"ticker": "META", "name": "Meta Platforms", "aliases": ["meta platforms", "meta", "facebook"]},
    {"ticker": "NVDA", "name": "NVIDIA", "aliases": ["nvidia"]},
    {"ticker": "TSLA", "name": "Tesla", "aliases": ["tesla"]},
    {"ticker": "NFLX", "name": "Netflix", "aliases": ["netflix"]},
    {"ticker": "ADBE", "name": "Adobe", "aliases": ["adobe"]},
    {"ticker": "CRM", "name": "Salesforce", "aliases": ["salesforce"]},
    {"ticker": "ORCL", "name": "Oracle", "aliases": ["oracle"]},
    {"ticker": "INTC", "name": "Intel", "aliases": ["intel"]},
    {"ticker": "AMD", "name": "AMD", "aliases": ["advanced micro devices", "amd"]},
    {"ticker": "CSCO", "name": "Cisco", "aliases": ["cisco"]},
    {"ticker": "QCOM", "name": "Qualcomm", "aliases": ["qualcomm"]},
    {"ticker": "TXN", "name": "Texas Instruments", "aliases": ["texas instruments"]},
    {"ticker": "IBM", "name": "IBM", "aliases": ["ibm", "international business machines"]},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "aliases": ["johnson & johnson", "johnson and johnson", "j&j"]},
    {"ticker": "PFE", "name": "Pfizer", "aliases": ["pfizer"]},
    {"ticker": "MRK", "name": "Merck", "aliases": ["merck"]},
    {"ticker": "ABBV", "name": "AbbVie", "aliases": ["abbvie"]},
    {"ticker": "LLY", "name": "Eli Lilly", "aliases": ["eli lilly", "lilly"]},
    {"ticker": "KO", "name": "Coca-Cola", "aliases": ["coca-cola", "coca cola", "coke"]},
    {"ticker": "PEP", "name": "PepsiCo", "aliases": ["pepsico", "pepsi"]},
    {"ticker": "PG", "name": "Procter & Gamble", "aliases": ["procter & gamble", "procter and gamble", "p&g"]},
    {"ticker": "WMT", "name": "Walmart", "aliases": ["walmart"]},
    {"ticker": "HD", "name": "Home Depot", "aliases": ["home depot"]},
    {"ticker": "NKE", "name": "Nike", "aliases": ["nike"]},
    {"ticker": "MCD", "name": "McDonald's", "aliases": ["mcdonald's", "mcdonalds"]},
    {"ticker": "CAT", "name": "Caterpillar", "aliases": ["caterpillar"]},
]
