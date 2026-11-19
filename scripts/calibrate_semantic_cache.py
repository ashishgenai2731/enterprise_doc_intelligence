"""
Picks the similarity threshold for the semantic cache from real question pairs.

A cached answer may be reused only when a new question asks for the SAME figure. The risky
neighbours are questions about another line item of the same company and year ("revenue" vs
"net income"): they share almost every word. This script embeds, with the production embedding model:

  positives: a golden question vs. a human-style rewording of it          -> should hit
  negatives: a rewording vs. the golden question of ANOTHER metric,
             same company and year                                        -> must miss
  year swap: a rewording vs. the same metric one year earlier             -> must miss
  segment / quarter: the same line item for one segment or one quarter    -> must miss

For each it reports how often cosine similarity alone would reuse the answer, and how often the
production rule (same scope key AND similarity >= threshold) would, so the threshold is set
from data, not guessed.

    python -m scripts.calibrate_semantic_cache --pairs 12
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings
from src.cache.semantic_cache import SemanticCache
from src.retrieval.query_filters import detect_filters

THRESHOLDS = (0.75, 0.80, 0.85, 0.90)

GOLDEN = Path("data/eval/edgar_golden_qa.jsonl")
OUT = Path("data/eval/semantic_cache_calibration.json")

# Two everyday rewordings per lookup metric ({co} = company name, {y} = fiscal year)
REWORDINGS = {
    "revenue": ["How much revenue did {co} make in FY{y}?", "{co} total net sales for fiscal year {y}?"],
    "net_income": ["How much profit did {co} earn in fiscal {y}?", "{co} FY{y} net income?"],
    "operating_income": ["{co}'s operating profit for fiscal {y}?", "What did {co} report as income from operations in FY{y}?"],
    "rd": ["How much did {co} spend on R&D in fiscal {y}?", "{co} research and development spending in FY{y}?"],
    "cfo": ["{co}'s operating cash flow in fiscal {y}?", "How much cash did {co} generate from operations in FY{y}?"],
    "total_assets": ["What were {co}'s total assets at the end of FY{y}?", "{co} total assets on the balance sheet, fiscal {y}?"],
    "equity": ["{co} stockholders' equity at the end of fiscal {y}?", "What was {co}'s shareholders' equity at FY{y} year end?"],
    "eps_diluted": ["{co} diluted EPS for fiscal {y}?", "What were {co}'s diluted earnings per share in FY{y}?"],
}


def company_name(question: str) -> str:
    return re.match(r"What was (.+?)'s ", question).group(1)


def main(n_company_years: int, seed: int = 7):
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    by_cy = defaultdict(dict)
    for r in rows:
        by_cy[(r["ticker"], r["fiscal_year"])][r["id"].split("-", 2)[2]] = r["question"]

    # Company-years that have every lookup metric and the previous year too
    eligible = sorted(cy for cy, qs in by_cy.items()
                      if all(m in qs for m in REWORDINGS) and (cy[0], cy[1] - 1) in by_cy)
    picked = random.Random(seed).sample(eligible, min(n_company_years, len(eligible)))

    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    def embed(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=64)

    cache = SemanticCache(redis_client=None, embed=lambda text: [0.0])  # only scope_key() is used

    def same_scope(a: str, b: str) -> bool:
        key_a = cache.scope_key(a, detect_filters(a))
        return key_a is not None and key_a == cache.scope_key(b, detect_filters(b))

    groups = ("positive", "other_metric", "year_swap", "segment", "quarter")
    pairs = {g: [] for g in groups}  # (similarity, same scope?, new question, cached question)
    for ticker, year in picked:
        golden = by_cy[(ticker, year)]
        prev = by_cy[(ticker, year - 1)]
        co = company_name(golden["revenue"])
        for metric, templates in REWORDINGS.items():
            for template in templates:
                asked = template.format(co=co, y=year)
                cases = [("positive", asked, golden[metric])]
                cases += [("other_metric", asked, q) for m, q in golden.items() if m != metric]
                if metric in prev:
                    cases.append(("year_swap", asked, prev[metric]))
                # Same line item, but a part of the year or of the company: a different figure
                cases.append(("segment", asked.rstrip("?") + " in the Americas segment?", golden[metric]))
                cases.append(("quarter", asked.rstrip("?") + " for the fourth quarter only?", golden[metric]))
                vecs = embed([c[1] for c in cases] + [c[2] for c in cases])
                half = len(cases)
                for i, (group, new_q, cached_q) in enumerate(cases):
                    sim = float(vecs[i] @ vecs[half + i])
                    pairs[group].append((sim, same_scope(new_q, cached_q), new_q, cached_q))

    summary = {}
    for group in groups:
        sims = np.array([p[0] for p in pairs[group]])
        scoped = [p for p in pairs[group] if p[1]]
        summary[group] = {
            "pairs": len(sims),
            "similarity_min": round(float(sims.min()), 3),
            "similarity_median": round(float(np.median(sims)), 3),
            "similarity_max": round(float(sims.max()), 3),
            "same_scope_key": f"{len(scoped)} of {len(sims)}",
            "same_scope_similarity_range": [round(min(p[0] for p in scoped), 3), round(max(p[0] for p in scoped), 3)]
            if scoped else None,
        }
    for t in THRESHOLDS:
        row = {}
        for group in groups:
            cosine_only = np.mean([p[0] >= t for p in pairs[group]])
            guarded = np.mean([p[0] >= t and p[1] for p in pairs[group]])
            row[group] = f"cosine only {cosine_only:.1%} | scope key + cosine {guarded:.1%}"
        summary[f"hit_rate_at_{t}"] = row
    summary["company_years"] = [f"{t}-{y}" for t, y in picked]
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=12, help="number of company-years to sample")
    main(parser.parse_args().pairs)
