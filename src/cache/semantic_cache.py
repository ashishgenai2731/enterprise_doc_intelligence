"""
Semantic cache for agent answers, stored in Redis.

An agent run takes 35-272 s (median 77 s, docs/PROJECT_METRICS.md). Analysts ask the same
question in different words ("Apple FY2024 net income?" vs "How much profit did Apple make in
fiscal 2024?"), so an exact-text cache rarely hits. This cache reuses an answer when a new
question MEANS the same thing.

The danger in finance is a near-identical question about a different figure. Measured with
bge-large (scripts/calibrate_semantic_cache.py): rewordings of the same question score 0.73-0.91,
but "How much profit did Meta earn in fiscal 2025?" vs. Meta's OPERATING income question scores
0.90, and the same metric one year earlier scores up to 0.88. The ranges overlap, so no cosine
threshold alone is safe. Two guards are used instead:

1. Scope key (deterministic). Entries live in a bucket keyed by the company/year metadata filter,
   the years written in the question and the financial line items it names (revenue, net income,
   operating income, margins, growth, ...). A question about net income in 2024 never even sees
   operating-income or 2023 entries. A question whose line item is not recognised is not cached.
2. Similarity threshold. Inside a bucket, an entry is reused only above a cosine similarity, which
   catches extra conditions the scope key cannot see (e.g. a segment or a region).

Only answers that passed the audit are stored, and every Redis error is swallowed: a broken
cache must never break an answer, it only makes it slower.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

# Line items, checked in order; a phrase is removed once matched so "operating income" is not also
# read as "net income" ("income") or "operating cash flow" as "operating income".
_METRIC_PATTERNS = [
    ("eps_diluted", r"diluted (?:eps|earnings per share)|(?:eps|earnings per share)"),
    ("cfo", r"cash (?:flow )?(?:provided by |generated (?:by|from) |from )?operat\w*|operating cash flow|cash\b[^?.]{0,40}\bfrom operations"),
    ("gross_margin", r"gross margin|gross profit margin"),
    ("operating_margin", r"operating margin"),
    ("net_margin", r"net (?:profit )?margin|profit margin"),
    ("rd", r"research (?:and|&) development|r ?& ?d"),
    ("operating_income", r"operating income|operating profit|income from operations"),
    ("net_income", r"net income|net profit|net earnings|profit|earn(?:ed|ings)?"),
    ("revenue", r"revenue|net sales|total sales|sales|turnover"),
    ("current_ratio", r"current ratio"),
    ("total_assets", r"total assets|assets"),
    ("equity", r"equity"),
]
# A part of the year or of the company is a different figure from the full-year consolidated one,
# and embeddings cannot tell them apart (measured: "... in the Americas segment?" and "... for the
# fourth quarter only?" score 0.71-0.89 against the plain question, the same range as true
# rewordings). Such questions are never cached.
_PART_RE = re.compile(
    r"\b(?:quarter\w*|q[1-4]|(?:three|six|nine) months|(?:first|second) half|half[- ]year|monthly|month"
    r"|segments?|regions?|regional|geograph\w*|americas|europe|emea|apac|asia|china|japan"
    r"|excluding|adjusted|non-gaap|organic|constant currency|by product|product lines?|divisions?)(?!\w)"
)

_RATIOS = {"gross_margin", "operating_margin", "net_margin", "current_ratio", "rd_share"}
_CHANGE_RE = re.compile(r"\b(?:growth|grow|grew|change|changed|increase|increased|decrease|decreased|rise|rose|fall|fell)\b")
_SHARE_RE = re.compile(r"(?:percent|percentage|%|share|intensity) of (?:total )?revenue|as a (?:percent|percentage|%)")


def detect_metrics(query: str) -> List[str]:
    """Financial line items a question asks about, e.g. ['net_income_change'] or ['rd_share']."""
    text = query.lower()
    found = []
    for name, pattern in _METRIC_PATTERNS:
        regex = re.compile(rf"\b(?:{pattern})(?!\w)")  # whole words: "sales" must not match "Salesforce"
        if regex.search(text):
            found.append(name)
            text = regex.sub(" ", text)
    if not found:
        return []
    # "R&D as a percentage of revenue" is one figure, not R&D plus revenue
    if "rd" in found and _SHARE_RE.search(query.lower()):
        found = ["rd_share"]
    # A ratio is one figure; the parts named in its definition ("gross profit / total revenue",
    # "current assets / current liabilities") are not extra line items
    ratios = [m for m in found if m in _RATIOS]
    if ratios:
        found = ratios
    # A change between years is a different figure from the value in one year
    if _CHANGE_RE.search(query.lower()) and not any(m.endswith("margin") for m in found):
        found = [f"{m}_change" for m in found]
    return sorted(found)


class SemanticCache:
    def __init__(
        self,
        redis_client,
        embed: Callable[[str], List[float]],
        threshold: float = 0.80,
        ttl_sec: int = 24 * 3600,
        max_entries_per_scope: int = 200,
        prefix: str = "semcache:v1",
    ):
        """
        redis_client: a redis.Redis (or anything with lrange / rpush / ltrim / expire)
        embed: text -> vector; it is normalised here, so any embedding model works
        threshold: minimum cosine similarity to reuse an answer (see the calibration script)
        ttl_sec: a scope bucket expires this long after its last write (filings change yearly,
                 but a day keeps a bad entry from living forever)
        max_entries_per_scope: oldest entries are dropped beyond this, so a lookup stays a
                 small dot product instead of a scan of everything ever asked
        """
        self.redis = redis_client
        self.embed = embed
        self.threshold = threshold
        self.ttl_sec = ttl_sec
        self.max_entries = max_entries_per_scope
        self.prefix = prefix

    def scope_key(self, query: str, filters: Dict[str, Any]) -> Optional[str]:
        """Bucket for a question: company/year filter + years written + line items asked.
        None (not cached) when no company or no line item is recognised, or when the question asks
        for part of the year or of the company (a quarter, a segment, a region, an adjusted figure)."""
        metrics = detect_metrics(query)
        if not filters or not metrics or _PART_RE.search(query.lower()):
            return None
        filter_part = json.dumps(
            {field: sorted(cond["$in"]) for field, cond in sorted(filters.items())},
            sort_keys=True,
        )
        years = sorted(set(_YEAR_RE.findall(query)))
        return f"{self.prefix}:{filter_part}:{','.join(years)}:{','.join(metrics)}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.asarray(self.embed(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def lookup(self, query: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Returns {'payload', 'similarity', 'cached_question'} for the closest entry above the
        threshold in this question's scope, or None."""
        key = self.scope_key(query, filters)
        if key is None:
            return None
        try:
            raw_entries = self.redis.lrange(key, 0, -1)
        except Exception as err:
            logger.warning("Semantic cache read failed (%s); running the agent", err)
            return None
        if not raw_entries:
            return None

        entries = [json.loads(raw) for raw in raw_entries]
        matrix = np.asarray([entry["vector"] for entry in entries], dtype=np.float32)
        similarities = matrix @ self._vector(query)
        best = int(np.argmax(similarities))
        if similarities[best] < self.threshold:
            return None
        return {
            "payload": entries[best]["payload"],
            "similarity": round(float(similarities[best]), 4),
            "cached_question": entries[best]["question"],
        }

    def store(self, query: str, filters: Dict[str, Any], payload: Dict[str, Any]) -> bool:
        key = self.scope_key(query, filters)
        if key is None:
            return False
        entry = {"question": query, "vector": self._vector(query).round(6).tolist(), "payload": payload}
        try:
            self.redis.rpush(key, json.dumps(entry))
            self.redis.ltrim(key, -self.max_entries, -1)
            self.redis.expire(key, self.ttl_sec)
            return True
        except Exception as err:
            logger.warning("Semantic cache write failed (%s); answer not cached", err)
            return False
