"""
Semantic cache: scope rules, threshold, Redis failures. Offline: Redis and the embedder are faked.
"""
import json

import numpy as np
import pytest

from src.cache.semantic_cache import SemanticCache, detect_metrics

APPLE_2024 = {"ticker": {"$in": ["AAPL"]}, "fiscal_year": {"$in": [2024]}}
APPLE_2023 = {"ticker": {"$in": ["AAPL"]}, "fiscal_year": {"$in": [2023]}}


class FakeRedis:
    """The four list commands the cache uses, kept in a dict."""

    def __init__(self):
        self.lists, self.ttl = {}, {}

    def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start:end + 1]

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key, start, end):
        items = self.lists.get(key, [])
        self.lists[key] = items[start:] if end == -1 else items[start:end + 1]

    def expire(self, key, seconds):
        self.ttl[key] = seconds


class BrokenRedis(FakeRedis):
    def lrange(self, *args):
        raise ConnectionError("redis down")

    def rpush(self, *args):
        raise ConnectionError("redis down")


VOCAB = ["revenue", "sales", "much", "apple", "make", "total", "fy2024", "fiscal", "2024", "how", "did"]


def bag_of_words(text):
    """Deterministic stand-in for the embedding model: word counts over a tiny vocabulary."""
    words = text.lower().replace("?", "").replace("'s", "").split()
    return [float(words.count(w)) for w in VOCAB] + [1.0]


def make_cache(redis=None, threshold=0.80, **kwargs):
    return SemanticCache(redis or FakeRedis(), embed=bag_of_words, threshold=threshold, **kwargs)


PAYLOAD = {"query": "q", "report": "Revenue was 391,035 [Source: AAPL-FY2024-chunk-3]", "audit_passed": True,
           "iterations": 1, "sources": ["AAPL-FY2024-chunk-3"], "unsupported_claims": []}


# ---------- line-item detection ----------

@pytest.mark.parametrize("question, expected", [
    ("What was Apple's total revenue in fiscal 2024?", ["revenue"]),
    ("Apple total net sales for fiscal year 2024?", ["revenue"]),
    ("How much profit did Apple earn in fiscal 2024?", ["net_income"]),
    ("What was Apple's operating income in fiscal 2024?", ["operating_income"]),
    ("Apple's operating profit for fiscal 2024?", ["operating_income"]),
    ("How much cash did Apple generate from operations in FY2024?", ["cfo"]),
    ("What were Apple's diluted earnings per share in FY2024?", ["eps_diluted"]),
    ("What was Apple's gross margin (gross profit / total revenue) in fiscal 2024?", ["gross_margin"]),
    ("Apple current ratio (total current assets / total current liabilities) 2024?", ["current_ratio"]),
    ("What percentage of total revenue did Apple spend on research and development in 2024?", ["rd_share"]),
    ("By what percentage did Apple's net income change from fiscal 2023 to fiscal 2024?", ["net_income_change"]),
    ("Salesforce net income in fiscal 2025?", ["net_income"]),  # 'sales' inside 'Salesforce' is not revenue
    ("Who is Apple's CEO?", []),
])
def test_detect_metrics(question, expected):
    assert detect_metrics(question) == expected


# ---------- scope key ----------

def test_rewordings_share_a_scope():
    cache = make_cache()
    assert cache.scope_key("How much revenue did Apple make in FY2024?", APPLE_2024) == \
        cache.scope_key("What was Apple's total revenue in fiscal 2024?", APPLE_2024)


@pytest.mark.parametrize("other_question, other_filters", [
    ("What was Apple's net income in fiscal 2024?", APPLE_2024),       # another line item
    ("What was Apple's total revenue in fiscal 2023?", APPLE_2023),    # another year
    ("How fast did Apple's revenue grow in fiscal 2024?", APPLE_2024), # a change, not a value
])
def test_different_figures_get_different_scopes(other_question, other_filters):
    cache = make_cache()
    assert cache.scope_key("What was Apple's total revenue in fiscal 2024?", APPLE_2024) != \
        cache.scope_key(other_question, other_filters)


@pytest.mark.parametrize("question, filters", [
    ("What was Apple's revenue in the Americas segment in fiscal 2024?", APPLE_2024),
    ("What was Apple's revenue for the fourth quarter of fiscal 2024?", APPLE_2024),
    ("Apple Q4 2024 revenue?", APPLE_2024),
    ("Apple adjusted EPS for fiscal 2024?", APPLE_2024),
    ("Who is Apple's CEO?", APPLE_2024),                      # no line item
    ("What was total revenue in fiscal 2024?", {}),           # no company
])
def test_uncacheable_questions_have_no_scope(question, filters):
    cache = make_cache()
    assert cache.scope_key(question, filters) is None
    assert cache.store(question, filters, PAYLOAD) is False
    assert cache.lookup(question, filters) is None


# ---------- lookup / store ----------

def test_hit_on_a_rewording_above_threshold():
    cache = make_cache()
    assert cache.store("How much revenue did Apple make in FY2024?", APPLE_2024, PAYLOAD)
    hit = cache.lookup("How much revenue did Apple make in fiscal 2024?", APPLE_2024)
    assert hit is not None
    assert hit["payload"] == PAYLOAD
    assert hit["cached_question"] == "How much revenue did Apple make in FY2024?"
    assert 0.80 <= hit["similarity"] <= 1.0


def test_miss_below_threshold_in_the_same_scope():
    cache = make_cache(threshold=0.99)
    cache.store("How much revenue did Apple make in FY2024?", APPLE_2024, PAYLOAD)
    assert cache.lookup("Apple total sales fiscal 2024?", APPLE_2024) is None


def test_other_year_never_sees_the_entry():
    cache = make_cache(threshold=0.0)  # even with no similarity bar, the scope key keeps years apart
    cache.store("What was Apple's total revenue in fiscal 2024?", APPLE_2024, PAYLOAD)
    assert cache.lookup("What was Apple's total revenue in fiscal 2023?", APPLE_2023) is None


def test_best_match_wins():
    cache = make_cache(threshold=0.0)
    cache.store("Apple total sales fiscal 2024?", APPLE_2024, {**PAYLOAD, "report": "far"})
    cache.store("How much revenue did Apple make in FY2024?", APPLE_2024, {**PAYLOAD, "report": "close"})
    hit = cache.lookup("How much revenue did Apple make in FY2024?", APPLE_2024)
    assert hit["payload"]["report"] == "close"
    assert hit["similarity"] == pytest.approx(1.0, abs=1e-4)


def test_bucket_is_trimmed_and_given_a_ttl():
    redis = FakeRedis()
    cache = make_cache(redis, max_entries_per_scope=2, ttl_sec=60)
    for i in range(3):
        cache.store("How much revenue did Apple make in FY2024?", APPLE_2024, {**PAYLOAD, "report": str(i)})
    key = cache.scope_key("How much revenue did Apple make in FY2024?", APPLE_2024)
    assert [json.loads(e)["payload"]["report"] for e in redis.lists[key]] == ["1", "2"]  # oldest dropped
    assert redis.ttl[key] == 60


def test_stored_vector_is_normalised():
    redis = FakeRedis()
    cache = make_cache(redis)
    cache.store("How much revenue did Apple make in FY2024?", APPLE_2024, PAYLOAD)
    (raw,) = next(iter(redis.lists.values()))
    assert np.linalg.norm(json.loads(raw)["vector"]) == pytest.approx(1.0, abs=1e-4)


def test_redis_failure_is_a_miss_not_a_crash():
    cache = make_cache(BrokenRedis())
    assert cache.lookup("How much revenue did Apple make in FY2024?", APPLE_2024) is None
    assert cache.store("How much revenue did Apple make in FY2024?", APPLE_2024, PAYLOAD) is False
