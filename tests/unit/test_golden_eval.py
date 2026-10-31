from src.eval.golden_eval import extract_figures, is_correct, summarize


def test_billions_are_converted_to_millions():
    assert extract_figures("Net sales were $233.7 billion.", "usd_millions") == [233700.0]


def test_citation_ids_do_not_count_as_answers():
    assert not is_correct("No figure [Source: Form10K.pdf-chunk-408]", 408, "usd_millions")


def test_percent_tolerance_and_loose_sign():
    assert is_correct("Net income rose 35.1% [Source: c1].", 35.14, "percent")
    assert is_correct("iPad sales decreased by 23.30%.", -23.3, "percent")
    assert not is_correct("Net income rose 34.5%.", 35.14, "percent")


def test_summary_counts_false_passes():
    results = [
        {"type": "lookup", "correct": True, "audit_passed": True, "sources": ["a"], "iterations": 1, "latency_sec": 10},
        {"type": "computed", "correct": False, "audit_passed": True, "sources": [], "iterations": 1, "latency_sec": 20},
        {"type": "computed", "correct": False, "audit_passed": False, "sources": [], "iterations": 3, "latency_sec": 30},
    ]
    s = summarize(results)
    assert s["false_pass_rate"] == 0.5
    assert s["wrong_caught_by_audit"] == 1
    assert s["accuracy_lookup"] == 1.0 and s["accuracy_computed"] == 0.0
