"""
QLoRA training-set builder: target code, context patching, leakage guard. Offline.
"""
import json
from pathlib import Path

import pytest

from src.agents.nodes import build_code_prompt
from src.training.dataset_prep import (add_missing_evidence, build_code, close_enough,
                                       run_like_sandbox)
from src.eval.golden_eval import sample_questions

GOLDEN = [json.loads(line) for line in Path("data/eval/edgar_golden_qa.jsonl").read_text().splitlines() if line.strip()]


def test_target_code_reproduces_every_golden_answer():
    """Every one of the 1,074 targets runs under the sandbox builtins and gives the XBRL answer."""
    wrong = [q["id"] for q in GOLDEN
             if not close_enough(run_like_sandbox(build_code(q)), q["answer"], q["unit"])]
    assert wrong == []


def test_target_code_shape():
    item = {"id": "AAPL-2024-revenue_growth", "type": "computed", "fiscal_year": 2024,
            "evidence": ["391,035", "383,285"], "answer": 2.02, "unit": "percent"}
    assert build_code(item) == (
        "revenue_2024 = 391035\n"
        "revenue_2023 = 383285\n"
        "result = round((revenue_2024 / revenue_2023 - 1) * 100, 2)"
    )
    assert run_like_sandbox(build_code(item)) == pytest.approx(2.02, abs=0.01)


def test_missing_figure_replaces_the_weakest_chunk():
    item = {"evidence": ["391,035"]}
    retrieved = [{"id": "a", "text": "segment sales 201,183"}, {"id": "b", "text": "iPhone 201,183"}]
    filing = [{"id": "z", "text": "Total net sales — 2024: 391,035; 2023: 383,285"}]
    patched = add_missing_evidence(retrieved, item, filing)
    assert [c["id"] for c in patched] == ["a", "z"]


def test_context_already_complete_is_untouched():
    item = {"evidence": ["391,035"]}
    retrieved = [{"id": "a", "text": "Total net sales — 2024: 391,035"}]
    assert add_missing_evidence(retrieved, item, []) == retrieved


def test_figure_absent_from_filing_drops_the_example():
    assert add_missing_evidence([{"id": "a", "text": "x"}], {"evidence": ["1,234"]}, []) is None


def test_training_companies_never_appear_in_the_eval_sample():
    eval_tickers = {q["ticker"] for q in sample_questions(GOLDEN, 60, seed=7)}
    train_path = Path("data/training/code_sft_train.jsonl")
    if not train_path.exists():
        pytest.skip("training set not built yet")
    train_tickers = {json.loads(line)["id"].split("-")[0] for line in train_path.read_text().splitlines()}
    assert train_tickers and not (train_tickers & eval_tickers)


def test_prompt_is_the_production_prompt():
    train_path = Path("data/training/code_sft_train.jsonl")
    if not train_path.exists():
        pytest.skip("training set not built yet")
    row = json.loads(train_path.read_text().splitlines()[0])
    prompt = row["prompt"][0]["content"]
    question = prompt.split("User Question: ", 1)[1].split("\n", 1)[0]
    context = prompt.split("Financial Context:\n", 1)[1].split("\n\nWrite a clean Python", 1)[0]
    assert prompt == build_code_prompt(question, context)
