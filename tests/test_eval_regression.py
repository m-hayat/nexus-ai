"""
Nexus-AI Platform — Eval dataset regression tests.

Verifies:
- Dataset schema integrity
- No duplicate IDs
- All required fields present
- Category distribution is sane
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parent.parent / "data" / "eval_dataset.json"


@pytest.fixture(scope="module")
def dataset():
    """Load eval dataset."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_dataset_exists():
    assert DATASET_PATH.exists(), "eval_dataset.json missing"


def test_dataset_has_cases(dataset):
    assert "test_cases" in dataset
    assert len(dataset["test_cases"]) >= 30, "Need at least 30 cases for eval"


def test_dataset_no_duplicate_ids(dataset):
    ids = [c["id"] for c in dataset["test_cases"]]
    assert len(ids) == len(set(ids)), "Duplicate case IDs found"


def test_dataset_schema(dataset):
    """Prottek case e required fields ache."""
    required = {"id", "category", "query", "expected_routing", "expected_citations", "expected_answer_contains", "should_refuse"}
    valid_routings = {"policy", "database", "escalation", "chitchat"}
    
    for case in dataset["test_cases"]:
        missing = required - set(case.keys())
        assert not missing, f"Case {case.get('id')} missing fields: {missing}"
        assert case["expected_routing"] in valid_routings, f"Invalid routing in {case['id']}"


def test_dataset_category_distribution(dataset):
    """Distribution check — sob category e reasonable cases ache."""
    categories = [c["category"] for c in dataset["test_cases"]]
    from collections import Counter
    counts = Counter(categories)
    
    # At least 5 policy, 5 escalation, 5 database
    assert counts.get("policy", 0) >= 5
    assert counts.get("escalation", 0) >= 5
    assert counts.get("database", 0) >= 5


def test_dataset_non_policy_no_citations_required(dataset):
    """Non-policy cases e citations empty thaka uchit."""
    for case in dataset["test_cases"]:
        if case["expected_routing"] in ("escalation", "chitchat", "database"):
            assert case["expected_citations"] == [], (
                f"{case['id']} ({case['expected_routing']}) should not expect citations"
            )