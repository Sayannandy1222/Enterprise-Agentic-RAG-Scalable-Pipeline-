from __future__ import annotations

import pytest

from app.evaluation.retrieval import (
    evaluate_retrieval,
    mrr,
    recall_at_k,
)


def test_recall_at_k_returns_one_for_hit():
    results = [
        ["chunk-a", "chunk-b", "chunk-c"],
    ]

    relevant = [
        {"chunk-a"},
    ]

    assert recall_at_k(
        results,
        relevant,
        3,
    ) == pytest.approx(1.0)


def test_recall_at_k_returns_zero_for_miss():
    results = [
        ["chunk-x", "chunk-y", "chunk-z"],
    ]

    relevant = [
        {"chunk-a"},
    ]

    assert recall_at_k(
        results,
        relevant,
        3,
    ) == pytest.approx(0.0)


def test_mrr_uses_first_relevant_rank():
    results = [
        ["chunk-x", "chunk-b", "chunk-a"],
    ]

    relevant = [
        {"chunk-a"},
    ]

    assert mrr(
        results,
        relevant,
    ) == pytest.approx(1.0 / 3.0)


def test_evaluate_retrieval_returns_expected_structure():
    results = [
        ["chunk-a", "chunk-x"],
        ["chunk-x", "chunk-b"],
    ]

    relevant = [
        {"chunk-a"},
        {"chunk-b"},
    ]

    result = evaluate_retrieval(
        results,
        relevant,
    )

    assert isinstance(
        result,
        dict,
    )

    assert "recall_at_3" in result
    assert "recall_at_5" in result
    assert "mrr" in result

    assert result["recall_at_3"] == pytest.approx(1.0)

    assert result["recall_at_5"] == pytest.approx(1.0)

    # Case 1: relevant document at rank 1 -> 1.0
    # Case 2: relevant document at rank 2 -> 0.5
    # Mean reciprocal rank = (1.0 + 0.5) / 2 = 0.75
    assert result["mrr"] == pytest.approx(0.75)
