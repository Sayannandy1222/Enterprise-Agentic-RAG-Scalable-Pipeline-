from __future__ import annotations

import pytest

from app.evaluation.sweep import (
    RetrievalConfig,
    SweepResult,
    generate_configurations,
    select_best,
)


def test_configuration_rejects_invalid_candidate_order():
    with pytest.raises(
        ValueError,
        match="rerank_limit cannot exceed rrf_limit",
    ):
        RetrievalConfig(
            dense_limit=16,
            bm25_limit=16,
            rrf_limit=8,
            rerank_limit=12,
            dense_weight=1.0,
            lexical_weight=1.0,
        ).validate()


def test_configuration_rejects_insufficient_dense_budget():
    with pytest.raises(
        ValueError,
        match="dense_limit must be >= top_k",
    ):
        RetrievalConfig(
            dense_limit=4,
            bm25_limit=16,
            rrf_limit=8,
            rerank_limit=8,
            dense_weight=1.0,
            lexical_weight=1.0,
        ).validate()


def test_configuration_rejects_invalid_weight():
    with pytest.raises(
        ValueError,
        match="dense_weight",
    ):
        RetrievalConfig(
            dense_limit=16,
            bm25_limit=16,
            rrf_limit=8,
            rerank_limit=8,
            dense_weight=0.0,
            lexical_weight=1.0,
        ).validate()


def test_configuration_rejects_excessive_rrf_budget():
    with pytest.raises(
        ValueError,
        match="rrf_limit cannot exceed retrieval candidates",
    ):
        RetrievalConfig(
            dense_limit=8,
            bm25_limit=8,
            rrf_limit=20,
            rerank_limit=8,
            dense_weight=1.0,
            lexical_weight=1.0,
        ).validate()


def test_generated_configurations_are_valid():
    configurations = generate_configurations()

    assert configurations

    for configuration in configurations:
        configuration.validate()


def test_generated_configurations_are_unique():
    configurations = generate_configurations()

    assert len(configurations) == len(set(configurations))


def test_select_best_is_deterministic():
    results = [
        SweepResult(
            config=RetrievalConfig(
                16,
                16,
                8,
                8,
                1.0,
                1.0,
            ),
            recall_at_3=0.75,
            recall_at_5=0.80,
            mrr=0.50,
            mean_latency_ms=100.0,
            p95_latency_ms=150.0,
        ),
        SweepResult(
            config=RetrievalConfig(
                32,
                32,
                24,
                12,
                1.5,
                1.0,
            ),
            recall_at_3=0.80,
            recall_at_5=0.85,
            mrr=0.60,
            mean_latency_ms=120.0,
            p95_latency_ms=180.0,
        ),
    ]

    best = select_best(results)

    assert best.recall_at_3 == 0.80
    assert best.mrr == 0.60


def test_selection_prefers_lower_latency_when_quality_ties():
    slow = SweepResult(
        config=RetrievalConfig(
            32,
            32,
            24,
            12,
            1.0,
            1.0,
        ),
        recall_at_3=0.80,
        recall_at_5=0.85,
        mrr=0.60,
        mean_latency_ms=200.0,
        p95_latency_ms=300.0,
    )

    fast = SweepResult(
        config=RetrievalConfig(
            16,
            16,
            12,
            8,
            1.0,
            1.0,
        ),
        recall_at_3=0.80,
        recall_at_5=0.85,
        mrr=0.60,
        mean_latency_ms=100.0,
        p95_latency_ms=150.0,
    )

    assert select_best([slow, fast]) == fast
