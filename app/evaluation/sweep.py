from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from app.evaluation.retrieval import evaluate_retrieval
from app.evaluation.runner import (
    EMBEDDING_CACHE_SIZE,
    RERANK_CACHE_SIZE,
    TOP_K,
    build_retrieval_service,
    extract_chunk_id,
    load_dataset,
)
from app.retrieval.service import RetrievalService


@dataclass(frozen=True)
class RetrievalConfig:
    dense_limit: int
    bm25_limit: int
    rrf_limit: int
    rerank_limit: int
    dense_weight: float
    lexical_weight: float

    def validate(self, top_k: int = TOP_K) -> None:
        if self.dense_limit < top_k:
            raise ValueError("dense_limit must be >= top_k")

        if self.bm25_limit < top_k:
            raise ValueError("bm25_limit must be >= top_k")

        if self.rrf_limit < top_k:
            raise ValueError("rrf_limit must be >= top_k")

        if self.rerank_limit < top_k:
            raise ValueError("rerank_limit must be >= top_k")

        if self.rerank_limit > self.rrf_limit:
            raise ValueError("rerank_limit cannot exceed rrf_limit")

        if self.rrf_limit > self.dense_limit + self.bm25_limit:
            raise ValueError("rrf_limit cannot exceed retrieval candidates")

        if self.dense_weight <= 0:
            raise ValueError("dense_weight must be greater than zero")

        if self.lexical_weight <= 0:
            raise ValueError("lexical_weight must be greater than zero")


@dataclass(frozen=True)
class SweepResult:
    config: RetrievalConfig
    recall_at_3: float
    recall_at_5: float
    mrr: float
    mean_latency_ms: float
    p95_latency_ms: float

    @property
    def quality_score(self) -> tuple[float, float, float]:
        return (
            self.recall_at_3,
            self.mrr,
            self.recall_at_5,
        )

    @property
    def candidate_budget(self) -> int:
        return (
            self.config.dense_limit
            + self.config.bm25_limit
            + self.config.rrf_limit
            + self.config.rerank_limit
        )


def _percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    index = int((len(ordered) - 1) * percentile)

    return ordered[index]


def _build_service(
    base_service: RetrievalService,
    config: RetrievalConfig,
) -> RetrievalService:
    """
    Create an isolated RetrievalService sharing the already-built
    offline retrieval stores.

    This prevents every sweep point from rebuilding embeddings.
    """

    config.validate()

    return RetrievalService(
        base_service.vector_store,
        base_service.bm25,
        reranker=base_service.reranker,
        dense_limit=config.dense_limit,
        bm25_limit=config.bm25_limit,
        rrf_limit=config.rrf_limit,
        rerank_limit=config.rerank_limit,
        top_k=TOP_K,
        embedding_cache_size=EMBEDDING_CACHE_SIZE,
        rerank_cache_size=RERANK_CACHE_SIZE,
    )


def _search_with_weighted_rrf(
    service: RetrievalService,
    query: str,
    config: RetrievalConfig,
) -> dict[str, Any]:
    """
    Execute the service with the requested RRF weighting.

    The production service currently owns the complete pipeline, so
    this function temporarily applies the sweep's RRF weights at the
    fusion boundary without changing production configuration.
    """

    import app.retrieval.service as retrieval_service_module

    original = retrieval_service_module.reciprocal_rank_fusion

    def weighted_rrf(
        dense,
        lexical,
        limit=8,
        k=60,
        **kwargs,
    ):
        return original(
            dense,
            lexical,
            limit=limit,
            k=k,
            dense_weight=config.dense_weight,
            lexical_weight=config.lexical_weight,
        )

    retrieval_service_module.reciprocal_rank_fusion = weighted_rrf

    try:
        return service.search_with_metrics(
            query=query,
            top_k=TOP_K,
        )
    finally:
        retrieval_service_module.reciprocal_rank_fusion = original


def evaluate_configuration(
    base_service: RetrievalService,
    cases: list[dict[str, Any]],
    config: RetrievalConfig,
) -> SweepResult:
    """
    Evaluate one retrieval configuration over the complete dataset.
    """

    config.validate()

    service = _build_service(
        base_service,
        config,
    )

    retrieved: list[list[str]] = []
    relevant: list[list[str]] = []
    latencies: list[float] = []

    for case in cases:
        started = perf_counter()

        result = _search_with_weighted_rrf(
            service,
            case["query"],
            config,
        )

        elapsed_ms = (perf_counter() - started) * 1000

        latencies.append(elapsed_ms)

        retrieved_ids: list[str] = []

        for document in result["results"]:
            chunk_id = extract_chunk_id(document)

            if chunk_id is not None:
                retrieved_ids.append(chunk_id)

        retrieved.append(retrieved_ids)
        relevant.append(case["relevant_chunk_ids"])

    metrics = evaluate_retrieval(
        retrieved,
        relevant,
        ks=(3, 5),
    )

    return SweepResult(
        config=config,
        recall_at_3=metrics["recall_at_3"],
        recall_at_5=metrics["recall_at_5"],
        mrr=metrics["mrr"],
        mean_latency_ms=mean(latencies),
        p95_latency_ms=_percentile(
            latencies,
            0.95,
        ),
    )


def generate_configurations() -> list[RetrievalConfig]:
    """
    Generate a deliberately bounded search space.

    The sweep is large enough to explore retrieval-quality tradeoffs
    without creating an uncontrolled combinatorial explosion.
    """

    dense_limits = (16, 24, 32)
    bm25_limits = (16, 24, 32)
    rrf_limits = (8, 12, 16, 24)
    rerank_limits = (8, 12)

    weight_pairs = (
        (1.0, 1.0),
        (1.25, 1.0),
        (1.5, 1.0),
        (1.0, 1.25),
        (1.0, 1.5),
        (1.5, 1.25),
        (1.25, 1.5),
    )

    configurations: list[RetrievalConfig] = []

    for values in product(
        dense_limits,
        bm25_limits,
        rrf_limits,
        rerank_limits,
        weight_pairs,
    ):
        (
            dense_limit,
            bm25_limit,
            rrf_limit,
            rerank_limit,
            weights,
        ) = values

        dense_weight, lexical_weight = weights

        config = RetrievalConfig(
            dense_limit=dense_limit,
            bm25_limit=bm25_limit,
            rrf_limit=rrf_limit,
            rerank_limit=rerank_limit,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
        )

        try:
            config.validate()
        except ValueError:
            continue

        configurations.append(config)

    return configurations


def select_best(
    results: Sequence[SweepResult],
) -> SweepResult:
    """
    Deterministically select the strongest configuration.

    Priority:
        1. Recall@3
        2. MRR
        3. Recall@5
        4. lower P95 latency
        5. lower candidate budget
    """

    if not results:
        raise ValueError("cannot select from empty sweep results")

    return max(
        results,
        key=lambda result: (
            result.recall_at_3,
            result.mrr,
            result.recall_at_5,
            -result.p95_latency_ms,
            -result.candidate_budget,
        ),
    )


def main() -> None:
    print("=" * 80)
    print("AUTOMATED RETRIEVAL PARAMETER SWEEP")
    print("=" * 80)

    cases = load_dataset()

    print(f"Evaluation cases: {len(cases)}")

    base_service = build_retrieval_service()

    configurations = generate_configurations()

    print(f"Valid configurations: {len(configurations)}")

    results: list[SweepResult] = []

    for index, config in enumerate(
        configurations,
        start=1,
    ):
        result = evaluate_configuration(
            base_service,
            cases,
            config,
        )

        results.append(result)

        print(
            f"[{index:03d}/{len(configurations):03d}] "
            f"D={config.dense_limit:2d} "
            f"B={config.bm25_limit:2d} "
            f"RRF={config.rrf_limit:2d} "
            f"RR={config.rerank_limit:2d} "
            f"W={config.dense_weight:.2f}/"
            f"{config.lexical_weight:.2f} "
            f"| R3={result.recall_at_3:.4f} "
            f"R5={result.recall_at_5:.4f} "
            f"MRR={result.mrr:.4f} "
            f"P95={result.p95_latency_ms:.2f}ms"
        )

    best = select_best(results)

    print()
    print("=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)

    print(f"Dense candidates:    {best.config.dense_limit}")
    print(f"BM25 candidates:     {best.config.bm25_limit}")
    print(f"RRF candidates:      {best.config.rrf_limit}")
    print(f"Rerank candidates:   {best.config.rerank_limit}")
    print(f"Dense weight:        {best.config.dense_weight:.2f}")
    print(f"Lexical weight:      {best.config.lexical_weight:.2f}")

    print()
    print(f"Recall@3:            {best.recall_at_3:.4f}")
    print(f"Recall@5:            {best.recall_at_5:.4f}")
    print(f"MRR:                 {best.mrr:.4f}")
    print(f"Mean latency:        {best.mean_latency_ms:.2f} ms")
    print(f"P95 latency:         {best.p95_latency_ms:.2f} ms")

    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
