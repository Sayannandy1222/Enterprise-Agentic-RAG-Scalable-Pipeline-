from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(
    results: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    k: int,
) -> float:
    """
    Calculate Recall@K over multiple evaluation cases.

    A case counts as a hit when at least one relevant document
    appears in the first K retrieved documents.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not results:
        return 0.0

    if len(results) != len(relevant):
        raise ValueError("results and relevant must have the same length")

    hits = 0

    for retrieved, relevant_ids in zip(
        results,
        relevant,
    ):
        relevant_set = set(relevant_ids)

        retrieved_top_k = set(retrieved[:k])

        if retrieved_top_k & relevant_set:
            hits += 1

    return hits / len(results)


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: Iterable[str],
) -> float:
    """
    Calculate reciprocal rank for one evaluation case.

    Returns:
        1 / rank of the first relevant result,
        or 0.0 when no relevant result is retrieved.
    """

    relevant_set = set(relevant)

    for index, document_id in enumerate(
        retrieved,
        start=1,
    ):
        if document_id in relevant_set:
            return 1.0 / index

    return 0.0


def mean_reciprocal_rank(
    results: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
) -> float:
    """
    Calculate Mean Reciprocal Rank across evaluation cases.
    """

    if not results:
        return 0.0

    if len(results) != len(relevant):
        raise ValueError("results and relevant must have the same length")

    scores = [
        reciprocal_rank(
            retrieved,
            relevant_ids,
        )
        for retrieved, relevant_ids in zip(
            results,
            relevant,
        )
    ]

    return sum(scores) / len(scores)


def mrr(
    results: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
) -> float:
    """
    Backward-compatible alias for mean_reciprocal_rank().
    """

    return mean_reciprocal_rank(
        results,
        relevant,
    )


def evaluate_retrieval(
    results: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    ks: Sequence[int] = (3, 5),
) -> dict[str, float]:
    """
    Calculate the standard retrieval-quality metrics.
    """

    metrics: dict[str, float] = {}

    for k in ks:
        metrics[f"recall_at_{k}"] = recall_at_k(
            results,
            relevant,
            k,
        )

    metrics["mrr"] = mean_reciprocal_rank(
        results,
        relevant,
    )

    return metrics
