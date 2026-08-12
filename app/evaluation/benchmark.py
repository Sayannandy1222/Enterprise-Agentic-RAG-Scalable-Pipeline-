from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Sequence

from app.evaluation.retrieval import evaluate_retrieval
from app.embeddings.gemini import GeminiEmbedder
from app.ingestion.models import Document
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.service import RetrievalService
from app.retrieval.vector_store import InMemoryVectorStore


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = Path(__file__).parent / "data" / "rag_eval.json"

ITERATIONS = 10

DENSE_LIMIT = 16
BM25_LIMIT = 16
RRF_LIMIT = 8
RERANK_LIMIT = 5
TOP_K = 5

EMBEDDING_CACHE_SIZE = 256
RERANK_CACHE_SIZE = 256


# ============================================================================
# STATISTICS
# ============================================================================


def percentile(
    values: Sequence[float],
    percentile_value: float,
) -> float:
    """
    Calculate an interpolated percentile.

    This avoids the unstable behaviour of selecting a single
    array index and provides a more useful P50/P95 estimate.
    """

    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = percentile_value / 100.0 * (len(ordered) - 1)

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def calculate_stats(
    values: Sequence[float],
) -> dict[str, float]:
    if not values:
        return {
            "p50": 0.0,
            "p95": 0.0,
            "mean": 0.0,
        }

    return {
        "p50": float(median(values)),
        "p95": float(
            percentile(
                values,
                95.0,
            )
        ),
        "mean": float(mean(values)),
    }


# ============================================================================
# DATASET
# ============================================================================


def load_dataset() -> list[dict[str, Any]]:
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    cases = payload.get(
        "cases",
        [],
    )

    if not isinstance(
        cases,
        list,
    ):
        raise ValueError("'cases' must be a list")

    if not cases:
        raise ValueError("Evaluation dataset is empty.")

    return cases


# ============================================================================
# RETRIEVAL SERVICE
# ============================================================================


def build_retrieval_service() -> RetrievalService:
    """
    Construct the same production-style retrieval pipeline used
    by the application.

    The benchmark remains deterministic and local so that performance
    regressions can be measured without depending on external services.
    """

    embedder = GeminiEmbedder(
        cache_size=EMBEDDING_CACHE_SIZE,
    )

    documents = [
        Document(
            text=(
                "A DC motor converts electrical energy "
                "into mechanical energy. Armature conductors "
                "carrying current interact with the magnetic "
                "field and produce electromagnetic torque."
            ),
            source="dc-motor",
            metadata={
                "chunk_id": "chunk-a",
            },
        ),
        Document(
            text=(
                "The speed of a DC motor is affected by "
                "applied voltage, flux, back EMF, and "
                "armature resistance. Increasing applied "
                "voltage generally increases motor speed."
            ),
            source="dc-motor",
            metadata={
                "chunk_id": "chunk-b",
            },
        ),
        Document(
            text=(
                "Back EMF is the voltage generated in the "
                "armature of a rotating DC motor. It opposes "
                "the applied voltage and limits armature current."
            ),
            source="dc-motor",
            metadata={
                "chunk_id": "chunk-c",
            },
        ),
        Document(
            text=(
                "Armature resistance causes a voltage drop "
                "inside a DC motor and affects armature current "
                "and the resulting operating characteristics."
            ),
            source="dc-motor",
            metadata={
                "chunk_id": "chunk-d",
            },
        ),
        Document(
            text=(
                "The mechanical equation of a DC motor relates "
                "electromagnetic torque, load torque, inertia, "
                "angular velocity, and friction."
            ),
            source="dc-motor",
            metadata={
                "chunk_id": "chunk-e",
            },
        ),
    ]

    vector_store = InMemoryVectorStore(embedder)

    vector_store.upsert(documents)

    bm25 = BM25Retriever(documents)

    return RetrievalService(
        vector_store=vector_store,
        bm25=bm25,
        dense_limit=DENSE_LIMIT,
        bm25_limit=BM25_LIMIT,
        rrf_limit=RRF_LIMIT,
        rerank_limit=RERANK_LIMIT,
        top_k=TOP_K,
        embedding_cache_size=EMBEDDING_CACHE_SIZE,
        rerank_cache_size=RERANK_CACHE_SIZE,
    )


# ============================================================================
# DOCUMENT IDENTIFICATION
# ============================================================================


def extract_chunk_id(
    document: Any,
) -> str | None:

    if isinstance(
        document,
        dict,
    ):
        value = document.get("chunk_id")

        if value is not None:
            return str(value)

        metadata = document.get("metadata")

        if isinstance(
            metadata,
            dict,
        ):
            value = metadata.get("chunk_id")

            if value is not None:
                return str(value)

    metadata = getattr(
        document,
        "metadata",
        None,
    )

    if isinstance(
        metadata,
        dict,
    ):
        value = metadata.get("chunk_id")

        if value is not None:
            return str(value)

    value = getattr(
        document,
        "chunk_id",
        None,
    )

    if value is not None:
        return str(value)

    return None


# ============================================================================
# LATENCY BENCHMARK
# ============================================================================


def run_latency_benchmark(
    retrieval: RetrievalService,
    queries: Sequence[str],
) -> dict[str, Any]:

    stages = (
        "Embedding",
        "Dense/Qdrant",
        "BM25",
        "RRF",
        "Rerank Cache",
        "FlashRank",
        "TOTAL",
    )

    cold: dict[str, list[float]] = {stage: [] for stage in stages}

    warm: dict[str, list[float]] = {stage: [] for stage in stages}

    cold_embedding_hits = 0
    warm_embedding_hits = 0

    cold_rerank_hits = 0
    warm_rerank_hits = 0

    def execute(
        query: str,
        target: dict[str, list[float]],
    ) -> tuple[bool, bool]:

        result = retrieval.search_with_metrics(
            query=query,
            top_k=TOP_K,
        )

        target["Embedding"].append(float(result["embedding_latency_ms"]))

        target["Dense/Qdrant"].append(float(result["dense_latency_ms"]))

        target["BM25"].append(float(result["bm25_latency_ms"]))

        target["RRF"].append(float(result["rrf_latency_ms"]))

        target["Rerank Cache"].append(float(result["rerank_cache_lookup_latency_ms"]))

        target["FlashRank"].append(float(result["flashrank_execution_latency_ms"]))

        target["TOTAL"].append(float(result["total_latency_ms"]))

        return (
            bool(result["embedding_cache_hit"]),
            bool(result["rerank_cache_hit"]),
        )

    # ------------------------------------------------------------------
    # COLD CACHE
    # ------------------------------------------------------------------

    for _ in range(ITERATIONS):
        for query in queries:
            retrieval.clear_embedding_cache()
            retrieval.clear_rerank_cache()

            embedding_hit, rerank_hit = execute(
                query,
                cold,
            )

            if embedding_hit:
                cold_embedding_hits += 1

            if rerank_hit:
                cold_rerank_hits += 1

    # ------------------------------------------------------------------
    # WARM CACHE
    # ------------------------------------------------------------------

    retrieval.clear_embedding_cache()
    retrieval.clear_rerank_cache()

    for query in queries:
        execute(
            query,
            warm,
        )

    warm_embedding_hits = 0
    warm_rerank_hits = 0

    for _ in range(ITERATIONS):
        for query in queries:
            embedding_hit, rerank_hit = execute(
                query,
                warm,
            )

            if embedding_hit:
                warm_embedding_hits += 1

            if rerank_hit:
                warm_rerank_hits += 1

    return {
        "cold": {stage: calculate_stats(values) for stage, values in cold.items()},
        "warm": {stage: calculate_stats(values) for stage, values in warm.items()},
        "cold_samples": len(cold["TOTAL"]),
        "warm_samples": len(warm["TOTAL"]),
        "cold_cache_hits": cold_embedding_hits,
        "warm_cache_hits": warm_embedding_hits,
        "cold_rerank_cache_hits": cold_rerank_hits,
        "warm_rerank_cache_hits": warm_rerank_hits,
    }


# ============================================================================
# QUALITY BENCHMARK
# ============================================================================


def run_quality_benchmark(
    retrieval: RetrievalService,
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:

    retrieved: list[list[str]] = []

    relevant: list[list[str]] = []

    per_case: list[dict[str, Any]] = []

    for case in cases:
        query = str(case["query"])

        expected = [str(chunk_id) for chunk_id in case["relevant_chunk_ids"]]

        result = retrieval.search_with_metrics(
            query=query,
            top_k=TOP_K,
        )

        results = result["results"]

        retrieved_ids: list[str] = []

        for document in results:
            chunk_id = extract_chunk_id(document)

            if chunk_id is not None:
                retrieved_ids.append(chunk_id)

        retrieved.append(retrieved_ids)

        relevant.append(expected)

        case_metrics = evaluate_retrieval(
            [retrieved_ids],
            [expected],
            ks=(3, 5),
        )

        per_case.append(
            {
                "id": str(case["id"]),
                "recall_at_3": (case_metrics["recall_at_3"]),
                "recall_at_5": (case_metrics["recall_at_5"]),
                "mrr": case_metrics["mrr"],
                "retrieved": (retrieved_ids),
            }
        )

    overall = evaluate_retrieval(
        retrieved,
        relevant,
        ks=(3, 5),
    )

    return {
        "metrics": overall,
        "cases": per_case,
    }


# ============================================================================
# REPORTING
# ============================================================================


def print_latency_table(
    title: str,
    metrics: dict[
        str,
        dict[str, float],
    ],
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(f"{'Stage':<22}{'P50 (ms)':>14}{'P95 (ms)':>14}{'Mean (ms)':>14}")

    print("-" * 80)

    for stage, values in metrics.items():
        print(
            f"{stage:<22}"
            f"{values['p50']:>14.3f}"
            f"{values['p95']:>14.3f}"
            f"{values['mean']:>14.3f}"
        )


def print_report(
    latency: dict[str, Any],
    quality: dict[str, Any],
    cache: dict[str, Any],
) -> None:

    print_latency_table(
        "COLD CACHE LATENCY",
        latency["cold"],
    )

    print_latency_table(
        "WARM CACHE LATENCY",
        latency["warm"],
    )

    print()
    print("=" * 80)
    print("CACHE")
    print("=" * 80)

    print(f"Cold samples:            {latency['cold_samples']}")

    print(f"Warm samples:            {latency['warm_samples']}")

    print(f"Cold embedding hits:     {latency['cold_cache_hits']}")

    print(f"Warm embedding hits:     {latency['warm_cache_hits']}")

    print(f"Cold rerank hits:        {latency['cold_rerank_cache_hits']}")

    print(f"Warm rerank hits:        {latency['warm_rerank_cache_hits']}")

    print()

    rerank_cache = latency.get("rerank_cache", {})

    if rerank_cache:
        print(f"Rerank cache size:       {rerank_cache['size']}")

        print(f"Rerank cache capacity:   {rerank_cache['capacity']}")

        print(f"Rerank cache hit rate:   {rerank_cache['hit_rate'] * 100:.2f}%")

    print()
    print("=" * 80)
    print("RETRIEVAL QUALITY")
    print("=" * 80)

    metrics = quality["metrics"]

    print(f"Recall@3:                {metrics['recall_at_3']:.4f}")

    print(f"Recall@5:                {metrics['recall_at_5']:.4f}")

    print(f"MRR:                     {metrics['mrr']:.4f}")

    print()
    print("=" * 80)
    print("EMBEDDING CACHE")
    print("=" * 80)

    print(f"Cache size:              {cache['size']}")

    print(f"Cache capacity:          {cache['capacity']}")

    print(f"Cache hits:              {cache['hits']}")

    print(f"Cache misses:            {cache['misses']}")

    print(f"Cache hit rate:          {cache['hit_rate'] * 100:.2f}%")

    print()
    print("=" * 80)
    print("PER-CASE RESULTS")
    print("=" * 80)

    for case in quality["cases"]:
        print(
            f"{case['id']}: "
            f"Recall@3="
            f"{case['recall_at_3']:.4f}, "
            f"Recall@5="
            f"{case['recall_at_5']:.4f}, "
            f"MRR="
            f"{case['mrr']:.4f}"
        )

        print("  Retrieved: " + ", ".join(case["retrieved"]))

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    print("=" * 80)
    print("PRODUCTION RAG RETRIEVAL BENCHMARK")
    print("=" * 80)

    cases = load_dataset()

    queries = [str(case["query"]) for case in cases]

    retrieval = build_retrieval_service()

    print()
    print("Configuration")
    print("-" * 80)

    print(f"Dense candidates:       {DENSE_LIMIT}")

    print(f"BM25 candidates:        {BM25_LIMIT}")

    print(f"RRF candidates:         {RRF_LIMIT}")

    print(f"FlashRank results:      {RERANK_LIMIT}")

    print(f"Embedding cache:        {EMBEDDING_CACHE_SIZE}")

    print(f"Iterations:             {ITERATIONS}")

    print()
    print("Running latency benchmark...")

    latency = run_latency_benchmark(
        retrieval,
        queries,
    )

    print("Running retrieval-quality benchmark...")

    quality = run_quality_benchmark(
        retrieval,
        cases,
    )

    cache = retrieval.embedding_cache_stats()

    print_report(
        latency,
        quality,
        cache,
    )


if __name__ == "__main__":
    main()
