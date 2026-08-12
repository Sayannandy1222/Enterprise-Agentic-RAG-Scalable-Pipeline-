from __future__ import annotations

import hashlib

from collections import deque
from statistics import median
from time import perf_counter
from typing import Any, Sequence

from app.reranking.flashrank import FlashRankReranker
from app.retrieval.cache import BoundedLRUCache
from app.retrieval.fusion import reciprocal_rank_fusion


class RetrievalService:
    """
    Production-oriented hybrid retrieval service.

    Pipeline:

        Query
          |
          v
        Embedding cache
          |
          v
        Dense / Qdrant
          +
        BM25
          |
          v
        Reciprocal Rank Fusion
          |
          v
        Candidate pruning
          |
          v
        FlashRank
          |
          v
        Final Top-K

    The service records per-stage latency for production
    observability and benchmarking.
    """

    def __init__(
        self,
        vector_store,
        bm25,
        reranker=None,
        dense_limit: int = 16,
        bm25_limit: int = 16,
        rrf_limit: int = 8,
        rerank_limit: int = 8,
        top_k: int = 5,
        embedding_cache_size: int = 256,
        rerank_cache_size: int = 256,
    ) -> None:
        if dense_limit <= 0:
            raise ValueError("dense_limit must be greater than zero")

        if bm25_limit <= 0:
            raise ValueError("bm25_limit must be greater than zero")

        if rrf_limit <= 0:
            raise ValueError("rrf_limit must be greater than zero")

        if rerank_limit <= 0:
            raise ValueError("rerank_limit must be greater than zero")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if dense_limit < top_k:
            raise ValueError("dense_limit must be >= top_k")

        if bm25_limit < top_k:
            raise ValueError("bm25_limit must be >= top_k")

        if rrf_limit < top_k:
            raise ValueError("rrf_limit must be >= top_k")

        if rerank_limit < top_k:
            raise ValueError("rerank_limit must be >= top_k")

        if rerank_limit > rrf_limit:
            raise ValueError("rerank_limit cannot exceed rrf_limit")

        if rrf_limit > dense_limit + bm25_limit:
            raise ValueError("rrf_limit cannot exceed dense_limit + bm25_limit")

        if embedding_cache_size <= 0:
            raise ValueError("embedding_cache_size must be greater than zero")

        if rerank_cache_size <= 0:
            raise ValueError("rerank_cache_size must be greater than zero")

        self.vector_store = vector_store
        self.bm25 = bm25

        self.reranker = reranker if reranker is not None else FlashRankReranker()

        self.dense_limit = dense_limit
        self.bm25_limit = bm25_limit
        self.rrf_limit = rrf_limit
        self.rerank_limit = rerank_limit
        self.top_k = top_k

        self.embedding_cache_size = embedding_cache_size

        self._embedding_cache: dict[
            str,
            list[float],
        ] = {}

        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0

        self.rerank_cache = BoundedLRUCache[list[Any]](rerank_cache_size)

        self.latencies: deque[tuple[str, float]] = deque(maxlen=10_000)

    # ------------------------------------------------------------------
    # EMBEDDING CACHE
    # ------------------------------------------------------------------

    def _get_query_embedding(
        self,
        query: str,
    ) -> tuple[list[float], bool]:
        """
        Get an embedding from the retrieval-level query cache.

        Returns:
            (embedding, cache_hit)
        """

        key = query.strip()

        cached = self._embedding_cache.get(key)

        if cached is not None:
            self.embedding_cache_hits += 1

            return list(cached), True

        self.embedding_cache_misses += 1

        embedder = getattr(
            self.vector_store,
            "embedder",
            None,
        )

        if embedder is None:
            raise RuntimeError("vector_store must expose an embedder")

        embedding = embedder.embed(key)

        # Simple bounded FIFO eviction.
        if len(self._embedding_cache) >= (self.embedding_cache_size):
            oldest_key = next(iter(self._embedding_cache))

            del self._embedding_cache[oldest_key]

        self._embedding_cache[key] = list(embedding)

        return list(embedding), False

    def clear_embedding_cache(self) -> None:
        """
        Clear the retrieval-level embedding cache.

        Used by cold-cache benchmarks and operational cache
        invalidation.
        """

        self._embedding_cache.clear()

    def embedding_cache_stats(
        self,
    ) -> dict[str, Any]:
        """
        Return embedding-cache statistics.
        """

        hits = self.embedding_cache_hits
        misses = self.embedding_cache_misses
        total = hits + misses

        hit_rate = hits / total if total > 0 else 0.0

        return {
            "size": len(self._embedding_cache),
            "capacity": self.embedding_cache_size,
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
        }

    # ------------------------------------------------------------------
    # RERANK CACHE
    # ------------------------------------------------------------------

    @staticmethod
    def _document_identity(document: Any) -> str:
        """Return a stable identity for a retrieval candidate."""

        if hasattr(document, "metadata"):
            metadata = getattr(document, "metadata", {}) or {}

            for key in (
                "chunk_id",
                "id",
                "document_id",
                "source",
            ):
                value = metadata.get(key)
                if value is not None:
                    return str(value)

        if hasattr(document, "text"):
            text = str(document.text)
        else:
            text = str(document)

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _rerank_cache_key(
        self,
        query: str,
        documents: Sequence[Any],
        limit: int,
    ) -> str:
        candidate_ids = "|".join(
            self._document_identity(document) for document in documents
        )

        raw = f"{query.strip()}::{limit}::{candidate_ids}"

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def clear_rerank_cache(self) -> None:
        """Clear the reranking cache."""

        self.rerank_cache.clear()

    def rerank_cache_stats(self) -> dict[str, int | float]:
        """Return reranking-cache statistics."""

        return self.rerank_cache.stats()

    def clear_caches(self) -> None:
        """Clear retrieval embedding and reranking caches."""

        self.clear_embedding_cache()
        self.clear_rerank_cache()

    # ------------------------------------------------------------------
    # LATENCY
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(
        values: Sequence[float],
        percentile: float,
    ) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)

        index = int((len(ordered) - 1) * percentile)

        return ordered[index]

    def _record_latency(
        self,
        stage: str,
        started: float,
    ) -> float:
        latency_ms = (perf_counter() - started) * 1000

        self.latencies.append(
            (
                stage,
                latency_ms,
            )
        )

        return latency_ms

    def metrics(self) -> dict[str, Any]:
        """
        Return latency statistics.

        Metric names:

            embedding
            dense
            bm25
            rrf
            flashrank
            rerank
            total

        `rerank` is intentionally retained as a backward-compatible
        alias for `flashrank`.
        """

        grouped: dict[
            str,
            list[float],
        ] = {}

        for stage, latency in self.latencies:
            grouped.setdefault(
                stage,
                [],
            ).append(latency)

        output: dict[str, Any] = {}

        for stage, values in grouped.items():
            output[stage] = {
                "count": len(values),
                "p50": round(
                    median(values),
                    3,
                ),
                "p95": round(
                    self._percentile(
                        values,
                        0.95,
                    ),
                    3,
                ),
                "mean": round(
                    sum(values) / len(values),
                    3,
                ),
            }

        # Preserve the original API/test contract.
        if "flashrank" in output:
            output["rerank"] = output["flashrank"]

        return output

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Any]:
        """
        Execute hybrid retrieval and return final results.
        """

        result = self.search_with_metrics(
            query=query,
            top_k=top_k,
        )

        return result["results"]

    def search_with_metrics(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute the complete hybrid retrieval pipeline.

        Returns final results plus detailed stage-level metrics.
        """

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        requested_top_k = top_k if top_k is not None else self.top_k

        if requested_top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # ==============================================================
        # 1. EMBEDDING
        # ==============================================================

        embedding_started = perf_counter()

        (
            query_vector,
            embedding_cache_hit,
        ) = self._get_query_embedding(query)

        embedding_latency_ms = self._record_latency(
            "embedding",
            embedding_started,
        )

        # ==============================================================
        # 2. DENSE RETRIEVAL
        # ==============================================================

        dense_started = perf_counter()

        dense_results = self.vector_store.search(
            query_vector=query_vector,
            limit=self.dense_limit,
        )

        dense_latency_ms = self._record_latency(
            "dense",
            dense_started,
        )

        # ==============================================================
        # 3. BM25
        # ==============================================================

        bm25_started = perf_counter()

        lexical_results = self.bm25.search(
            query,
            limit=self.bm25_limit,
        )

        bm25_latency_ms = self._record_latency(
            "bm25",
            bm25_started,
        )

        # ==============================================================
        # RETRIEVAL HEALTH GUARDRAIL
        # ==============================================================

        if not dense_results and not lexical_results:
            raise RuntimeError("retrieval indexes returned zero candidates")

        # ==============================================================
        # 4. RRF
        # ==============================================================

        rrf_started = perf_counter()

        hybrid_candidates = reciprocal_rank_fusion(
            dense_results,
            lexical_results,
            limit=self.rrf_limit,
        )

        rrf_latency_ms = self._record_latency(
            "rrf",
            rrf_started,
        )

        # ==============================================================
        # RRF HEALTH GUARDRAIL
        # ==============================================================

        if not hybrid_candidates:
            raise RuntimeError("RRF produced zero candidates")

        # RRF must never exceed its configured capacity.
        if len(hybrid_candidates) > self.rrf_limit:
            raise RuntimeError("RRF returned more candidates than configured")

        # ==============================================================
        # 5. CANDIDATE PRUNING
        # ==============================================================

        candidate_limit = max(
            requested_top_k,
            self.rerank_limit,
        )

        pruned_candidates = hybrid_candidates[:candidate_limit]

        # ==============================================================
        # 6. RERANK CACHE + FLASHRANK
        # ==============================================================

        rerank_stage_started = perf_counter()

        rerank_cache_key = self._rerank_cache_key(
            query,
            pruned_candidates,
            requested_top_k,
        )

        rerank_cache_started = perf_counter()

        cached_reranked, rerank_cache_hit = self.rerank_cache.get(rerank_cache_key)

        rerank_cache_lookup_latency_ms = self._record_latency(
            "rerank_cache",
            rerank_cache_started,
        )

        flashrank_execution_latency_ms = 0.0

        if rerank_cache_hit and cached_reranked is not None:
            reranked = list(cached_reranked)

        else:
            flashrank_started = perf_counter()

            reranked = self.reranker.rerank(
                query,
                pruned_candidates,
                limit=requested_top_k,
            )

            flashrank_execution_latency_ms = self._record_latency(
                "flashrank",
                flashrank_started,
            )

            self.rerank_cache.set(
                rerank_cache_key,
                list(reranked),
            )

        rerank_latency_ms = (perf_counter() - rerank_stage_started) * 1000

        results = reranked[:requested_top_k]

        # ==============================================================
        # FINAL RESULT GUARDRAILS
        # ==============================================================

        if not results:
            raise RuntimeError("reranking produced zero final results")

        if len(results) > requested_top_k:
            raise RuntimeError("retrieval returned more results than requested top_k")

        # ==============================================================
        # 7. TOTAL
        # ==============================================================

        total_latency_ms = (
            embedding_latency_ms
            + dense_latency_ms
            + bm25_latency_ms
            + rrf_latency_ms
            + rerank_latency_ms
        )

        self.latencies.append(
            (
                "total",
                total_latency_ms,
            )
        )

        return {
            "results": results,
            "embedding_latency_ms": round(
                embedding_latency_ms,
                3,
            ),
            "dense_latency_ms": round(
                dense_latency_ms,
                3,
            ),
            "bm25_latency_ms": round(
                bm25_latency_ms,
                3,
            ),
            "rrf_latency_ms": round(
                rrf_latency_ms,
                3,
            ),
            "reranking_latency_ms": round(
                rerank_latency_ms,
                3,
            ),
            "rerank_cache_lookup_latency_ms": round(
                rerank_cache_lookup_latency_ms,
                3,
            ),
            "flashrank_execution_latency_ms": round(
                flashrank_execution_latency_ms,
                3,
            ),
            "total_latency_ms": round(
                total_latency_ms,
                3,
            ),
            "embedding_cache_hit": (embedding_cache_hit),
            "rerank_cache_hit": (rerank_cache_hit),
            "dense_candidate_count": len(dense_results),
            "bm25_candidate_count": len(lexical_results),
            "hybrid_candidate_count": len(hybrid_candidates),
            "rerank_candidate_count": len(pruned_candidates),
            "result_count": len(results),
        }
