from __future__ import annotations

import pytest

from app.retrieval.service import RetrievalService


class DummyVectorStore:
    embedder = None


class DummyBM25:
    pass


def make_service(**overrides):
    config = {
        "vector_store": DummyVectorStore(),
        "bm25": DummyBM25(),
        "dense_limit": 16,
        "bm25_limit": 16,
        "rrf_limit": 12,
        "rerank_limit": 8,
        "top_k": 5,
    }

    config.update(overrides)

    return RetrievalService(**config)


def test_dense_limit_must_cover_top_k():
    with pytest.raises(ValueError, match="dense_limit must be >= top_k"):
        make_service(
            dense_limit=4,
            top_k=5,
        )


def test_bm25_limit_must_cover_top_k():
    with pytest.raises(ValueError, match="bm25_limit must be >= top_k"):
        make_service(
            bm25_limit=4,
            top_k=5,
        )


def test_rrf_limit_must_cover_top_k():
    with pytest.raises(ValueError, match="rrf_limit must be >= top_k"):
        make_service(
            rrf_limit=4,
            top_k=5,
        )


def test_rerank_limit_must_cover_top_k():
    with pytest.raises(ValueError, match="rerank_limit must be >= top_k"):
        make_service(
            rerank_limit=4,
            top_k=5,
        )


def test_rerank_limit_cannot_exceed_rrf_limit():
    with pytest.raises(
        ValueError,
        match="rerank_limit cannot exceed rrf_limit",
    ):
        make_service(
            rrf_limit=8,
            rerank_limit=12,
        )


def test_rrf_limit_cannot_exceed_upstream_capacity():
    with pytest.raises(
        ValueError,
        match="rrf_limit cannot exceed dense_limit",
    ):
        make_service(
            dense_limit=4,
            bm25_limit=4,
            rrf_limit=9,
            top_k=4,
        )


def test_valid_production_configuration():
    service = make_service()

    assert service.dense_limit == 16
    assert service.bm25_limit == 16
    assert service.rrf_limit == 12
    assert service.rerank_limit == 8
    assert service.top_k == 5
