from __future__ import annotations

import pytest

from app.embeddings.gemini import GeminiEmbedder
from app.ingestion.models import Document
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.vector_store import InMemoryVectorStore


def test_bm25_respects_result_limit():
    documents = [
        Document("alpha beta"),
        Document("alpha gamma"),
        Document("alpha delta"),
        Document("unrelated content"),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.search(
        "alpha",
        limit=2,
    )

    assert len(results) <= 2


def test_bm25_returns_relevant_document():
    documents = [
        Document("DC motor converts electrical energy into mechanical energy."),
        Document("Transformer transfers electrical energy between circuits."),
        Document("Induction motor uses electromagnetic induction."),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.search(
        "DC motor",
        limit=3,
    )

    assert results

    assert any("DC motor" in document.text for _, document in results)


def test_rrf_preserves_high_ranked_candidates():
    dense = [
        (1.0, Document("alpha")),
        (0.8, Document("beta")),
        (0.6, Document("gamma")),
    ]

    lexical = [
        (1.0, dense[0][1]),
        (0.7, Document("delta")),
        (0.5, Document("epsilon")),
    ]

    results = reciprocal_rank_fusion(
        dense,
        lexical,
        limit=3,
    )

    assert results

    result_documents = list(results)

    assert result_documents[0].text == "alpha"


def test_vector_store_respects_search_limit():
    embedder = GeminiEmbedder()

    store = InMemoryVectorStore(embedder)

    documents = [
        Document("DC motor"),
        Document("transformer"),
        Document("induction motor"),
        Document("power system"),
    ]

    store.upsert(documents)

    query_vector = embedder.embed("motor")

    results = store.search(
        query_vector=query_vector,
        limit=2,
    )

    assert len(results) <= 2


def test_rrf_supports_weighted_lexical_evidence():
    dense = [
        (1.0, Document("dense-only")),
        (0.9, Document("shared")),
    ]

    lexical = [
        (1.0, dense[1][1]),
        (0.9, Document("lexical-only")),
    ]

    results = reciprocal_rank_fusion(
        dense,
        lexical,
        limit=3,
        dense_weight=1.0,
        lexical_weight=2.0,
    )

    assert results

    texts = [document.text for document in results]

    assert "shared" in texts
    assert "lexical-only" in texts


def test_rrf_rejects_invalid_weights():
    dense = [
        (1.0, Document("alpha")),
    ]

    lexical = []

    with pytest.raises(
        ValueError,
        match="lexical_weight",
    ):
        reciprocal_rank_fusion(
            dense,
            lexical,
            dense_weight=1.0,
            lexical_weight=0.0,
        )


def test_rrf_order_is_deterministic():
    dense = [
        (1.0, Document("alpha")),
        (1.0, Document("beta")),
    ]

    lexical = []

    first = reciprocal_rank_fusion(
        dense,
        lexical,
        limit=2,
    )

    second = reciprocal_rank_fusion(
        dense,
        lexical,
        limit=2,
    )

    assert [document.text for document in first] == [
        document.text for document in second
    ]
