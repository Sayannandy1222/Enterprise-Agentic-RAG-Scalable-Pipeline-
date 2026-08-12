from __future__ import annotations

from app.ingestion.models import Document
from app.reranking.flashrank import FlashRankReranker


def test_reranker_respects_limit():
    reranker = FlashRankReranker()

    documents = [
        Document("DC motor speed depends on voltage."),
        Document("DC motor produces torque."),
        Document("Transformer changes voltage."),
        Document("Power systems transmit electricity."),
    ]

    results = reranker.rerank(
        "DC motor speed",
        documents,
        limit=2,
    )

    assert len(results) <= 2


def test_reranker_returns_documents():
    reranker = FlashRankReranker()

    documents = [
        Document("DC motor speed depends on voltage."),
        Document("Transformer transfers energy."),
    ]

    results = reranker.rerank(
        "DC motor speed",
        documents,
        limit=2,
    )

    assert results
    assert all(isinstance(document, Document) for document in results)


def test_reranker_handles_empty_candidates():
    reranker = FlashRankReranker()

    results = reranker.rerank(
        "DC motor speed",
        [],
        limit=5,
    )

    assert results == []
