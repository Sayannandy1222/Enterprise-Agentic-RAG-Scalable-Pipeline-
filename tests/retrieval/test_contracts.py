from __future__ import annotations

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
        Document(
            "DC motor converts electrical energy into mechanical energy."
        ),
        Document(
            "Transformer transfers electrical energy between circuits."
        ),
        Document(
            "Induction motor uses electromagnetic induction."
        ),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.search(
        "DC motor",
        limit=3,
    )

    assert results

    assert any(
        "DC motor" in document.text
        for _, document in results
    )


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

    store = InMemoryVectorStore(
        embedder
    )

    documents = [
        Document("DC motor"),
        Document("transformer"),
        Document("induction motor"),
        Document("power system"),
    ]

    store.upsert(documents)

    query_vector = embedder.embed(
        "motor"
    )

    results = store.search(
        query_vector=query_vector,
        limit=2,
    )

    assert len(results) <= 2
