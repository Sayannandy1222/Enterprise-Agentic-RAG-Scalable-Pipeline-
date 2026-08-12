from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.retrieval import evaluate_retrieval
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.service import RetrievalService
from app.retrieval.vector_store import InMemoryVectorStore
from app.embeddings.gemini import GeminiEmbedder
from app.services.ingestion import IngestionService


DATASET_PATH = Path(__file__).parent / "data" / "rag_eval.json"
PDF_PATH = Path("data/NIPS-2017-attention-is-all-you-need-Paper.pdf")

TOP_K = 5
EMBEDDING_CACHE_SIZE = 256
RERANK_CACHE_SIZE = 256


def load_dataset(
    path: Path = DATASET_PATH,
) -> list[dict[str, Any]]:
    """Load and validate the retrieval evaluation dataset."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    cases = payload.get("cases", [])

    if not isinstance(cases, list):
        raise ValueError("evaluation dataset 'cases' must be a list")

    validated: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")

        case_id = case.get("id")
        query = case.get("query")
        relevant = case.get("relevant_chunk_ids")

        if not case_id:
            raise ValueError(f"case {index} is missing 'id'")

        if not query:
            raise ValueError(f"{case_id} is missing 'query'")

        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"{case_id} must contain 'relevant_chunk_ids'")

        validated.append(
            {
                "id": str(case_id),
                "query": str(query),
                "relevant_chunk_ids": [str(value) for value in relevant],
            }
        )

    return validated


def extract_chunk_id(document: Any) -> str | None:
    """Extract the chunk identifier used by the evaluation dataset."""

    metadata = getattr(document, "metadata", {}) or {}

    if "chunk_id" in metadata:
        return str(metadata["chunk_id"])

    # Evaluation chunks are generated in ingestion order.
    # Retrieval documents may not carry an explicit chunk_id.
    return None


def build_retrieval_service() -> RetrievalService:
    """Build an offline retrieval pipeline for the PDF evaluation."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Evaluation PDF not found: {PDF_PATH}")

    ingestion = IngestionService()
    document = ingestion.load(str(PDF_PATH))
    chunks = ingestion.chunk(document)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"chunk_{index:03d}"

    embedder = GeminiEmbedder(
        cache_size=EMBEDDING_CACHE_SIZE,
    )

    vector_store = InMemoryVectorStore(embedder)
    vector_store.upsert(chunks)

    bm25 = BM25Retriever(chunks)

    return RetrievalService(
        vector_store,
        bm25,
        dense_limit=16,
        bm25_limit=16,
        rrf_limit=8,
        rerank_limit=8,
        top_k=TOP_K,
        embedding_cache_size=EMBEDDING_CACHE_SIZE,
        rerank_cache_size=RERANK_CACHE_SIZE,
    )


def main() -> None:
    """Run retrieval evaluation against the Attention paper."""

    cases = load_dataset()

    print("=" * 80)
    print("ATTENTION PAPER RAG EVALUATION")
    print("=" * 80)
    print(f"PDF:             {PDF_PATH}")
    print(f"Evaluation cases: {len(cases)}")
    print()

    retrieval = build_retrieval_service()

    retrieved: list[list[str]] = []
    relevant: list[list[str]] = []

    print("Running retrieval evaluation...")
    print()

    for case in cases:
        result = retrieval.search_with_metrics(
            query=case["query"],
            top_k=TOP_K,
        )

        results = result["results"]

        retrieved_ids: list[str] = []

        for document in results:
            chunk_id = extract_chunk_id(document)

            if chunk_id is not None:
                retrieved_ids.append(chunk_id)

        retrieved.append(retrieved_ids)
        relevant.append(case["relevant_chunk_ids"])

        case_metrics = evaluate_retrieval(
            [retrieved_ids],
            [case["relevant_chunk_ids"]],
            ks=(3, 5),
        )

        print(
            f"{case['id']}: "
            f"Recall@3={case_metrics['recall_at_3']:.4f}, "
            f"Recall@5={case_metrics['recall_at_5']:.4f}, "
            f"MRR={case_metrics['mrr']:.4f}"
        )

        print("  Retrieved: " + ", ".join(retrieved_ids))

    if len(retrieved) != len(relevant):
        raise RuntimeError(
            "Evaluation invariant violated: "
            f"{len(retrieved)} retrieved cases != "
            f"{len(relevant)} relevant cases"
        )

    metrics = evaluate_retrieval(
        retrieved,
        relevant,
        ks=(3, 5),
    )

    print()
    print("=" * 80)
    print("RETRIEVAL QUALITY")
    print("=" * 80)
    print(f"Cases:       {len(cases)}")
    print(f"Recall@3:    {metrics['recall_at_3']:.4f}")
    print(f"Recall@5:    {metrics['recall_at_5']:.4f}")
    print(f"MRR:         {metrics['mrr']:.4f}")

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
