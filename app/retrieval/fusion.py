from __future__ import annotations

from collections import defaultdict
from typing import Any


def _document_key(document: Any) -> str:
    """
    Return a stable logical identity for a document.

    Qdrant and BM25 can materialize separate Python Document objects
    representing the same chunk. Python object identity must therefore
    not be used for fusion.
    """

    metadata = getattr(document, "metadata", {}) or {}

    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None:
        return f"chunk:{chunk_id}"

    document_id = metadata.get("id")
    if document_id is not None:
        return f"id:{document_id}"

    source = getattr(document, "source", "") or metadata.get("source", "")
    start = metadata.get("start")
    end = metadata.get("end")

    if start is not None or end is not None:
        return f"span:{source}:{start}:{end}"

    text = getattr(document, "text", None)

    if text is not None:
        return f"text:{source}:{text}"

    return f"object:{id(document)}"


def reciprocal_rank_fusion(
    dense,
    lexical,
    limit: int = 8,
    k: int = 60,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
):
    """
    Fuse dense and lexical rankings using weighted Reciprocal Rank Fusion.

    The stable document identity ensures that independently materialized
    dense and BM25 Document objects representing the same chunk are
    correctly merged.

    Args:
        dense:
            Dense retrieval results as (score, document).

        lexical:
            BM25 retrieval results as (score, document).

        limit:
            Maximum number of fused candidates.

        k:
            RRF rank constant.

        dense_weight:
            Weight applied to dense retrieval evidence.

        lexical_weight:
            Weight applied to lexical retrieval evidence.
    """

    if limit <= 0:
        return []

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if dense_weight <= 0:
        raise ValueError("dense_weight must be greater than zero")

    if lexical_weight <= 0:
        raise ValueError("lexical_weight must be greater than zero")

    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, Any] = {}

    for rank, (_, document) in enumerate(dense, start=1):
        key = _document_key(document)

        scores[key] += dense_weight / (k + rank)
        documents.setdefault(key, document)

    for rank, (_, document) in enumerate(lexical, start=1):
        key = _document_key(document)

        scores[key] += lexical_weight / (k + rank)
        documents.setdefault(key, document)

    ranked_keys = sorted(
        scores,
        key=lambda key: (-scores[key], key),
    )

    return [documents[key] for key in ranked_keys[:limit]]
