from __future__ import annotations

from typing import Any


class FlashRankReranker:
    """
    Low-overhead FlashRank reranker.

    The model is initialized once per process and reused across
    requests. Candidate pruning happens upstream so FlashRank only
    processes a small high-quality candidate set.
    """

    _ranker: Any = None
    _initialization_failed = False

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
    ) -> None:
        self.model_name = model_name

        if (
            FlashRankReranker._ranker is None
            and not FlashRankReranker._initialization_failed
        ):
            self._initialize()

    def _initialize(self) -> None:
        try:
            from flashrank import Ranker

            FlashRankReranker._ranker = Ranker(model_name=self.model_name)

        except Exception:
            # Keep deterministic local fallback available for
            # development/test environments where the model is
            # unavailable.
            FlashRankReranker._initialization_failed = True
            FlashRankReranker._ranker = None

    @staticmethod
    def _text(document: Any) -> str:
        if hasattr(document, "text"):
            return str(document.text)

        return str(document)

    def _fallback(
        self,
        query: str,
        documents: list[Any],
        limit: int,
    ) -> list[Any]:
        query_terms = set(query.lower().split())

        scored = []

        for index, document in enumerate(documents):
            text = self._text(document).lower()

            score = sum(term in text for term in query_terms)

            scored.append(
                (
                    score,
                    index,
                    document,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                -item[1],
            ),
            reverse=True,
        )

        return [document for _, _, document in scored[:limit]]

    def rerank(
        self,
        query: str,
        documents: list[Any],
        limit: int = 8,
    ) -> list[Any]:
        if not documents:
            return []

        limit = min(
            max(1, limit),
            len(documents),
        )

        ranker = FlashRankReranker._ranker

        if ranker is None:
            return self._fallback(
                query,
                documents,
                limit,
            )

        try:
            from flashrank import (
                RerankRequest,
            )

            passages = [
                {
                    "id": index,
                    "text": self._text(document),
                }
                for index, document in enumerate(documents)
            ]

            request = RerankRequest(
                query=query,
                passages=passages,
            )

            ranked = ranker.rerank(request)

            output: list[Any] = []

            for item in ranked[:limit]:
                index = item.get("id")

                if isinstance(index, int) and 0 <= index < len(documents):
                    output.append(documents[index])

            if output:
                return output

        except Exception:
            # Fallback keeps the retrieval service functional
            # when the local reranker model is unavailable.
            pass

        return self._fallback(
            query,
            documents,
            limit,
        )
