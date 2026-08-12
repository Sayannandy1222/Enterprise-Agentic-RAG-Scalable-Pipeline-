from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class BM25Retriever:
    """
    Production-oriented BM25 lexical retriever.

    Implements Okapi BM25 with:
        k1 = 1.5
        b  = 0.75

    The retriever maintains:
        - tokenized documents
        - term frequencies
        - document frequencies
        - inverse document frequency
        - document lengths
        - average document length

    Search returns:
        [(score, document), ...]
    """

    def __init__(
        self,
        documents: list[Any] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.documents = list(documents or [])

        self.k1 = k1
        self.b = b

        self.tokens: list[list[str]] = []
        self.term_frequencies: list[Counter[str]] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.document_lengths: list[int] = []
        self.avgdl = 0.0

        self._rebuild()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _document_text(document: Any) -> str:
        if hasattr(document, "text"):
            return str(document.text)

        return str(document)

    def _rebuild(self) -> None:
        self.tokens = [
            self._tokenize(self._document_text(document)) for document in self.documents
        ]

        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]

        self.document_lengths = [len(tokens) for tokens in self.tokens]

        self.df = {}

        for tokens in self.tokens:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        document_count = len(self.documents)

        self.avgdl = (
            sum(self.document_lengths) / document_count if document_count else 0.0
        )

        self.idf = {}

        for term, document_frequency in self.df.items():
            # Standard Robertson/Sparck Jones-style BM25 IDF.
            self.idf[term] = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

    def add(self, documents: list[Any]) -> None:
        """
        Add documents and rebuild the BM25 statistics.
        """

        self.documents.extend(documents)
        self._rebuild()

    def search(
        self,
        query: str,
        limit: int = 16,
    ) -> list[tuple[float, Any]]:
        """
        Search using Okapi BM25.

        Returns results ordered from highest BM25 score
        to lowest score.
        """

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        if not self.documents:
            return []

        query_terms = set(self._tokenize(query))

        if not query_terms:
            return []

        document_count = len(self.documents)

        if self.avgdl <= 0:
            return []

        results: list[tuple[float, Any]] = []

        for index in range(document_count):
            term_frequencies = self.term_frequencies[index]
            document_length = self.document_lengths[index]

            score = 0.0

            length_normalization = 1.0 - self.b + self.b * document_length / self.avgdl

            for term in query_terms:
                frequency = term_frequencies.get(term, 0)

                if frequency == 0:
                    continue

                idf = self.idf.get(term, 0.0)

                numerator = frequency * (self.k1 + 1.0)

                denominator = frequency + self.k1 * length_normalization

                score += idf * (numerator / denominator)

            if score > 0.0:
                results.append(
                    (
                        score,
                        self.documents[index],
                    )
                )

        results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return results[:limit]
