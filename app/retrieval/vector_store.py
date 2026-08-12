from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

from app.ingestion.models import Document


class InMemoryVectorStore:
    """
    Local vector store used for development and deterministic tests.

    The search API accepts either raw query text or a precomputed query
    vector. Production retrieval should use query_vector so embedding
    latency is measured separately from vector-search latency.
    """

    def __init__(self, embedder=None) -> None:
        self.embedder = embedder
        self.items: list[Document] = []
        self._vectors: dict[int, list[float]] = {}

    def upsert(self, documents: Iterable[Document]) -> None:
        for document in documents:
            if document in self.items:
                continue

            self.items.append(document)

            if self.embedder is None:
                raise ValueError("embedder is required")

            self._vectors[id(document)] = self.embedder.embed(
                document.text
            )

    add = upsert

    def count(self) -> int:
        return len(self.items)

    @staticmethod
    def _dot(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        return sum(a * b for a, b in zip(left, right))

    def search(
        self,
        query: str | None = None,
        limit: int = 16,
        query_vector: Sequence[float] | None = None,
    ) -> list[tuple[float, Document]]:
        if query_vector is None:
            if self.embedder is None:
                raise ValueError("embedder is required")

            if query is None:
                raise ValueError(
                    "query or query_vector is required"
                )

            query_vector = self.embedder.embed(query)

        scored: list[tuple[float, Document]] = []

        for document in self.items:
            vector = self._vectors.get(id(document))

            if vector is None:
                continue

            score = self._dot(
                query_vector,
                vector,
            )

            scored.append((score, document))

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return scored[:limit]


class QdrantVectorStore(InMemoryVectorStore):
    """
    Qdrant-backed vector store with an in-memory fallback.

    Query embeddings are computed by RetrievalService and passed through
    query_vector, keeping embedding and vector-search latency separate.
    """

    def __init__(
        self,
        embedder,
        url: str = "",
        collection: str = "documents",
    ) -> None:
        super().__init__(embedder)

        self.url = url
        self.collection = collection
        self.client = None

        try:
            from qdrant_client import QdrantClient

            if url:
                self.client = QdrantClient(
                    url=url,
                    timeout=5,
                )
        except Exception:
            self.client = None

    def upsert(
        self,
        documents: Iterable[Document],
    ) -> None:
        documents = list(documents)

        if not documents:
            return

        super().upsert(documents)

        if not self.client:
            return

        try:
            from qdrant_client.models import (
                Distance,
                PointStruct,
                VectorParams,
            )

            first_vector = self._vectors[
                id(documents[0])
            ]

            if not self.client.collection_exists(
                self.collection
            ):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=len(first_vector),
                        distance=Distance.COSINE,
                    ),
                )

            points: list[PointStruct] = []

            for document in documents:
                vector = self._vectors[id(document)]

                point_id = hashlib.sha1(
                    (
                        f"{document.source}:"
                        f"{document.text}"
                    ).encode("utf-8")
                ).hexdigest()[:16]

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": document.text,
                            "source": document.source,
                            "metadata": document.metadata,
                        },
                    )
                )

            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )

        except Exception:
            self.client = None

    def search(
        self,
        query: str | None = None,
        limit: int = 16,
        query_vector: Sequence[float] | None = None,
    ) -> list[tuple[float, Document]]:
        if self.client:
            try:
                if query_vector is None:
                    if self.embedder is None:
                        raise ValueError(
                            "embedder is required"
                        )

                    if query is None:
                        raise ValueError(
                            "query or query_vector is required"
                        )

                    query_vector = self.embedder.embed(
                        query
                    )

                hits = self.client.search(
                    collection_name=self.collection,
                    query_vector=list(query_vector),
                    limit=limit,
                )

                return [
                    (
                        float(hit.score),
                        Document(
                            text=hit.payload.get(
                                "text",
                                "",
                            ),
                            source=hit.payload.get(
                                "source",
                                "",
                            ),
                            metadata=hit.payload.get(
                                "metadata",
                                {},
                            ),
                        ),
                    )
                    for hit in hits
                ]

            except Exception:
                self.client = None

        return super().search(
            query=query,
            limit=limit,
            query_vector=query_vector,
        )
