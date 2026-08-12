from __future__ import annotations

import hashlib
import uuid
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

            self._vectors[id(document)] = self.embedder.embed(document.text)

    add = upsert

    def count(self) -> int:
        if self.client is not None:
            try:
                result = self.client.count(
                    collection_name=self.collection,
                    exact=True,
                )
                return int(result.count)
            except Exception:
                pass

        return len(self.items)

    @staticmethod
    def _dot(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        return sum(a * b for a, b in zip(left, right))

    def load_documents(self, batch_size: int = 256) -> list[Document]:
        """
        Load persisted documents from Qdrant Cloud.

        Qdrant is the persistent source of truth. This method hydrates
        application-level document state so components such as BM25 can
        be reconstructed after process restart.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.client is None:
            return []

        documents: list[Document] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                documents.append(
                    Document(
                        text=str(payload.get("text", "")),
                        source=str(payload.get("source", "")),
                        metadata=payload.get("metadata", {}) or {},
                    )
                )

            if offset is None:
                break

        return documents

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
                raise ValueError("query or query_vector is required")

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
        api_key: str = "",
    ) -> None:
        super().__init__(embedder)

        self.url = url
        self.collection = collection
        self.api_key = api_key
        self.client = None

        try:
            from qdrant_client import QdrantClient

            if url:
                self.client = QdrantClient(
                    url=url,
                    api_key=api_key or None,
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

            first_vector = self._vectors[id(documents[0])]

            if not self.client.collection_exists(self.collection):
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

                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{document.source}:{document.text}",
                    )
                )

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

    def load_documents(self, batch_size: int = 256) -> list[Document]:
        """
        Load persisted documents from Qdrant Cloud.

        Qdrant is the persistent source of truth. This method hydrates
        application-level document state so components such as BM25 can
        be reconstructed after process restart.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.client is None:
            return []

        documents: list[Document] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}

                documents.append(
                    Document(
                        text=str(payload.get("text", "")),
                        source=str(payload.get("source", "")),
                        metadata=payload.get("metadata", {}) or {},
                    )
                )

            if offset is None:
                break

        return documents

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
                        raise ValueError("embedder is required")

                    if query is None:
                        raise ValueError("query or query_vector is required")

                    query_vector = self.embedder.embed(query)

                response = self.client.query_points(
                    collection_name=self.collection,
                    query=list(query_vector),
                    limit=limit,
                    with_payload=True,
                )

                return [
                    (
                        float(point.score),
                        Document(
                            text=(point.payload or {}).get(
                                "text",
                                "",
                            ),
                            source=(point.payload or {}).get(
                                "source",
                                "",
                            ),
                            metadata=(point.payload or {}).get(
                                "metadata",
                                {},
                            ),
                        ),
                    )
                    for point in response.points
                ]

            except Exception:
                self.client = None

        return super().search(
            query=query,
            limit=limit,
            query_vector=query_vector,
        )
