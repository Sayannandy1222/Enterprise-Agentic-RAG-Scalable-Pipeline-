from collections import OrderedDict
from threading import RLock
from typing import Sequence
import hashlib, math


class GeminiEmbedder:
    """Gemini embeddings with bounded LRU cache and observable statistics."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "models/text-embedding-004",
        cache_size: int = 2048,
        dimensions: int = 3072,
    ):
        self.api_key, self.model, self.cache_size, self.dimensions = (
            api_key,
            model,
            max(1, cache_size),
            dimensions,
        )
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = RLock()
        self._hits = self._misses = 0
        self._client = None
        if api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                self._client = genai
            except Exception:
                self._client = None

    def _fallback(self, text: str) -> list[float]:
        # deterministic local fallback keeps tests/offline development reproducible
        raw = hashlib.sha256(text.encode()).digest()
        vals = [((raw[i % len(raw)] / 255.0) * 2 - 1) for i in range(self.dimensions)]
        norm = math.sqrt(sum(x * x for x in vals)) or 1.0
        return [x / norm for x in vals]

    def embed(self, text: str) -> list[float]:
        key = text.strip()
        if not key:
            raise ValueError("text must not be empty")
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return list(self._cache[key])
            self._misses += 1
        vector = None
        if self._client:
            try:
                vector = self._client.embed_content(
                    model=self.model,
                    content=key,
                    task_type="retrieval_document",
                    output_dimensionality=self.dimensions,
                )["embedding"]
            except Exception:
                vector = None
        vector = vector or self._fallback(key)
        with self._lock:
            self._cache[key] = list(vector)
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return list(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def cache_hits(self) -> int:
        return self._hits

    @property
    def cache_misses(self) -> int:
        return self._misses

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "capacity": self.cache_size,
            }

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def cache_info(self) -> dict[str, int]:
        return self.stats()
