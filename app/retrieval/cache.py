from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Generic, TypeVar


T = TypeVar("T")


class BoundedLRUCache(Generic[T]):
    """
    Thread-safe bounded LRU cache.

    The cache stores at most `capacity` entries and evicts the
    least-recently-used entry when the capacity is exceeded.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")

        self.capacity = capacity
        self._data: OrderedDict[str, T] = OrderedDict()
        self._lock = RLock()

        self.hits = 0
        self.misses = 0

    def get(
        self,
        key: str,
    ) -> tuple[T | None, bool]:
        """
        Return `(value, hit)`.

        A successful lookup promotes the entry to the MRU position.
        """

        with self._lock:
            if key not in self._data:
                self.misses += 1
                return None, False

            value = self._data.pop(key)
            self._data[key] = value

            self.hits += 1

            return value, True

    def set(
        self,
        key: str,
        value: T,
    ) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)

            self._data[key] = value

            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total = self.hits + self.misses

            return {
                "size": len(self._data),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total if total else 0.0),
            }
