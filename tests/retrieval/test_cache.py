from app.retrieval.cache import BoundedLRUCache


def test_lru_cache_hit():
    cache = BoundedLRUCache[int](2)

    cache.set("a", 1)

    value, hit = cache.get("a")

    assert value == 1
    assert hit is True

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_lru_cache_miss():
    cache = BoundedLRUCache[int](2)

    value, hit = cache.get("missing")

    assert value is None
    assert hit is False

    stats = cache.stats()
    assert stats["misses"] == 1


def test_lru_cache_eviction():
    cache = BoundedLRUCache[int](2)

    cache.set("a", 1)
    cache.set("b", 2)

    # Make a the most recently used entry.
    assert cache.get("a") == (1, True)

    cache.set("c", 3)

    # b should have been evicted.
    assert cache.get("b") == (None, False)

    # a and c should remain.
    assert cache.get("a") == (1, True)
    assert cache.get("c") == (3, True)


def test_lru_cache_clear():
    cache = BoundedLRUCache[int](2)

    cache.set("a", 1)
    cache.clear()

    value, hit = cache.get("a")

    assert value is None
    assert hit is False


def test_lru_cache_capacity_validation():
    try:
        BoundedLRUCache[int](0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero capacity")
