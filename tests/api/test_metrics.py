from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.metrics import router


class FakeRetrieval:
    def __init__(self, metrics_payload=None):
        if metrics_payload is None:
            metrics_payload = {
                "requests": 10,
                "cache_hits": 4,
                "cache_misses": 6,
                "p95_latency_ms": 42.5,
            }

        self.payload = metrics_payload
        self.calls = 0

    def metrics(self):
        self.calls += 1
        return self.payload


def build_app(retrieval=None):
    app = FastAPI()
    app.include_router(router)

    if retrieval is not None:
        app.state.retrieval = retrieval

    return app


def test_metrics_endpoint_exists():
    client = TestClient(build_app(FakeRetrieval()))

    response = client.get("/metrics")

    assert response.status_code == 200


def test_metrics_endpoint_returns_retrieval_key():
    client = TestClient(build_app(FakeRetrieval()))

    response = client.get("/metrics")

    body = response.json()

    assert "retrieval" in body


def test_metrics_endpoint_returns_retrieval_metrics():
    retrieval = FakeRetrieval(
        {
            "requests": 25,
            "cache_hits": 12,
            "cache_misses": 13,
        }
    )

    client = TestClient(build_app(retrieval))

    response = client.get("/metrics")

    assert response.json()["retrieval"] == {
        "requests": 25,
        "cache_hits": 12,
        "cache_misses": 13,
    }


def test_metrics_calls_retrieval_metrics_once():
    retrieval = FakeRetrieval()

    client = TestClient(build_app(retrieval))

    response = client.get("/metrics")

    assert response.status_code == 200
    assert retrieval.calls == 1


def test_metrics_without_retrieval_returns_empty_metrics():
    client = TestClient(build_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == {"retrieval": {}}


def test_metrics_preserves_numeric_values():
    retrieval = FakeRetrieval(
        {
            "requests": 100,
            "cache_hits": 55,
            "cache_misses": 45,
            "p95_latency_ms": 12.75,
        }
    )

    client = TestClient(build_app(retrieval))

    body = client.get("/metrics").json()

    assert body["retrieval"]["requests"] == 100
    assert body["retrieval"]["cache_hits"] == 55
    assert body["retrieval"]["cache_misses"] == 45
    assert body["retrieval"]["p95_latency_ms"] == pytest.approx(12.75)


def test_metrics_preserves_nested_metrics():
    retrieval = FakeRetrieval(
        {
            "latency": {
                "p50_ms": 5.2,
                "p95_ms": 11.8,
            },
            "cache": {
                "hits": 20,
                "misses": 5,
            },
        }
    )

    client = TestClient(build_app(retrieval))

    body = client.get("/metrics").json()

    assert body["retrieval"]["latency"]["p50_ms"] == pytest.approx(5.2)
    assert body["retrieval"]["latency"]["p95_ms"] == pytest.approx(11.8)
    assert body["retrieval"]["cache"]["hits"] == 20
    assert body["retrieval"]["cache"]["misses"] == 5


def test_metrics_supports_empty_retrieval_metrics():
    retrieval = FakeRetrieval({})

    client = TestClient(build_app(retrieval))

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json() == {"retrieval": {}}
    assert retrieval.calls == 1


def test_metrics_is_read_only():
    payload = {
        "requests": 7,
        "cache_hits": 3,
        "cache_misses": 4,
    }

    retrieval = FakeRetrieval(payload)

    client = TestClient(build_app(retrieval))

    first = client.get("/metrics").json()
    second = client.get("/metrics").json()

    assert first == second
    assert retrieval.payload == payload
    assert retrieval.calls == 2
