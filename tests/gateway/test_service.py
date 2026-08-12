from __future__ import annotations

import pytest

from app.gateway.service import (
    CircuitBreaker,
    GatewayConfig,
    LLMGateway,
    LLMRequest,
    LLMResponse,
)


class FakeProvider:
    def __init__(
        self,
        name: str = "fake",
        responses=None,
        failures=None,
    ):
        self.name = name
        self.model = "fake-model"
        self.responses = list(responses or [])
        self.failures = list(failures or [])
        self.calls = 0

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.calls += 1

        if self.failures:
            failure = self.failures.pop(0)

            if isinstance(failure, Exception):
                raise failure

            raise RuntimeError(str(failure))

        if self.responses:
            response = self.responses.pop(0)

            if isinstance(response, LLMResponse):
                return response

            return LLMResponse(
                text=str(response),
                provider=self.name,
                model=self.model,
            )

        return LLMResponse(
            text="success",
            provider=self.name,
            model=self.model,
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
        )


class HTTPError(Exception):
    def __init__(
        self,
        status_code: int,
    ):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def gateway(
    primary,
    fallback=None,
    max_retries=2,
    threshold=3,
):
    return LLMGateway(
        provider=primary,
        fallback_provider=fallback,
        config=GatewayConfig(
            max_retries=max_retries,
            retry_delay_seconds=0,
            circuit_failure_threshold=threshold,
            circuit_reset_seconds=0.01,
        ),
    )


def test_success_without_retry():
    provider = FakeProvider(
        name="primary",
        responses=["hello"],
    )

    response = gateway(
        provider,
        max_retries=2,
    ).generate(LLMRequest(prompt="hello"))

    assert response.text == "hello"
    assert response.provider == "primary"
    assert provider.calls == 1


def test_metrics_snapshot_is_serializable():
    provider = FakeProvider(
        responses=["hello"],
    )

    gw = gateway(provider)

    gw.generate("hello")

    snapshot = gw.metrics_snapshot()

    assert hasattr(
        snapshot,
        "to_dict",
    )

    data = snapshot.to_dict()

    assert isinstance(data, dict)
    assert data["requests"] == 1
    assert data["successes"] == 1


def test_gateway_snapshot_contains_gateway_and_circuit_data():
    provider = FakeProvider(
        name="primary",
        responses=["hello"],
    )

    gw = gateway(provider)

    gw.generate("hello")

    snapshot = gw.snapshot()

    assert "gateway" in snapshot
    assert "circuit_breakers" in snapshot
    assert "primary" in snapshot["circuit_breakers"]


def test_retries_transient_error_and_succeeds():
    provider = FakeProvider(
        name="primary",
        failures=[
            HTTPError(503),
        ],
        responses=["recovered"],
    )

    gw = gateway(
        provider,
        max_retries=2,
    )

    response = gw.generate(LLMRequest(prompt="hello"))

    assert response.text == "recovered"
    assert provider.calls == 2
    assert gw.metrics_snapshot().retries == 1


@pytest.mark.parametrize(
    "status_code",
    [
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    ],
)
def test_transient_status_codes_are_retryable(
    status_code,
):
    provider = FakeProvider(
        name="primary",
        failures=[
            HTTPError(status_code),
        ],
        responses=["success"],
    )

    gw = gateway(
        provider,
        max_retries=1,
    )

    response = gw.generate("hello")

    assert response.text == "success"
    assert provider.calls == 2
    assert gw.metrics_snapshot().retries == 1


def test_non_transient_error_is_not_retried():
    provider = FakeProvider(
        name="primary",
        failures=[
            HTTPError(400),
        ],
    )

    gw = gateway(
        provider,
        max_retries=3,
    )

    response = gw.generate("hello")

    assert response.metadata["failed"] is True
    assert provider.calls == 1
    assert gw.metrics_snapshot().retries == 0


def test_retry_history_is_recorded():
    provider = FakeProvider(
        name="primary",
        failures=[
            HTTPError(503),
        ],
        responses=["success"],
    )

    response = gateway(
        provider,
        max_retries=1,
    ).generate("hello")

    attempts = response.metadata["attempts"]

    assert len(attempts) == 1
    assert attempts[0]["provider"] == "primary"
    assert attempts[0]["retry_index"] == 0
    assert attempts[0]["retry"] is True


def test_gateway_latency_is_recorded():
    provider = FakeProvider(
        responses=["success"],
    )

    gw = gateway(provider)

    response = gw.generate("hello")

    snapshot = gw.metrics_snapshot()

    assert response.latency_ms >= 0
    assert snapshot.total_latency_ms >= 0


def test_token_metrics_are_recorded():
    provider = FakeProvider(
        responses=[
            LLMResponse(
                text="hello world",
                provider="primary",
                model="fake",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            )
        ]
    )

    gw = gateway(provider)

    gw.generate("hello")

    snapshot = gw.metrics_snapshot()

    assert snapshot.input_tokens == 10
    assert snapshot.output_tokens == 5
    assert snapshot.total_tokens == 15


def test_fallback_provider_succeeds_after_primary_exhausts_retries():
    primary = FakeProvider(
        name="primary",
        failures=[
            HTTPError(503),
            HTTPError(503),
        ],
    )

    fallback = FakeProvider(
        name="fallback",
        responses=["fallback-success"],
    )

    gw = gateway(
        primary,
        fallback,
        max_retries=1,
    )

    response = gw.generate("hello")

    assert response.text == "fallback-success"
    assert response.provider == "fallback"
    assert primary.calls == 2
    assert fallback.calls == 1

    snapshot = gw.metrics_snapshot()

    assert snapshot.fallbacks == 1
    assert snapshot.retries == 1


def test_fallback_provider_can_retry_transient_failure():
    primary = FakeProvider(
        name="primary",
        failures=[
            HTTPError(500),
            HTTPError(500),
        ],
    )

    fallback = FakeProvider(
        name="fallback",
        failures=[
            HTTPError(503),
        ],
        responses=["fallback-recovered"],
    )

    gw = gateway(
        primary,
        fallback,
        max_retries=1,
    )

    response = gw.generate("hello")

    assert response.text == "fallback-recovered"
    assert primary.calls == 2
    assert fallback.calls == 2

    snapshot = gw.metrics_snapshot()

    assert snapshot.fallbacks == 1
    assert snapshot.retries == 2


def test_retry_history_contains_primary_and_fallback_attempts():
    primary = FakeProvider(
        name="primary",
        failures=[
            HTTPError(503),
        ],
    )

    fallback = FakeProvider(
        name="fallback",
        responses=["fallback"],
    )

    response = gateway(
        primary,
        fallback,
        max_retries=0,
    ).generate("hello")

    history = response.metadata["attempts"]

    assert len(history) == 1
    assert history[0]["provider"] == "primary"

    assert response.metadata["fallback"] is True


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(
        threshold=2,
        reset_seconds=100,
    )

    assert breaker.allow_request()

    breaker.record_failure()

    assert breaker.allow_request()

    breaker.record_failure()

    assert breaker.open is True
    assert breaker.allow_request() is False


def test_circuit_breaker_resets_after_success():
    breaker = CircuitBreaker(
        threshold=2,
        reset_seconds=100,
    )

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.open

    breaker.state = breaker.HALF_OPEN

    breaker.record_success()

    assert breaker.open is False
    assert breaker.failures == 0
    assert breaker.allow_request() is True


def test_empty_prompt_is_rejected():
    provider = FakeProvider()

    gw = gateway(provider)

    with pytest.raises(ValueError):
        gw.generate("")


def test_invalid_request_type_is_rejected():
    provider = FakeProvider()

    gw = gateway(provider)

    with pytest.raises(TypeError):
        gw.generate(123)


def test_string_request_is_supported():
    provider = FakeProvider(
        responses=["hello"],
    )

    response = gateway(provider).generate("hello")

    assert response.text == "hello"
    assert provider.calls == 1
