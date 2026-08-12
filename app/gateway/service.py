"""Production-grade provider-neutral LLM gateway.

Responsibilities
----------------
* Provider abstraction
* Retry handling with exponential backoff
* Transient HTTP/status-code classification
* Primary -> fallback failover
* Per-provider circuit breakers
* Retry/failover history
* Latency and token accounting
* Serializable gateway metrics
* Deterministic local provider for offline development/tests
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import monotonic, perf_counter, sleep
from typing import Any, Protocol


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    provider: str = "local"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class GatewayConfig:
    max_retries: int = 2
    retry_delay_seconds: float = 0.05
    retry_backoff_factor: float = 2.0

    fallback_provider_name: str | None = None

    circuit_failure_threshold: int = 3
    circuit_reset_seconds: float = 30.0

    transient_status_codes: tuple[int, ...] = (
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    )


# ============================================================================
# PROVIDER CONTRACT
# ============================================================================


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, request: LLMRequest) -> LLMResponse:
        ...


# ============================================================================
# LOCAL PROVIDER
# ============================================================================


class LocalProvider:
    """Deterministic provider used for offline execution and tests."""

    name = "local"
    model = "extractive-demo"

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not request.prompt.strip():
            raise ValueError("prompt must not be empty")

        prompt = request.prompt.strip()

        if "Question:" in prompt:
            context = prompt.split("Question:", 1)[0]
        else:
            context = prompt

        context = context.replace("Context:", "").strip()

        if context:
            answer = (
                "Based on the retrieved context: "
                f"{context[:600]}"
            )
        else:
            answer = (
                "No relevant context was retrieved."
            )

        input_tokens = _estimate_tokens(request.prompt)
        output_tokens = _estimate_tokens(answer)

        return LLMResponse(
            text=answer,
            provider=self.name,
            model=request.model or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )


# ============================================================================
# METRICS
# ============================================================================


@dataclass
class GatewayMetricsSnapshot:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    fallbacks: int = 0

    total_latency_ms: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    transient_failures: int = 0
    circuit_open_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot."""

        return asdict(self)

    @property
    def average_latency_ms(self) -> float:
        if self.successes + self.failures == 0:
            return 0.0

        return self.total_latency_ms / (
            self.successes + self.failures
        )


class GatewayMetrics:
    """Mutable metrics collector."""

    def __init__(self) -> None:
        self._snapshot = GatewayMetricsSnapshot()

    def record_request(self) -> None:
        self._snapshot.requests += 1

    def record_success(
        self,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self._snapshot.successes += 1
        self._snapshot.total_latency_ms += latency_ms
        self._snapshot.input_tokens += input_tokens
        self._snapshot.output_tokens += output_tokens
        self._snapshot.total_tokens += (
            input_tokens + output_tokens
        )

    def record_failure(
        self,
        latency_ms: float = 0.0,
    ) -> None:
        self._snapshot.failures += 1
        self._snapshot.total_latency_ms += latency_ms

    def record_retry(self) -> None:
        self._snapshot.retries += 1

    def record_fallback(self) -> None:
        self._snapshot.fallbacks += 1

    def record_transient_failure(self) -> None:
        self._snapshot.transient_failures += 1

    def record_circuit_open(self) -> None:
        self._snapshot.circuit_open_events += 1

    def snapshot(self) -> GatewayMetricsSnapshot:
        return GatewayMetricsSnapshot(
            **self._snapshot.to_dict()
        )

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot().to_dict()


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


class CircuitBreaker:
    """Small stateful circuit breaker.

    CLOSED
        Requests are allowed.

    OPEN
        Requests are rejected until reset timeout expires.

    HALF-OPEN
        One request is allowed after timeout. Success closes the circuit;
        failure opens it again.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        threshold: int = 3,
        reset_seconds: float = 30.0,
    ) -> None:
        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than zero"
            )

        if reset_seconds < 0:
            raise ValueError(
                "reset_seconds must not be negative"
            )

        self.threshold = threshold
        self.reset_seconds = reset_seconds

        self.failures = 0
        self.state = self.CLOSED
        self.opened_at: float | None = None

    @property
    def open(self) -> bool:
        return self.state == self.OPEN

    def allow_request(self) -> bool:
        if self.state == self.CLOSED:
            return True

        if self.state == self.OPEN:
            if self.opened_at is None:
                return False

            elapsed = monotonic() - self.opened_at

            if elapsed >= self.reset_seconds:
                self.state = self.HALF_OPEN
                return True

            return False

        # HALF_OPEN
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = self.CLOSED
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1

        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
            self.opened_at = monotonic()
            return

        if self.failures >= self.threshold:
            self.state = self.OPEN
            self.opened_at = monotonic()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "open": self.open,
            "failures": self.failures,
            "threshold": self.threshold,
            "reset_seconds": self.reset_seconds,
        }


# ============================================================================
# HELPERS
# ============================================================================


def _estimate_tokens(text: str) -> int:
    return max(
        0,
        len(text.strip().split()),
    )


def _extract_status_code(exc: Exception) -> int | None:
    """Extract an HTTP-style status code from common exceptions."""

    for attribute in (
        "status_code",
        "status",
        "code",
    ):
        value = getattr(exc, attribute, None)

        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)

    if response is not None:
        value = getattr(
            response,
            "status_code",
            None,
        )

        if isinstance(value, int):
            return value

    return None


# ============================================================================
# GATEWAY
# ============================================================================


class LLMGateway:
    """Production-oriented LLM gateway."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        fallback_provider: LLMProvider | None = None,
        registry: Any | None = None,
        config: GatewayConfig | None = None,
        metrics: GatewayMetrics | None = None,
        **_: Any,
    ) -> None:
        self.provider = provider or LocalProvider()
        self.fallback_provider = fallback_provider

        self.registry = registry

        self.config = config or GatewayConfig()

        self.metrics = (
            metrics
            if metrics is not None
            else GatewayMetrics()
        )

        self._breakers: dict[str, CircuitBreaker] = {}

        self.requests = 0

    # ------------------------------------------------------------------
    # Provider / breaker helpers
    # ------------------------------------------------------------------

    def _get_circuit_breaker(
        self,
        provider: LLMProvider,
    ) -> CircuitBreaker:
        name = getattr(
            provider,
            "name",
            provider.__class__.__name__,
        )

        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                threshold=(
                    self.config.circuit_failure_threshold
                ),
                reset_seconds=(
                    self.config.circuit_reset_seconds
                ),
            )

        return self._breakers[name]

    def _providers(self) -> list[LLMProvider]:
        providers: list[LLMProvider] = []

        if self.provider is not None:
            providers.append(self.provider)

        if (
            self.fallback_provider is not None
            and self.fallback_provider is not self.provider
        ):
            providers.append(
                self.fallback_provider
            )

        return providers

    def _circuit_snapshots(self) -> dict[str, Any]:
        return {
            name: breaker.snapshot()
            for name, breaker in self._breakers.items()
        }

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------

    def _is_transient(
        self,
        exc: Exception,
    ) -> bool:
        status_code = _extract_status_code(exc)

        if status_code is not None:
            return (
                status_code
                in self.config.transient_status_codes
            )

        # Network-like exceptions are generally transient.
        transient_names = {
            "TimeoutError",
            "ConnectionError",
            "TemporaryError",
        }

        return exc.__class__.__name__ in transient_names

    def _retry_delay(
        self,
        retry_index: int,
    ) -> float:
        return (
            self.config.retry_delay_seconds
            * (
                self.config.retry_backoff_factor
                ** retry_index
            )
        )

    # ------------------------------------------------------------------
    # Main generation path
    # ------------------------------------------------------------------

    def generate(
        self,
        request: LLMRequest | str,
        **kwargs: Any,
    ) -> LLMResponse:
        if isinstance(request, str):
            request = LLMRequest(
                prompt=request,
                metadata=kwargs,
            )

        if not isinstance(request, LLMRequest):
            raise TypeError(
                "request must be LLMRequest or str"
            )

        if not request.prompt.strip():
            raise ValueError(
                "prompt must not be empty"
            )

        started = perf_counter()

        self.requests += 1
        self.metrics.record_request()

        retry_history: list[dict[str, Any]] = []

        providers = self._providers()

        if not providers:
            raise RuntimeError(
                "no LLM providers configured"
            )

        for provider_index, provider in enumerate(
            providers
        ):
            provider_name = getattr(
                provider,
                "name",
                provider.__class__.__name__,
            )

            breaker = self._get_circuit_breaker(
                provider
            )

            is_fallback = provider_index > 0

            if is_fallback:
                self.metrics.record_fallback()

            if not breaker.allow_request():
                self.metrics.record_circuit_open()

                retry_history.append(
                    {
                        "provider": provider_name,
                        "attempt": 0,
                        "retry_index": 0,
                        "fallback": is_fallback,
                        "status": "circuit_open",
                        "circuit": breaker.snapshot(),
                    }
                )

                continue

            for retry_index in range(
                self.config.max_retries + 1
            ):
                attempt_number = (
                    len(retry_history) + 1
                )

                try:
                    response = provider.generate(
                        request
                    )

                    if response is None:
                        raise RuntimeError(
                            "provider returned None"
                        )

                    breaker.record_success()

                    latency_ms = (
                        perf_counter() - started
                    ) * 1000.0

                    response.latency_ms = round(
                        latency_ms,
                        3,
                    )

                    response.metadata = {
                        **(
                            response.metadata
                            or {}
                        ),
                        "attempts": retry_history,
                        "fallback": is_fallback,
                        "provider_index": provider_index,
                    }

                    self.metrics.record_success(
                        latency_ms=latency_ms,
                        input_tokens=(
                            response.input_tokens
                        ),
                        output_tokens=(
                            response.output_tokens
                        ),
                    )

                    return response

                except Exception as exc:
                    transient = (
                        self._is_transient(exc)
                    )

                    if transient:
                        self.metrics.record_transient_failure()

                    should_retry = (
                        transient
                        and retry_index
                        < self.config.max_retries
                    )

                    retry_history.append(
                        {
                            "provider": provider_name,
                            "attempt": attempt_number,
                            "retry_index": retry_index,
                            "fallback": is_fallback,
                            "status": (
                                _extract_status_code(exc)
                                or "error"
                            ),
                            "transient": transient,
                            "retry": should_retry,
                            "error": str(exc),
                        }
                    )

                    # A transient failure should not immediately
                    # consume the circuit threshold multiple times
                    # inside the same retry sequence.
                    if not should_retry:
                        breaker.record_failure()

                    if should_retry:
                        self.metrics.record_retry()

                        delay = self._retry_delay(
                            retry_index
                        )

                        if delay > 0:
                            sleep(delay)

                        continue

                    # Non-transient errors also move us toward
                    # circuit opening.
                    if not transient:
                        breaker.record_failure()

                    break

        latency_ms = (
            perf_counter() - started
        ) * 1000.0

        self.metrics.record_failure(
            latency_ms=latency_ms
        )

        return LLMResponse(
            text=(
                "No LLM provider succeeded; "
                "retrieved context is available "
                "for review."
            ),
            provider="gateway",
            model="",
            latency_ms=round(
                latency_ms,
                3,
            ),
            metadata={
                "attempts": retry_history,
                "fallback": False,
                "failed": True,
                "circuit_breakers": (
                    self._circuit_snapshots()
                ),
            },
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "gateway": self.metrics.to_dict(),
            "circuit_breakers": (
                self._circuit_snapshots()
            ),
        }

    def metrics_snapshot(
        self,
    ) -> GatewayMetricsSnapshot:
        return self.metrics.snapshot()
