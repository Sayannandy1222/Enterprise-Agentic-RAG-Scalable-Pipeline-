"""Provider-neutral LLM gateway used by the RAG orchestrator.

The gateway keeps provider details behind one interface and provides retries,
fallback routing, circuit breaking, latency/token metrics, and a deterministic
local provider for demos when Portkey/Groq credentials are unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol


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


@dataclass
class GatewayConfig:
    max_retries: int = 2
    retry_delay_seconds: float = 0.05
    fallback_provider_name: str | None = None
    circuit_failure_threshold: int = 3
    circuit_reset_seconds: float = 30.0


class LLMProvider(Protocol):
    name: str
    model: str
    def generate(self, request: LLMRequest) -> LLMResponse: ...


class LocalProvider:
    name = "local"
    model = "extractive-demo"
    def generate(self, request: LLMRequest) -> LLMResponse:
        # Deterministic answer for offline demonstrations; production providers
        # can be injected without changing the RAG service.
        text = request.prompt.strip()
        context = text.split("Question:", 1)[0].replace("Context:", "").strip()
        answer = (f"Based on the retrieved context: {context[:600]}" if context else "No relevant context was retrieved.")
        output = _estimate_tokens(answer)
        return LLMResponse(answer, self.name, self.model, _estimate_tokens(request.prompt), output, _estimate_tokens(request.prompt) + output)


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
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()


class CircuitBreaker:
    def __init__(self, threshold: int = 3): self.threshold, self.failures, self.open = threshold, 0, False
    def allow_request(self) -> bool: return not self.open
    def record_success(self): self.failures = 0; self.open = False
    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold: self.open = True
    def snapshot(self): return {"open": self.open, "failures": self.failures, "threshold": self.threshold}


def _estimate_tokens(text: str) -> int: return max(0, len(text.strip().split()))


class LLMGateway:
    def __init__(self, provider: LLMProvider | None = None, fallback_provider: LLMProvider | None = None,
                 registry: Any | None = None, config: GatewayConfig | None = None, **_: Any):
        self.provider = provider or LocalProvider()
        self.fallback_provider = fallback_provider
        self.registry = registry
        self.config = config or GatewayConfig()
        self.metrics = GatewayMetricsSnapshot()
        self._breakers: dict[str, CircuitBreaker] = {}
        self.requests = 0

    def _breaker(self, provider: LLMProvider) -> CircuitBreaker:
        return self._breakers.setdefault(provider.name, CircuitBreaker(self.config.circuit_failure_threshold))

    def generate(self, request: LLMRequest | str, **kwargs: Any) -> LLMResponse:
        if isinstance(request, str): request = LLMRequest(prompt=request, metadata=kwargs)
        if not request.prompt.strip(): raise ValueError("prompt must not be empty")
        started = perf_counter(); self.requests += 1; self.metrics.requests += 1
        attempts: list[dict[str, Any]] = []
        providers = [self.provider] + ([self.fallback_provider] if self.fallback_provider else [])
        for index, provider in enumerate(providers):
            breaker = self._breaker(provider)
            if not breaker.allow_request(): continue
            if index: self.metrics.fallbacks += 1
            for retry in range(self.config.max_retries + 1):
                try:
                    response = provider.generate(request)
                    breaker.record_success()
                    response.latency_ms = round((perf_counter() - started) * 1000, 3)
                    response.metadata = {**response.metadata, "attempts": attempts, "fallback": bool(index)}
                    self.metrics.successes += 1; self.metrics.total_latency_ms += response.latency_ms
                    self.metrics.input_tokens += response.input_tokens; self.metrics.output_tokens += response.output_tokens
                    return response
                except Exception as exc:
                    breaker.record_failure(); self.metrics.retries += int(retry < self.config.max_retries)
                    attempts.append({"provider": provider.name, "retry": retry, "error": str(exc)})
        self.metrics.failures += 1
        latency = round((perf_counter() - started) * 1000, 3); self.metrics.total_latency_ms += latency
        return LLMResponse("No LLM provider succeeded; retrieved context is available for review.", latency_ms=latency, metadata={"attempts": attempts})

    def snapshot(self) -> dict[str, Any]: return self.metrics.to_dict()
