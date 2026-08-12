from app.gateway.service import LLMGateway

class GenerationService:
    SYSTEM_PROMPT = "Answer using only the supplied context. If context is insufficient, say so."
    def __init__(self, gateway=None): self.gateway = gateway or LLMGateway()
    def generate(self, query: str, context: str, conversation: str = "") -> dict:
        prompt = f"{self.SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n{conversation}"[:12000]
        response = self.gateway.generate(prompt)
        return {"answer": response.text, "provider": response.provider, "model": response.model, "latency_ms": response.latency_ms, "metadata": response.metadata or {}}
