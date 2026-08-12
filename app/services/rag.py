from app.memory.store import ConversationStore
from app.services.context import ContextBuilder
from time import perf_counter


class RAGService:
    def __init__(self, retrieval, gateway=None, memory=None, context_builder=None):
        self.retrieval, self.gateway, self.memory = (
            retrieval,
            gateway,
            memory or ConversationStore(),
        )
        self.context_builder = context_builder or ContextBuilder()

    def query(
        self, query: str, conversation_id: str = "default", top_k: int | None = None
    ) -> dict:
        if not query.strip():
            raise ValueError("query must not be empty")
        started = perf_counter()
        self.memory.add_message(conversation_id, "user", query)
        retrieval_started = perf_counter()
        docs = self.retrieval.search(query, top_k)
        retrieval_latency = round((perf_counter() - retrieval_started) * 1000, 3)
        context, selected = self.context_builder.build(docs)
        answer = (
            f"Based on the retrieved context: {context[:600]}"
            if selected
            else "I could not find relevant context for that question."
        )
        metadata = {
            "context_chars": len(context),
            "retrieval": self.retrieval.metrics(),
        }
        llm_latency = 0.0
        input_tokens = output_tokens = total_tokens = 0
        if self.gateway:
            llm_started = perf_counter()
            response = self.gateway.generate(
                f"Context:\n{context}\n\nQuestion: {query}"
            )
            llm_latency = round((perf_counter() - llm_started) * 1000, 3)
            answer, metadata = response.text, {**metadata, **(response.metadata or {})}
            input_tokens, output_tokens, total_tokens = (
                response.input_tokens,
                response.output_tokens,
                response.total_tokens,
            )
        self.memory.add_message(conversation_id, "assistant", answer)
        metadata.update(
            {
                "retrieval_latency_ms": retrieval_latency,
                "llm_latency_ms": llm_latency,
                "total_latency_ms": round((perf_counter() - started) * 1000, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "source_count": len(selected),
                "candidates": len(docs),
            }
        )
        return {
            "answer": answer,
            "query": query,
            "conversation_id": conversation_id,
            "sources": [
                {"text": d.text, "source": d.source, "metadata": d.metadata}
                for d in selected
            ],
            "metadata": metadata,
        }
