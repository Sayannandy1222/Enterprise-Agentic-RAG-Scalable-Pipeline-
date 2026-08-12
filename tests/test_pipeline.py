from fastapi.testclient import TestClient

from app.main import app
from app.embeddings.gemini import GeminiEmbedder
from app.ingestion.models import Document
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector_store import InMemoryVectorStore
from app.retrieval.service import RetrievalService
from app.gateway.service import LLMGateway, LLMRequest, LLMResponse
from app.services.context import ContextBuilder


def test_cache_is_bounded_and_clearable():
    embedder = GeminiEmbedder(cache_size=2)
    embedder.embed("one")
    embedder.embed("two")
    embedder.embed("three")
    assert embedder.stats()["size"] == 2
    embedder.clear_cache()
    assert embedder.stats()["size"] == 0


def test_hybrid_retrieval_and_stage_metrics():
    docs = [Document("alpha semantic retrieval"), Document("beta lexical system")]
    embedder = GeminiEmbedder()
    store = InMemoryVectorStore(embedder)
    store.upsert(docs)
    service = RetrievalService(store, BM25Retriever(docs))
    assert service.search("alpha")
    assert {"dense", "bm25", "rerank"} <= set(service.metrics())


def test_api_upload_then_query():
    client = TestClient(app)
    uploaded = client.post(
        "/api/v1/documents/upload",
        files={"file": ("note.txt", b"hybrid retrieval", "text/plain")},
    )
    assert uploaded.status_code == 200
    queried = client.post("/api/v1/rag/query", json={"query": "retrieval"})
    assert queried.status_code == 200
    assert queried.json()["sources"]


def test_gateway_and_context_budget():
    response = LLMGateway().generate(
        LLMRequest(prompt="Context: alpha beta\nQuestion: explain")
    )
    assert (
        response.text
        and response.input_tokens > 0
        and response.total_tokens >= response.output_tokens
    )
    docs = [Document("word " * 100), Document("second " * 100)]
    context, selected = ContextBuilder(100).build(docs)
    assert len(context.split()) <= 100 and selected
