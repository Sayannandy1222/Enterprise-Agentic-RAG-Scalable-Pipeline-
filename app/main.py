from fastapi import FastAPI
from app.config import get_settings
from app.embeddings.gemini import GeminiEmbedder
from app.retrieval.vector_store import QdrantVectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.service import RetrievalService
from app.services.rag import RAGService
from app.services.context import ContextBuilder
from app.gateway.service import LLMGateway, LocalProvider
from app.gateway.providers import GroqProvider, PortkeyProvider
from app.api import rag,documents,metrics
settings=get_settings(); app=FastAPI(title=settings.app_name)
embedder=GeminiEmbedder(settings.gemini_api_key,settings.gemini_embedding_model,settings.embedding_cache_size,settings.embedding_dimensions)
store=QdrantVectorStore(embedder,settings.qdrant_url,settings.qdrant_collection)
retrieval=RetrievalService(store,BM25Retriever(),dense_limit=settings.dense_candidate_limit,bm25_limit=settings.bm25_candidate_limit,rrf_limit=settings.rrf_candidate_limit,rerank_limit=settings.rerank_candidate_limit,top_k=settings.top_k)
app.state.retrieval=retrieval; app.state.embedder=embedder
provider = None
if settings.portkey_api_key and settings.portkey_virtual_key:
    provider = PortkeyProvider(settings.portkey_api_key, settings.portkey_virtual_key, settings.groq_model)
elif settings.groq_api_key:
    provider = GroqProvider(settings.groq_api_key, settings.groq_model)
app.state.gateway = LLMGateway(provider=provider or LocalProvider(), fallback_provider=LocalProvider() if provider else None)
app.state.context_builder=ContextBuilder(settings.context_token_budget)
app.state.rag_service=RAGService(retrieval, gateway=app.state.gateway, memory=None, context_builder=app.state.context_builder)
app.include_router(rag.router); app.include_router(documents.router); app.include_router(metrics.router)
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/ready")
def ready(): return {"status":"ready", "embedding_cache": embedder.stats(), "documents": store.count()}
