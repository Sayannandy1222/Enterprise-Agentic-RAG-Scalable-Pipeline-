# Enterprise Agentic RAG (Scalable Pipeline)

A production-oriented retrieval pipeline preserving the established design: Gemini embeddings with a bounded LRU cache, Qdrant dense retrieval, BM25 lexical retrieval, reciprocal-rank fusion, bounded candidate sets, FlashRank reranking, stage latency metrics, and retrieval evaluation.

## Run

`make compile`, `make test`, then `make run`. Gemini and Qdrant are optional for offline tests; the embedder and vector store use deterministic local behavior when unavailable.

## API

- `GET /health`, `GET /ready`, `GET /metrics`
- `POST /api/v1/documents/upload` (multipart file)
- `POST /api/v1/rag/query` with `{"query":"...", "conversation_id":"..." }`

## Recovery note

This is a coherent reconstruction of the deleted tree. Exact deleted bytes cannot be guaranteed where the original heredoc bodies were not retained; the architecture and tested optimizations are preserved.

