from statistics import median
from time import perf_counter
def benchmark(fn,queries):
    cold=[]; warm=[]
    for q in queries:
        t=perf_counter(); fn(q); cold.append((perf_counter()-t)*1000)
    for q in queries:
        t=perf_counter(); fn(q); warm.append((perf_counter()-t)*1000)
    def stats(x): 
        y=sorted(x); return {"p50":median(y),"p95":y[min(len(y)-1,int(len(y)*.95))]}
    return {"cold":stats(cold),"warm":stats(warm)}

if __name__ == "__main__":
    from app.embeddings.gemini import GeminiEmbedder
    from app.ingestion.models import Document
    from app.retrieval.bm25 import BM25Retriever
    from app.retrieval.vector_store import InMemoryVectorStore
    from app.retrieval.service import RetrievalService
    embedder = GeminiEmbedder(); docs = [Document("retrieval engineering"), Document("agentic pipeline")]
    store = InMemoryVectorStore(embedder); store.upsert(docs)
    service = RetrievalService(store, BM25Retriever(docs))
    print(benchmark(service.search, ["retrieval", "pipeline"]))
