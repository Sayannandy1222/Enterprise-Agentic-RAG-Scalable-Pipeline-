from app.embeddings.gemini import GeminiEmbedder
from app.ingestion.models import Document
from app.ingestion.loaders.text import TextLoader
from app.retrieval.bm25 import BM25Retriever
from app.evaluation.retrieval import recall_at_k,mrr
def test_embedding_cache():
 e=GeminiEmbedder(cache_size=2); e.embed("a"); e.embed("a"); assert e.stats()["hits"]==1
def test_bm25():
 docs=[Document("alpha beta"),Document("gamma")]; assert BM25Retriever(docs).search("alpha")
def test_metrics():
 assert recall_at_k([["a"]],[{"a"}],1)==1; assert mrr([["x","a"]],[{"a"}])==.5

