from time import perf_counter
from collections import deque
from app.reranking.flashrank import FlashRankReranker
from app.retrieval.fusion import reciprocal_rank_fusion
class RetrievalService:
    def __init__(self,vector_store,bm25,reranker=None,dense_limit=16,bm25_limit=16,rrf_limit=8,rerank_limit=8,top_k=5):
        self.vector_store,self.bm25,self.reranker=vector_store,bm25,reranker or FlashRankReranker()
        self.limits=(dense_limit,bm25_limit,rrf_limit,rerank_limit,top_k); self.latencies=deque(maxlen=10_000)
    def search(self,query,top_k=None):
        t=perf_counter(); dense=self.vector_store.search(query,self.limits[0]); self.latencies.append(("dense",(perf_counter()-t)*1000))
        t=perf_counter(); lexical=self.bm25.search(query,self.limits[1]); self.latencies.append(("bm25",(perf_counter()-t)*1000))
        fused=reciprocal_rank_fusion(dense, lexical, self.limits[2])
        t=perf_counter(); reranked=self.reranker.rerank(query,fused,self.limits[3]); self.latencies.append(("rerank",(perf_counter()-t)*1000))
        return reranked[:(top_k or self.limits[4])]
    def metrics(self):
        out={}
        for stage,ms in self.latencies: out.setdefault(stage,[]).append(ms)
        return {s:{"count":len(v),"p50":sorted(v)[len(v)//2],"p95":sorted(v)[min(len(v)-1,int(len(v)*.95))]} for s,v in out.items()}
