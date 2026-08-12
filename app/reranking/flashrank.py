from functools import lru_cache
class FlashRankReranker:
    _ranker=None
    def __init__(self, model_name="ms-marco-MiniLM-L-12-v2"):
        self.model_name=model_name
        if FlashRankReranker._ranker is None:
            try:
                from flashrank import Ranker
                FlashRankReranker._ranker=Ranker(model_name=model_name)
            except Exception: FlashRankReranker._ranker=False
    def rerank(self,query,documents,limit=8):
        docs=list(documents)
        if not docs:return []
        if FlashRankReranker._ranker:
            try:
                from flashrank import RerankRequest
                req=RerankRequest(query=query,passages=[{"id":i,"text":d.text if hasattr(d,"text") else str(d)} for i,d in enumerate(docs)])
                ranked=self._ranker.rerank(req)
                return [docs[x["id"]] for x in ranked[:limit]]
            except Exception: pass
        terms=set(query.lower().split())
        return sorted(docs,key=lambda d:sum(t in d.text.lower() for t in terms),reverse=True)[:limit]

