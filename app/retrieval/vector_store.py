import hashlib
from typing import Iterable
class InMemoryVectorStore:
    def __init__(self,embedder=None): self.embedder=embedder; self.items=[]; self._vectors={}
    def upsert(self,documents: Iterable):
        for d in documents:
            if d not in self.items:
                self.items.append(d)
                self._vectors[id(d)] = self.embedder.embed(d.text)
    add = upsert
    def count(self) -> int: return len(self.items)
    def search(self,query,limit=16):
        q=self.embedder.embed(query); scored=[]
        for d in self.items:
            v=self._vectors.get(id(d)) or self.embedder.embed(d.text); dot=sum(a*b for a,b in zip(q,v)); scored.append((dot,d))
        return sorted(scored,key=lambda x:x[0],reverse=True)[:limit]
class QdrantVectorStore(InMemoryVectorStore):
    def __init__(self,embedder,url="",collection="documents"):
        super().__init__(embedder); self.url=url; self.collection=collection; self.client=None
        try:
            from qdrant_client import QdrantClient
            if url: self.client=QdrantClient(url=url,timeout=1)
        except Exception: self.client=None
    def upsert(self, documents: Iterable):
        docs=list(documents); super().upsert(docs)
        if not self.client or not docs: return
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            dim=len(self.embedder.embed(docs[0].text))
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(self.collection, vectors_config=VectorParams(size=dim,distance=Distance.COSINE))
            points=[]
            for i,d in enumerate(docs):
                pid=hashlib.sha1(f"{d.source}:{i}:{d.text}".encode()).hexdigest()[:16]
                points.append(PointStruct(id=pid, vector=self.embedder.embed(d.text), payload={"text":d.text,"source":getattr(d,"source",""),"metadata":getattr(d,"metadata",{})}))
            self.client.upsert(self.collection, points=points)
        except Exception: self.client=None
    def search(self, query, limit=16):
        if self.client:
            try:
                hits=self.client.search(self.collection, query_vector=self.embedder.embed(query), limit=limit)
                from app.ingestion.models import Document
                return [(float(h.score), Document(h.payload.get("text",""),h.payload.get("source",""),h.payload.get("metadata",{}))) for h in hits]
            except Exception: self.client=None
        return super().search(query,limit)
