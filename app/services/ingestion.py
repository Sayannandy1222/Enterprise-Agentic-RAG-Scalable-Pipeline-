from app.ingestion.loaders.router import LoaderRouter
from app.ingestion.models import Chunk
class IngestionService:
    def __init__(self, router=None, chunk_size=800, overlap=100):
        self.router=router or LoaderRouter(); self.chunk_size=chunk_size; self.overlap=overlap
    def load(self,path): return self.router.load(path)
    def chunk(self, document):
        text=document.text.strip(); out=[]; start=0
        while start<len(text):
            end=min(len(text),start+self.chunk_size); out.append(Chunk(text[start:end],document.source,{**document.metadata,"start":start,"end":end}))
            if end==len(text): break
            start=max(start+1,end-self.overlap)
        return out
    def ingest(self,path): return self.chunk(self.load(path))

