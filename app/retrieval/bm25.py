import re, math
class BM25Retriever:
    def __init__(self, documents=None):
        self.documents=list(documents or []); self._rebuild()
    def _rebuild(self):
        self.tokens=[re.findall(r"\w+",d.text.lower()) for d in self.documents]; self.df={}
        for ts in self.tokens:
            for t in set(ts): self.df[t]=self.df.get(t,0)+1
        self.avgdl=sum(map(len,self.tokens))/len(self.tokens) if self.tokens else 0
    def add(self,documents): self.documents.extend(documents); self._rebuild()
    def search(self,query,limit=16):
        q=set(re.findall(r"\w+",query.lower())); n=len(self.documents); out=[]
        for i,ts in enumerate(self.tokens):
            score=0.0
            for term in q:
                tf=ts.count(term)
                if tf: score += math.log((n+1)/(self.df.get(term,0)+1))*tf
            if score: out.append((score,self.documents[i]))
        return sorted(out,key=lambda x:x[0],reverse=True)[:limit]

