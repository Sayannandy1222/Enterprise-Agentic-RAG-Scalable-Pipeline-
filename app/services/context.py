class ContextBuilder:
    def __init__(self, token_budget=1200): self.token_budget=max(100, token_budget)
    def build(self, documents):
        selected=[]; used=0; parts=[]
        for doc in documents:
            remaining=self.token_budget-used-2
            if remaining<=0: break
            text=" ".join(doc.text.split()[:remaining]); selected.append(doc); parts.append(text); used += len(text.split())
        return "\n\n".join(f"[Source {i+1}] {text}" for i,text in enumerate(parts)), selected
