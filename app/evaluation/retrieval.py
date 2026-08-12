def recall_at_k(results,relevant,k):
    return sum(bool(set(r[:k])&set(rel)) for r,rel in zip(results,relevant))/len(results) if results else 0.0
def mrr(results,relevant):
    vals=[]
    for r,rel in zip(results,relevant):
        vals.append(next((1/(i+1) for i,x in enumerate(r) if x in rel),0.0))
    return sum(vals)/len(vals) if vals else 0.0

