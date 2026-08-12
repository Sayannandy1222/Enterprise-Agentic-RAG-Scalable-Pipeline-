from collections import defaultdict

def reciprocal_rank_fusion(dense, lexical, limit=8, k=60):
    scores=defaultdict(float); docs={}
    for rank,(_,doc) in enumerate(dense): scores[id(doc)] += 1/(k+rank+1); docs[id(doc)] = doc
    for rank,(_,doc) in enumerate(lexical): scores[id(doc)] += 1/(k+rank+1); docs[id(doc)] = doc
    return sorted(docs.values(), key=lambda d: scores[id(d)], reverse=True)[:limit]
