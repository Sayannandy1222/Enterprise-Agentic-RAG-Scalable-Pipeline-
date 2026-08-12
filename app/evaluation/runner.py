from app.evaluation.retrieval import recall_at_k,mrr
def main():
    results=[["a","b","c"],["x","y","z"],["q","r"]]; rel=[{"a"},{"z"},{"r"}]
    print({"recall_at_3":recall_at_k(results,rel,3),"recall_at_5":recall_at_k(results,rel,5),"mrr":mrr(results,rel)})
if __name__=="__main__": main()

