"""MS MARCO data loader with reproducible subsets. Guarantees relevant docs in corpus."""
import random
from datasets import load_dataset


def load_msmarco_queries(n_queries: int = 500, seed: int = 42):
    """Load queries and qrels from MS MARCO dev set."""
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/msmarco-qrels", split="validation")
    
    qrels = {}
    for item in qrels_ds:
        qid = str(item["query-id"])
        pid = str(item["corpus-id"])
        if qid not in qrels:
            qrels[qid] = []
        qrels[qid].append(pid)
    
    valid_queries = []
    for item in queries_ds:
        qid = str(item["_id"])
        if qid in qrels:
            valid_queries.append({"id": qid, "text": item["text"]})
    
    random.seed(seed)
    random.shuffle(valid_queries)
    return valid_queries[:n_queries], qrels


def load_msmarco_passages(n_passages: int, queries: list, qrels: dict, seed: int = 42):
    """Load corpus via streaming, ensuring relevant passages are included."""
    # Collect relevant passage IDs we need
    relevant_pids = set()
    for q in queries:
        if q["id"] in qrels:
            relevant_pids.update(qrels[q["id"]])
    
    print(f"    Need {len(relevant_pids)} relevant passages")
    
    # Stream corpus, collect relevant + random others
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)
    
    relevant_found = {}
    others = []
    
    for item in ds:
        pid = item["_id"]
        entry = {"id": pid, "text": item["text"], "title": item.get("title", "")}
        if pid in relevant_pids:
            relevant_found[pid] = entry
        else:
            others.append(entry)
        
        # Stop once we have enough: all relevant + enough filler
        if len(others) >= n_passages and len(relevant_found) >= len(relevant_pids):
            break
    
    print(f"    Found {len(relevant_found)}/{len(relevant_pids)} relevant passages")
    
    # Build final corpus: relevant + random subset of others
    random.seed(seed)
    random.shuffle(others)
    n_fill = max(0, n_passages - len(relevant_found))
    corpus = list(relevant_found.values()) + others[:n_fill]
    
    # Shuffle so relevant aren't all at start
    random.seed(seed + 1)
    random.shuffle(corpus)
    
    return corpus
