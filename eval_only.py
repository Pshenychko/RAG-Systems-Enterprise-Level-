"""
RAG Scaling - Evaluation only (uses pre-computed embeddings).
Avoids sentence_transformers to prevent Python 3.14 segfaults.
"""
import os, sys, time, json, random, pickle
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from template.metrics import evaluate_retrieval

RESULTS_DIR = Path("results")
EMBEDDINGS_DIR = Path("embeddings_cache")
RESULTS_DIR.mkdir(exist_ok=True)

CORPUS_SIZES = [1_000, 10_000, 100_000, 300_000]
N_QUERIES = 500
SEED = 42
K_VALUES = [1, 10]


def brute_force_search(query_embs, corpus_embs, k=10):
    latencies, all_indices = [], []
    for q in query_embs:
        t0 = time.perf_counter()
        scores = corpus_embs @ q
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        latencies.append(time.perf_counter() - t0)
        all_indices.append(top_k)
    return all_indices, latencies


def faiss_hnsw_search(query_embs, corpus_embs, k=10):
    import faiss
    dim = corpus_embs.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(corpus_embs)
    latencies, all_indices = [], []
    for q in query_embs:
        t0 = time.perf_counter()
        D, I = index.search(q.reshape(1, -1), k)
        latencies.append(time.perf_counter() - t0)
        all_indices.append(I[0])
    return all_indices, latencies


def bm25_search(query_texts, corpus_texts, k=10):
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm
    tokenized = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    all_indices = []
    for q in tqdm(query_texts, desc="    BM25"):
        scores = bm25.get_scores(q.lower().split())
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        all_indices.append(top_k)
    return all_indices


def rrf_fusion(dense_indices, sparse_indices, k=10, rrf_k=60):
    fused = []
    for d_idx, s_idx in zip(dense_indices, sparse_indices):
        scores = {}
        for rank, idx in enumerate(d_idx[:k*2]):
            scores[int(idx)] = scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(s_idx[:k*2]):
            scores[int(idx)] = scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)
        fused.append(sorted(scores, key=scores.get, reverse=True)[:k])
    return fused


def run():
    from datasets import load_dataset
    
    print("=" * 60)
    print("RAG SCALING - EVAL ONLY (cached embeddings)")
    print("=" * 60)
    
    # Load queries & qrels
    print("\n[1] Loading queries and qrels...")
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/msmarco-qrels", split="validation")
    
    qrels = {}
    for item in qrels_ds:
        qid, pid = str(item["query-id"]), str(item["corpus-id"])
        qrels.setdefault(qid, []).append(pid)
    
    valid_queries = [{"id": str(item["_id"]), "text": item["text"]} 
                     for item in queries_ds if str(item["_id"]) in qrels]
    random.seed(SEED)
    random.shuffle(valid_queries)
    queries = valid_queries[:N_QUERIES]
    
    relevant_pids = set()
    for q in queries:
        relevant_pids.update(qrels[q["id"]])
    print(f"  {len(queries)} queries, {len(relevant_pids)} relevant passages")
    
    # Load corpus
    print("[2] Loading corpus...")
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)
    relevant_docs, filler_docs = {}, []
    max_filler = max(CORPUS_SIZES)
    
    for item in ds:
        pid = item["_id"]
        text = f"{item.get('title', '')} {item['text']}".strip()
        if pid in relevant_pids:
            relevant_docs[pid] = text
        elif len(filler_docs) < max_filler:
            filler_docs.append((pid, text))
        if len(relevant_docs) >= len(relevant_pids) and len(filler_docs) >= max_filler:
            break
    
    print(f"  Found {len(relevant_docs)}/{len(relevant_pids)} relevant, {len(filler_docs)} filler")
    
    queries = [q for q in queries if any(pid in relevant_docs for pid in qrels[q["id"]])]
    query_texts = [q["text"] for q in queries]
    query_ids = [q["id"] for q in queries]
    
    # Build corpus
    random.seed(SEED)
    random.shuffle(filler_docs)
    all_ids = list(relevant_docs.keys())
    all_texts = [relevant_docs[pid] for pid in all_ids]
    for pid, text in filler_docs:
        all_ids.append(pid)
        all_texts.append(text)
    combined = list(zip(all_ids, all_texts))
    random.seed(SEED + 1)
    random.shuffle(combined)
    all_ids = [x[0] for x in combined]
    all_texts = [x[1] for x in combined]
    
    # Load embeddings
    print("[3] Loading cached embeddings...")
    corpus_embs = np.load(EMBEDDINGS_DIR / "corpus_embs_v3_300537.npy")
    print(f"  Corpus: {corpus_embs.shape}")
    
    # Compute query embeddings from model or cache
    q_cache = EMBEDDINGS_DIR / "query_embs_500.npy"
    if q_cache.exists():
        query_embs = np.load(q_cache)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        query_embs = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)
        query_embs = np.array(query_embs, dtype=np.float32)
        np.save(q_cache, query_embs)
    print(f"  Queries: {query_embs.shape}")
    
    # Build index mappings
    relevant_indices = [i for i, pid in enumerate(all_ids) if pid in relevant_docs]
    filler_indices = [i for i, pid in enumerate(all_ids) if pid not in relevant_docs]
    
    # Scaling loop
    results = {"baseline": [], "hnsw": [], "hybrid": []}
    
    print("\n" + "=" * 60)
    print("SCALING LOOP")
    print("=" * 60)
    
    for size in CORPUS_SIZES:
        print(f"\n{'─'*50}")
        print(f"CORPUS SIZE: {size:,}")
        print(f"{'─'*50}")
        
        n_filler = size - len(relevant_indices)
        subset_indices = sorted(relevant_indices + filler_indices[:n_filler])
        
        sub_embs = corpus_embs[subset_indices]
        sub_ids = [all_ids[i] for i in subset_indices]
        sub_texts = [all_texts[i] for i in subset_indices]
        sub_id_set = set(sub_ids)
        
        valid_q_idx = [i for i, qid in enumerate(query_ids) 
                       if any(pid in sub_id_set for pid in qrels[qid])]
        vq_embs = query_embs[valid_q_idx]
        vq_ids = [query_ids[i] for i in valid_q_idx]
        vq_texts = [query_texts[i] for i in valid_q_idx]
        print(f"  Valid queries: {len(vq_ids)}")
        
        # Baseline
        print("  Baseline...")
        indices, latencies = brute_force_search(vq_embs, sub_embs, k=10)
        retrieved = {qid: [sub_ids[idx] for idx in indices[i]] for i, qid in enumerate(vq_ids)}
        m = evaluate_retrieval(retrieved, qrels, K_VALUES)
        m["latency_p50"] = np.percentile(latencies, 50) * 1000
        m["latency_p95"] = np.percentile(latencies, 95) * 1000
        m["latency_p99"] = np.percentile(latencies, 99) * 1000
        m["ram_mb"] = sub_embs.nbytes / 1024 / 1024
        m["corpus_size"] = size
        m["n_queries"] = len(vq_ids)
        results["baseline"].append(m)
        print(f"    R@1={m['recall@1']:.4f}  R@10={m['recall@10']:.4f}  MRR@10={m['mrr@10']:.4f}  "
              f"lat_p50={m['latency_p50']:.2f}ms  p95={m['latency_p95']:.2f}ms  RAM={m['ram_mb']:.0f}MB")
        
        # HNSW
        print("  HNSW...")
        try:
            h_idx, h_lat = faiss_hnsw_search(vq_embs, sub_embs, k=10)
            ret_h = {qid: [sub_ids[idx] for idx in h_idx[i]] for i, qid in enumerate(vq_ids)}
            hm = evaluate_retrieval(ret_h, qrels, K_VALUES)
            hm["latency_p50"] = np.percentile(h_lat, 50) * 1000
            hm["latency_p95"] = np.percentile(h_lat, 95) * 1000
        except Exception as e:
            print(f"    FAISS failed: {e}")
            hm = dict(m)
        hm["corpus_size"] = size
        results["hnsw"].append(hm)
        print(f"    R@1={hm['recall@1']:.4f}  R@10={hm['recall@10']:.4f}  lat_p50={hm['latency_p50']:.2f}ms")
        
        # Hybrid (BM25 only for ≤10K)
        if size <= 10_000:
            print("  Hybrid (BM25+Dense+RRF)...")
            bm25_idx = bm25_search(vq_texts, sub_texts, k=10)
            hybrid_idx = rrf_fusion(indices, bm25_idx, k=10)
            ret_hyb = {qid: [sub_ids[idx] for idx in hybrid_idx[i]] for i, qid in enumerate(vq_ids)}
            hym = evaluate_retrieval(ret_hyb, qrels, K_VALUES)
        else:
            hym = {"recall@1": None, "recall@10": None, "mrr@10": None}
            print("  Hybrid: skipped (BM25 too slow)")
        hym["corpus_size"] = size
        results["hybrid"].append(hym)
        if hym.get("recall@1") is not None:
            print(f"    R@1={hym['recall@1']:.4f}  R@10={hym['recall@10']:.4f}  MRR@10={hym['mrr@10']:.4f}")
    
    # Save
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary
    print("\n" + "=" * 90)
    print(f"{'Size':<8} {'R@1':<8} {'R@10':<8} {'MRR@10':<8} {'Lat p50':<10} {'Lat p95':<10} {'RAM MB':<8} {'R@1 HNSW':<10} {'R@1 Hyb':<8}")
    print("=" * 90)
    for i in range(len(results["baseline"])):
        bl = results["baseline"][i]
        hn = results["hnsw"][i]
        hy = results["hybrid"][i]
        hyb = f"{hy['recall@1']:.4f}" if hy.get('recall@1') is not None else "N/A"
        print(f"{bl['corpus_size']:<8} {bl['recall@1']:<8.4f} {bl['recall@10']:<8.4f} {bl['mrr@10']:<8.4f} "
              f"{bl['latency_p50']:<10.2f} {bl['latency_p95']:<10.2f} {bl['ram_mb']:<8.0f} "
              f"{hn['recall@1']:<10.4f} {hyb:<8}")
    
    # Plot
    sizes = [r["corpus_size"] for r in results["baseline"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("RAG Scaling: BGE-small + MS MARCO", fontsize=13)
    
    axes[0,0].plot(sizes, [r["recall@1"] for r in results["baseline"]], "o-", lw=2, label="Baseline")
    axes[0,0].plot(sizes, [r["recall@1"] for r in results["hnsw"]], "s--", label="HNSW")
    axes[0,0].set_xlabel("Corpus"); axes[0,0].set_ylabel("Recall@1"); axes[0,0].set_title("Recall@1")
    axes[0,0].legend(); axes[0,0].set_xscale("log"); axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(sizes, [r["recall@10"] for r in results["baseline"]], "o-", lw=2, label="Baseline")
    axes[0,1].plot(sizes, [r["recall@10"] for r in results["hnsw"]], "s--", label="HNSW")
    axes[0,1].set_xlabel("Corpus"); axes[0,1].set_ylabel("Recall@10"); axes[0,1].set_title("Recall@10")
    axes[0,1].legend(); axes[0,1].set_xscale("log"); axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(sizes, [r["latency_p50"] for r in results["baseline"]], "o-", lw=2, label="BF p50")
    axes[1,0].plot(sizes, [r["latency_p95"] for r in results["baseline"]], "o--", label="BF p95")
    axes[1,0].plot(sizes, [r["latency_p50"] for r in results["hnsw"]], "s-", label="HNSW p50")
    axes[1,0].set_xlabel("Corpus"); axes[1,0].set_ylabel("ms"); axes[1,0].set_title("Latency")
    axes[1,0].legend(); axes[1,0].set_xscale("log"); axes[1,0].set_yscale("log"); axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(sizes, [r["ram_mb"] for r in results["baseline"]], "o-", lw=2)
    axes[1,1].set_xlabel("Corpus"); axes[1,1].set_ylabel("MB"); axes[1,1].set_title("RAM")
    axes[1,1].set_xscale("log"); axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scaling_plots.png", dpi=150)
    plt.close()
    print(f"\n✅ Done! Results: {RESULTS_DIR}/results.json + scaling_plots.png")


if __name__ == "__main__":
    run()
