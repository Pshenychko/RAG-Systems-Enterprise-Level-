"""
RAG Scaling Experiment Pipeline v3
===================================
Key fix: for each corpus size, we build a subset that ALWAYS includes
all relevant passages, filling the rest with random non-relevant passages.
This ensures evaluation is meaningful at every scale.
"""
import os, sys, time, json
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from template.metrics import evaluate_retrieval

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"
CORPUS_SIZES = [1_000, 10_000, 100_000, 300_000]
N_QUERIES = 500
SEED = 42
RESULTS_DIR = Path("results")
EMBEDDINGS_DIR = Path("embeddings_cache")
K_VALUES = [1, 10]

RESULTS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)


def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_texts(model, texts, batch_size=256, show_progress=True):
    start = time.time()
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, normalize_embeddings=True)
    elapsed = time.time() - start
    return np.array(embs, dtype=np.float32), len(texts) / elapsed


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
    """FAISS HNSW - wrapped to handle potential segfaults on Python 3.14."""
    try:
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
    except Exception as e:
        print(f"    FAISS error: {e}, falling back to brute-force")
        return brute_force_search(query_embs, corpus_embs, k)


def bm25_search(query_texts, corpus_texts, k=10):
    from rank_bm25 import BM25Okapi
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


def run_experiment():
    import random
    from datasets import load_dataset
    
    print("=" * 60)
    print("RAG SCALING EXPERIMENT v3")
    print("=" * 60)
    
    # ─── Load queries & qrels ─────────────────────────────────────────────
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
    print(f"  {len(queries)} queries, {len(relevant_pids)} unique relevant passages needed")
    
    # ─── Load corpus (stream, collect relevant + random filler) ───────────
    print("[2] Loading corpus via streaming...")
    ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)
    
    relevant_docs = {}
    filler_docs = []
    max_filler = max(CORPUS_SIZES)
    
    for item in tqdm(ds, desc="  Streaming corpus"):
        pid = item["_id"]
        text = f"{item.get('title', '')} {item['text']}".strip()
        if pid in relevant_pids:
            relevant_docs[pid] = text
        elif len(filler_docs) < max_filler:
            filler_docs.append((pid, text))
        
        if len(relevant_docs) >= len(relevant_pids) and len(filler_docs) >= max_filler:
            break
    
    print(f"  Found {len(relevant_docs)}/{len(relevant_pids)} relevant docs")
    print(f"  Collected {len(filler_docs)} filler docs")
    
    # Filter queries to those where we found relevant docs
    queries = [q for q in queries if any(pid in relevant_docs for pid in qrels[q["id"]])]
    print(f"  Valid queries after filtering: {len(queries)}")
    
    # Shuffle fillers
    random.seed(SEED)
    random.shuffle(filler_docs)
    
    # ─── Build full corpus: relevant + fillers ────────────────────────────
    # Put relevant docs at known positions, fillers fill the rest
    all_ids = list(relevant_docs.keys())
    all_texts = [relevant_docs[pid] for pid in all_ids]
    
    for pid, text in filler_docs:
        all_ids.append(pid)
        all_texts.append(text)
    
    # Shuffle everything together with fixed seed
    combined = list(zip(all_ids, all_texts))
    random.seed(SEED + 1)
    random.shuffle(combined)
    all_ids = [x[0] for x in combined]
    all_texts = [x[1] for x in combined]
    
    # ─── Embed ────────────────────────────────────────────────────────────
    print(f"[3] Embedding {len(all_ids)} passages...")
    model = get_embedder()
    
    query_texts = [q["text"] for q in queries]
    query_ids = [q["id"] for q in queries]
    query_embs, _ = embed_texts(model, query_texts, show_progress=False)
    
    cache_path = EMBEDDINGS_DIR / f"corpus_embs_v3_{len(all_ids)}.npy"
    if cache_path.exists():
        print("  Loading cached embeddings...")
        corpus_embs = np.load(cache_path)
    else:
        corpus_embs, throughput = embed_texts(model, all_texts, batch_size=256)
        np.save(cache_path, corpus_embs)
        print(f"  Throughput: {throughput:.1f} passages/sec")
    
    print(f"  Shape: {corpus_embs.shape}, RAM: {corpus_embs.nbytes/1024/1024:.0f} MB")
    
    # ─── Build mapping for each size ─────────────────────────────────────
    # For each corpus size N, we pick the first N passages from our shuffled corpus.
    # Since relevant docs are mixed in, smaller sizes have fewer relevant docs.
    # But we have ~537 relevant docs in 300K - ratio is ~0.18%, so even at 1K
    # we'd have ~2. Let's verify and adjust.
    
    # Actually better approach: for each size, guarantee ALL relevant docs +
    # fill with random filler up to that size.
    
    # Separate relevant and filler indices
    relevant_indices = [i for i, pid in enumerate(all_ids) if pid in relevant_docs]
    filler_indices = [i for i, pid in enumerate(all_ids) if pid not in relevant_docs]
    
    print(f"  Relevant indices: {len(relevant_indices)}, Filler indices: {len(filler_indices)}")
    
    # ─── Scaling loop ────────────────────────────────────────────────────
    results = {"baseline": [], "hnsw": [], "hybrid": []}
    
    print("\n" + "=" * 60)
    print("SCALING LOOP")
    print("=" * 60)
    
    for size in CORPUS_SIZES:
        print(f"\n{'─'*50}")
        print(f"CORPUS SIZE: {size:,}")
        print(f"{'─'*50}")
        
        # Build subset: all relevant + random filler up to size
        n_filler = size - len(relevant_indices)
        if n_filler < 0:
            # More relevant than size - take random subset of relevant
            random.seed(SEED)
            subset_indices = sorted(random.sample(relevant_indices, size))
        else:
            subset_indices = sorted(relevant_indices + filler_indices[:n_filler])
        
        sub_embs = corpus_embs[subset_indices]
        sub_ids = [all_ids[i] for i in subset_indices]
        sub_texts = [all_texts[i] for i in subset_indices]
        
        # Verify
        sub_id_set = set(sub_ids)
        n_relevant_in_sub = len(relevant_pids & sub_id_set)
        
        # Count valid queries (at least 1 relevant doc in subset)
        valid_q_indices = [i for i, qid in enumerate(query_ids) 
                          if any(pid in sub_id_set for pid in qrels[qid])]
        
        vq_embs = query_embs[valid_q_indices]
        vq_ids = [query_ids[i] for i in valid_q_indices]
        vq_texts = [query_texts[i] for i in valid_q_indices]
        
        print(f"  Relevant docs in subset: {n_relevant_in_sub}")
        print(f"  Valid queries: {len(vq_ids)}")
        
        if len(vq_ids) == 0:
            print("  SKIP")
            continue
        
        # ── Baseline ──
        print("  Baseline (brute-force)...")
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
              f"lat_p50={m['latency_p50']:.2f}ms  lat_p95={m['latency_p95']:.2f}ms")
        
        # ── HNSW ──
        print("  HNSW...")
        h_indices, h_latencies = faiss_hnsw_search(vq_embs, sub_embs, k=10)
        retrieved_h = {qid: [sub_ids[idx] for idx in h_indices[i]] for i, qid in enumerate(vq_ids)}
        hm = evaluate_retrieval(retrieved_h, qrels, K_VALUES)
        hm["latency_p50"] = np.percentile(h_latencies, 50) * 1000
        hm["latency_p95"] = np.percentile(h_latencies, 95) * 1000
        hm["corpus_size"] = size
        results["hnsw"].append(hm)
        print(f"    R@1={hm['recall@1']:.4f}  R@10={hm['recall@10']:.4f}  MRR@10={hm['mrr@10']:.4f}  "
              f"lat_p50={hm['latency_p50']:.2f}ms")
        
        # ── Hybrid ──
        if size <= 50_000:
            print("  Hybrid (BM25+Dense+RRF)...")
            bm25_idx = bm25_search(vq_texts, sub_texts, k=10)
            hybrid_idx = rrf_fusion(indices, bm25_idx, k=10)
            retrieved_hyb = {qid: [sub_ids[idx] for idx in hybrid_idx[i]] for i, qid in enumerate(vq_ids)}
            hym = evaluate_retrieval(retrieved_hyb, qrels, K_VALUES)
            hym["corpus_size"] = size
            results["hybrid"].append(hym)
            print(f"    R@1={hym['recall@1']:.4f}  R@10={hym['recall@10']:.4f}  MRR@10={hym['mrr@10']:.4f}")
        else:
            print("  Hybrid skipped (BM25 too slow for this size)")
            results["hybrid"].append({"recall@1": None, "recall@10": None, "mrr@10": None, "corpus_size": size})
    
    # ─── Save & plot ─────────────────────────────────────────────────────
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    plot_results(results)
    print_summary(results)
    print(f"\n✅ Done! Results in {RESULTS_DIR}/")


def plot_results(results):
    sizes = [r["corpus_size"] for r in results["baseline"]]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("RAG Scaling: Baseline vs Fixes (BGE-small-en-v1.5, MS MARCO)", fontsize=13)
    
    ax = axes[0, 0]
    ax.plot(sizes, [r["recall@1"] for r in results["baseline"]], "o-", label="Baseline", lw=2)
    ax.plot(sizes, [r["recall@1"] for r in results["hnsw"]], "s--", label="HNSW")
    hyb_r1 = [(r["recall@1"] if r["recall@1"] is not None else None) for r in results["hybrid"]]
    hyb_sizes = [s for s, v in zip(sizes, hyb_r1) if v is not None]
    hyb_vals = [v for v in hyb_r1 if v is not None]
    if hyb_vals:
        ax.plot(hyb_sizes, hyb_vals, "^--", label="Hybrid")
    ax.set_xlabel("Corpus size"); ax.set_ylabel("Recall@1"); ax.set_title("Recall@1")
    ax.legend(); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(sizes, [r["recall@10"] for r in results["baseline"]], "o-", label="Baseline", lw=2)
    ax.plot(sizes, [r["recall@10"] for r in results["hnsw"]], "s--", label="HNSW")
    hyb_r10 = [(r["recall@10"] if r["recall@10"] is not None else None) for r in results["hybrid"]]
    hyb_vals10 = [v for v in hyb_r10 if v is not None]
    if hyb_vals10:
        ax.plot(hyb_sizes, hyb_vals10, "^--", label="Hybrid")
    ax.set_xlabel("Corpus size"); ax.set_ylabel("Recall@10"); ax.set_title("Recall@10")
    ax.legend(); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(sizes, [r["latency_p50"] for r in results["baseline"]], "o-", label="BF p50", lw=2)
    ax.plot(sizes, [r["latency_p95"] for r in results["baseline"]], "o--", label="BF p95")
    ax.plot(sizes, [r["latency_p50"] for r in results["hnsw"]], "s-", label="HNSW p50")
    ax.plot(sizes, [r["latency_p95"] for r in results["hnsw"]], "s--", label="HNSW p95")
    ax.set_xlabel("Corpus size"); ax.set_ylabel("Latency (ms)"); ax.set_title("Latency")
    ax.legend(); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(sizes, [r["ram_mb"] for r in results["baseline"]], "o-", lw=2)
    ax.set_xlabel("Corpus size"); ax.set_ylabel("RAM (MB)"); ax.set_title("Embeddings RAM")
    ax.set_xscale("log"); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scaling_plots.png", dpi=150)
    plt.close()


def print_summary(results):
    print("\n" + "=" * 90)
    print(f"{'Size':<8} {'R@1 BL':<9} {'R@1 HNSW':<10} {'R@1 Hyb':<9} "
          f"{'R@10 BL':<9} {'MRR BL':<9} {'Lat BL':<9} {'Lat HNSW':<9} {'RAM MB':<8}")
    print("=" * 90)
    for i in range(len(results["baseline"])):
        bl = results["baseline"][i]
        hn = results["hnsw"][i]
        hy = results["hybrid"][i]
        hyb_r1 = f"{hy['recall@1']:.4f}" if hy['recall@1'] is not None else "N/A"
        print(f"{bl['corpus_size']:<8} {bl['recall@1']:<9.4f} {hn['recall@1']:<10.4f} {hyb_r1:<9} "
              f"{bl['recall@10']:<9.4f} {bl['mrr@10']:<9.4f} {bl['latency_p50']:<9.2f} "
              f"{hn['latency_p50']:<9.2f} {bl['ram_mb']:<8.0f}")


if __name__ == "__main__":
    run_experiment()
