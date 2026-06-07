"""Fixed metrics for RAG evaluation. All metrics compare document IDs only (no LLM)."""
import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@K: fraction of relevant docs found in top-K retrieved."""
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """MRR@K: reciprocal rank of first relevant doc in top-K."""
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(all_retrieved: dict[str, list[str]], qrels: dict[str, list[str]], k_values: list[int] = [1, 10]):
    """Compute recall@k and mrr@k over all queries."""
    metrics = {}
    for k in k_values:
        recalls, mrrs = [], []
        for qid, retrieved in all_retrieved.items():
            if qid in qrels:
                relevant = qrels[qid]
                recalls.append(recall_at_k(retrieved, relevant, k))
                mrrs.append(mrr_at_k(retrieved, relevant, k))
        metrics[f"recall@{k}"] = np.mean(recalls) if recalls else 0.0
        metrics[f"mrr@{k}"] = np.mean(mrrs) if mrrs else 0.0
    return metrics
