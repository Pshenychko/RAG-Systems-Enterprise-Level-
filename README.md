# RAG Scaling Experiment

R&D-дослідження деградації RAG-системи при масштабуванні корпусу від 1K до 300K passages.

## Гіпотеза

При scaling корпусу recall@1 деградує швидше за recall@10, бо релевантний документ сповзає з першої позиції під тиском семантично близьких distractors. Latency зростає лінійно (brute-force O(N)). Hybrid retrieval (BM25 + dense + RRF) компенсує деградацію.

## Результати

| Corpus | R@1 | R@10 | MRR@10 | Latency p50 | RAM |
|--------|-----|------|--------|-------------|-----|
| 1K | 0.955 | 1.000 | 0.992 | 0.04ms | 1MB |
| 10K | 0.899 | 0.994 | 0.955 | 0.10ms | 15MB |
| 100K | 0.774 | 0.960 | 0.859 | 1.4ms | 146MB |
| 300K | 0.703 | 0.926 | 0.793 | 3.7ms | 439MB |

**Гіпотеза частково підтверджена**: recall@1 деградує на 26%, recall@10 лише на 7%. Однак hybrid BM25+dense не допоміг (погіршив результат на MS MARCO). HNSW дає 7× зниження latency.

## Стек

- **Embeddings**: BAAI/bge-small-en-v1.5 (dim=384)
- **Dataset**: MS MARCO Passage Ranking (BeIR)
- **Vector index**: numpy brute-force (baseline) → FAISS HNSW (fix)
- **Sparse**: BM25 via rank_bm25
- **Eval**: Recall@K, MRR@K (no LLM, $0)

## Запуск

```bash
pip install -r requirements.txt

# Повний pipeline (embedding + eval, ~30 хв на CPU)
python pipeline.py

# Тільки eval (потрібні закешовані embeddings)
python eval_only.py
```

## Структура

```
├── pipeline.py          # Повний pipeline: embed + scale + eval + fix
├── eval_only.py         # Eval з кешованих embeddings (без sentence_transformers)
├── template/
│   ├── data_loader.py   # MS MARCO streaming loader
│   └── metrics.py       # recall@k, mrr@k
├── results/
│   ├── results.json     # Числові результати
│   └── scaling_plots.png # Графіки
├── REPORT.md            # Детальний звіт
└── requirements.txt
```

## Висновки

Детальний аналіз — у [REPORT.md](REPORT.md).
