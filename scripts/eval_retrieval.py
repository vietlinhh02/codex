#!/usr/bin/env python
"""
Evaluate retrieval on prepared data with precision/recall@K.

Inputs:
- Corpus: CSV/JSONL from `scripts/prepare_ise_kvjson.py` (doc_id,text[,title])
- Claims: valid/train JSONL from the same prepare step (expects `claim` and `gold_evidence_ids`)

Methods evaluated:
- bm25: rank-bm25 over corpus text
- dense: SentenceTransformers dense index (FAISS if available)
- hybrid: RRF fusion of dense + bm25

Usage example:
  python scripts/eval_retrieval.py \
    --corpus prepared/warmup/corpus.csv --format csv \
    --claims prepared/warmup/valid.jsonl \
    --topk 10 --bm25_k 200 --dense_k 200

Notes:
- Only items with non-empty `gold_evidence_ids` are included in evaluation.
- You may override the dense model via `--dense_model`.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vifact.config import PipelineConfig
from vifact.data import load_corpus_from_csv, load_corpus_from_jsonl
from vifact.pipeline import ViFactPipeline
from vifact.retrieval.hybrid import reciprocal_rank_fusion
from vifact.prep.kvjson import tokenize_bm25


def read_claims_with_gold(path: str) -> List[Tuple[str, List[str]]]:
    items: List[Tuple[str, List[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            claim = obj.get("claim") or obj.get("text")
            gold: List[str] = obj.get("gold_evidence_ids") or []
            if claim and gold:
                items.append((str(claim), [str(g) for g in gold]))
    return items


def build_bm25_index(corpus_texts: Sequence[str]):
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize_bm25(t) for t in corpus_texts]
    return BM25Okapi(tokenized)


def bm25_search(bm25, query: str, topk: int) -> List[int]:
    tokens = tokenize_bm25(query)
    scores = bm25.get_scores(tokens)
    idx = np.argsort(-scores)[:topk]
    return idx.tolist()


def eval_metrics(golds: List[Iterable[str]], preds: List[Iterable[str]]) -> Dict[str, float]:
    assert len(golds) == len(preds)
    precs, recs, hits = [], [], 0
    for gold, pred in zip(golds, preds):
        g = set(gold)
        p = list(pred)
        inter = g.intersection(set(p))
        if inter:
            hits += 1
        precs.append(len(inter) / max(1, len(p)))
        recs.append(len(inter) / max(1, len(g)))
    n = max(1, len(golds))
    return {
        "hit_rate@k": round(hits / n, 4),
        "mean_precision@k": round(sum(precs) / n, 4),
        "mean_recall@k": round(sum(recs) / n, 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate BM25/Dense/Hybrid retrieval with precision/recall@K")
    ap.add_argument("--corpus", required=True, help="Path to CSV/JSONL corpus")
    ap.add_argument("--format", choices=["csv", "jsonl"], required=True)
    ap.add_argument("--id-col", default="doc_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--title-col", default=None)
    ap.add_argument("--claims", required=True, help="Path to valid/train JSONL from prepare step")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--bm25_k", type=int, default=200, help="BM25 candidate pool for fusion")
    ap.add_argument("--dense_k", type=int, default=200, help="Dense candidate pool for fusion")
    ap.add_argument("--dense_model", default=None, help="Optional override for dense model name")
    args = ap.parse_args()

    # Load corpus
    if args.format == "csv":
        docs = load_corpus_from_csv(args.corpus, text_col=args.text_col, id_col=args.id_col, title_col=args.title_col)
    else:
        docs = load_corpus_from_jsonl(args.corpus, text_field=args.text_col, id_field=args.id_col, title_field=args.title_col)

    # Build Dense index via pipeline
    cfg = PipelineConfig()
    if args.dense_model:
        cfg.retrieval.dense_model = args.dense_model
    cfg.retrieval.top_k_dense = max(args.topk, args.dense_k)
    cfg.retrieval.top_k_bm25 = max(args.topk, args.bm25_k)
    cfg.rerank.top_k = args.topk
    pipe = ViFactPipeline(cfg)
    pipe.build_dense_index(docs)

    # Build BM25 index (dynamic, independent of bm25.json)
    doc_ids = [d.doc_id for d in docs]
    texts = [d.text for d in docs]
    bm25 = build_bm25_index(texts)

    # Load claims with gold
    items = read_claims_with_gold(args.claims)
    if not items:
        raise SystemExit("No items with non-empty gold_evidence_ids found in claims file.")

    gold_lists: List[List[str]] = []
    bm25_preds: List[List[str]] = []
    dense_preds: List[List[str]] = []
    hybrid_preds: List[List[str]] = []

    for claim, gold in items:
        gold_lists.append(gold)

        # BM25-only
        b_idx = bm25_search(bm25, claim, topk=args.topk)
        bm25_preds.append([doc_ids[i] for i in b_idx])

        # Dense-only
        dense_scored = pipe.dense.search(claim, top_k=args.topk)
        dense_preds.append([d for d, _ in dense_scored])

        # Hybrid (RRF over pools)
        b_pool_idx = bm25_search(bm25, claim, topk=args.bm25_k)
        b_pool_ids = [doc_ids[i] for i in b_pool_idx]
        d_pool_scored = pipe.dense.search(claim, top_k=args.dense_k)
        d_pool_ids = [d for d, _ in d_pool_scored]
        fused = reciprocal_rank_fusion([d_pool_ids, b_pool_ids], k=cfg.retrieval.rrf_k, top_k=args.topk)
        hybrid_preds.append(fused)

    report = {
        "k": args.topk,
        "bm25": eval_metrics(gold_lists, bm25_preds),
        "dense": eval_metrics(gold_lists, dense_preds),
        "hybrid_rrf": eval_metrics(gold_lists, hybrid_preds),
        "n_items": len(items),
        "dense_model": cfg.retrieval.dense_model,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

