#!/usr/bin/env python
import argparse
import json
from pathlib import Path

# Bootstrap repo-local src on sys.path for direct script runs (Windows-friendly)
try:
    import vifact  # type: ignore
except Exception:
    import os, sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src = os.path.join(_root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from vifact.config import PipelineConfig
from vifact.data import load_corpus_from_csv, load_corpus_from_jsonl
from vifact.pipeline import ViFactPipeline


def main():
    ap = argparse.ArgumentParser(description="Run ViFact pipeline for claims")
    ap.add_argument("--corpus", required=True, help="Path to CSV or JSONL corpus")
    ap.add_argument("--format", choices=["csv", "jsonl"], required=True)
    ap.add_argument("--id-col", default="doc_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--title-col", default=None)
    ap.add_argument("--claims", required=False, help="Path to a JSONL/CSV with claims")
    ap.add_argument("--claim", required=False, help="Single claim string")
    ap.add_argument("--bm25", required=False, help="Optional path to JSON file mapping claim->list[doc_id]")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--reasoner_ckpt", required=False, help="Optional reasoner checkpoint dir to use for label")
    ap.add_argument("--out", required=False, help="Output JSONL path")
    args = ap.parse_args()

    if args.format == "csv":
        docs = load_corpus_from_csv(args.corpus, text_col=args.text_col, id_col=args.id_col, title_col=args.title_col)
    else:
        docs = load_corpus_from_jsonl(args.corpus, text_field=args.text_col, id_field=args.id_col, title_field=args.title_col)

    cfg = PipelineConfig()
    cfg.rerank.top_k = args.topk
    pl = ViFactPipeline(cfg, reasoner_ckpt=args.reasoner_ckpt)
    pl.build_dense_index(docs)

    bm25_map = None
    if args.bm25:
        with open(args.bm25, "r", encoding="utf-8") as f:
            bm25_map = json.load(f)

    outputs = []
    if args.claim:
        bm25_cands = bm25_map.get(args.claim) if bm25_map else None
        result = pl.run(args.claim, bm25_candidates=bm25_cands)
        outputs.append(result)
    elif args.claims:
        # Try to load as JSONL with fields: claim_id, claim
        path = Path(args.claims)
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    claim = obj.get("claim") or obj.get("text")
                    bm25_cands = bm25_map.get(claim) if bm25_map else None
                    outputs.append(pl.run(claim, bm25_candidates=bm25_cands))
        else:
            # Fallback simple CSV with a 'claim' column
            import pandas as pd

            df = pd.read_csv(path)
            for _, row in df.iterrows():
                claim = str(row["claim"]) if "claim" in df.columns else str(row[df.columns[0]])
                bm25_cands = bm25_map.get(claim) if bm25_map else None
                outputs.append(pl.run(claim, bm25_candidates=bm25_cands))
    else:
        raise SystemExit("Provide --claim or --claims")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for obj in outputs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
