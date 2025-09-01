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
    ap = argparse.ArgumentParser(description="Retrieve hybrid evidences per-claim and save JSONL files")
    ap.add_argument("--corpus", required=True, help="Path to CSV or JSONL corpus")
    ap.add_argument("--format", choices=["csv", "jsonl"], required=True)
    ap.add_argument("--id-col", default="doc_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--title-col", default=None)
    ap.add_argument("--claims", required=True, help="Path to JSONL with {claim_id, claim}")
    ap.add_argument("--bm25", required=False, help="Optional path to bm25.json mapping claim -> [doc_id]")
    ap.add_argument("--out_dir", required=True, help="Output dir; creates <claim_id>.jsonl with evidences")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--reasoner_ckpt", default=None, help="Optional: use reasoner for label/explain preview")
    args = ap.parse_args()

    # Load corpus
    if args.format == "csv":
        docs = load_corpus_from_csv(args.corpus, text_col=args.text_col, id_col=args.id_col, title_col=args.title_col)
    else:
        docs = load_corpus_from_jsonl(args.corpus, text_field=args.text_col, id_field=args.id_col, title_field=args.title_col)

    cfg = PipelineConfig()
    cfg.rerank.top_k = args.topk
    pipe = ViFactPipeline(cfg, reasoner_ckpt=args.reasoner_ckpt)
    pipe.build_dense_index(docs)

    bm25_map = None
    if args.bm25:
        with open(args.bm25, "r", encoding="utf-8") as f:
            bm25_map = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read claims
    claims = []
    with open(args.claims, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = str(obj.get("claim_id", obj.get("id")))
            ctext = obj.get("claim") or obj.get("text")
            if not cid or not ctext:
                continue
            claims.append((cid, ctext))

    # Process each claim
    for cid, ctext in claims:
        bm25_cands = bm25_map.get(ctext) if bm25_map else None
        result = pipe.run(ctext, bm25_candidates=bm25_cands)
        # Build evidences file for this claim
        ev_ids = [d for d, _ in result["reranked"][: args.topk]]
        evidences = []
        for did in ev_ids:
            if did in pipe.corpus:
                evidences.append({"doc_id": did, "text": pipe.corpus[did].text})
        # Write <claim_id>.jsonl
        with open(out_dir / f"{cid}.jsonl", "w", encoding="utf-8") as f:
            for ev in evidences:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        # Write preview result
        with open(out_dir / f"{cid}.result.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"Saved evidences to {out_dir} (one file per claim_id)")


if __name__ == "__main__":
    main()
