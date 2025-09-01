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

from vifact.prep.kvjson import prepare_from_kvjson
from vifact.prep.kvjson_parallel import prepare_from_kvjson_parallel


def main():
    ap = argparse.ArgumentParser(description="Prepare corpus/train from ISE-DSC01-style KV JSON (optimized version)")
    ap.add_argument("--input", required=True, help="Path to KV JSON (id -> {context, claim, verdict, evidence, ...})")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_tokens_per_chunk", type=int, default=220)
    ap.add_argument("--topk_bm25", type=int, default=20)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parallel", action="store_true", help="Use parallel processing")
    ap.add_argument("--n_workers", type=int, help="Number of workers for parallel processing")
    args = ap.parse_args()

    if args.parallel:
        print("Using parallel processing...")
        paths = prepare_from_kvjson_parallel(
            in_path=args.input,
            out_dir=args.out_dir,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            topk_bm25=args.topk_bm25,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            n_workers=args.n_workers,
        )
    else:
        print("Using optimized sequential processing...")
        paths = prepare_from_kvjson(
            in_path=args.input,
            out_dir=args.out_dir,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            topk_bm25=args.topk_bm25,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )

    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
