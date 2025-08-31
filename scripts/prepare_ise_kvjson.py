#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from vifact.prep.kvjson import prepare_from_kvjson


def main():
    ap = argparse.ArgumentParser(description="Prepare corpus/train from ISE-DSC01-style KV JSON")
    ap.add_argument("--input", required=True, help="Path to KV JSON (id -> {context, claim, verdict, evidence, ...})")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_tokens_per_chunk", type=int, default=220)
    ap.add_argument("--topk_bm25", type=int, default=20)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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

