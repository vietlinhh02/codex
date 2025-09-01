#!/usr/bin/env python
import argparse
import json
import time
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


def main():
    ap = argparse.ArgumentParser(description="Prepare corpus/train from ISE-DSC01-style KV JSON (with detailed logging)")
    ap.add_argument("--input", required=True, help="Path to KV JSON (id -> {context, claim, verdict, evidence, ...})")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_tokens_per_chunk", type=int, default=220)
    ap.add_argument("--topk_bm25", type=int, default=20)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 80)
    print(f"🚀 STARTING PROCESSING AT {time.strftime('%H:%M:%S %d/%m/%Y')}")
    print(f"📁 Input file: {args.input}")
    print(f"📂 Output dir: {args.out_dir}")
    print(f"⚙️  Config: max_tokens={args.max_tokens_per_chunk}, topk_bm25={args.topk_bm25}")
    print(f"💻 CPU Limit: Using optimized sequential processing (single core)")
    print("=" * 80)

    start_time = time.time()
    
    try:
        paths = prepare_from_kvjson(
            in_path=args.input,
            out_dir=args.out_dir,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            topk_bm25=args.topk_bm25,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        
        total_time = time.time() - start_time
        print("=" * 80)
        print(f"✅ SUCCESS! Completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print("📋 Output files:")
        for key, path in paths.items():
            print(f"   {key}: {path}")
        print("=" * 80)
        
    except Exception as e:
        total_time = time.time() - start_time
        print("=" * 80)
        print(f"❌ ERROR after {total_time:.1f} seconds: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
