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
from vifact.prep.kvjson_ultra_fast import prepare_from_kvjson_ultra_fast, prepare_from_kvjson_gpu


def main():
    ap = argparse.ArgumentParser(description="🚀 ULTRA-FAST corpus preparation with pre-filtering")
    ap.add_argument("--input", required=True, help="Path to KV JSON")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_tokens_per_chunk", type=int, default=220)
    ap.add_argument("--topk_bm25", type=int, default=20)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["sequential", "ultra", "gpu"], default="ultra",
                    help="Processing mode: sequential, ultra (pre-filter), gpu")
    ap.add_argument("--n_workers", type=int, default=6, help="Number of processes (max: 6)")
    ap.add_argument("--max_candidates", type=int, default=5000, 
                    help="Max candidates per claim (lower = faster, default: 5000)")
    args = ap.parse_args()

    print("=" * 100)
    print(f"🚀 ULTRA-FAST CORPUS PREPARATION")
    print(f"⏰ Started: {time.strftime('%H:%M:%S %d/%m/%Y')}")
    print(f"📁 Input: {args.input}")
    print(f"📂 Output: {args.out_dir}")
    print(f"⚙️  Config: max_tokens={args.max_tokens_per_chunk}, topk_bm25={args.topk_bm25}")
    print(f"🎯 Mode: {args.mode.upper()}")
    
    if args.mode in ["ultra", "gpu"]:
        print(f"💻 Processes: {min(args.n_workers, 6)}")
        print(f"🔍 Search optimization: Max {args.max_candidates} candidates per claim")
        print(f"⚡ Expected speedup: {300000//args.max_candidates}x faster than full search")
    
    print("=" * 100)

    start_time = time.time()
    
    try:
        if args.mode == "gpu":
            paths = prepare_from_kvjson_gpu(
                in_path=args.input,
                out_dir=args.out_dir,
                max_tokens_per_chunk=args.max_tokens_per_chunk,
                topk_bm25=args.topk_bm25,
                valid_ratio=args.valid_ratio,
                seed=args.seed,
            )
        elif args.mode == "ultra":
            paths = prepare_from_kvjson_ultra_fast(
                in_path=args.input,
                out_dir=args.out_dir,
                max_tokens_per_chunk=args.max_tokens_per_chunk,
                topk_bm25=args.topk_bm25,
                valid_ratio=args.valid_ratio,
                seed=args.seed,
                n_workers=min(args.n_workers, 6),
                max_prefilter_candidates=args.max_candidates,
            )
        else:  # sequential
            paths = prepare_from_kvjson(
                in_path=args.input,
                out_dir=args.out_dir,
                max_tokens_per_chunk=args.max_tokens_per_chunk,
                topk_bm25=args.topk_bm25,
                valid_ratio=args.valid_ratio,
                seed=args.seed,
            )
        
        total_time = time.time() - start_time
        print("=" * 100)
        print(f"🎉 SUCCESS! Completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Calculate performance safely
        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            num_items = len(input_data)
            
        print(f"⚡ Performance: {(num_items / (total_time/60)):.0f} claims/minute")
        print("📋 Output files:")
        for key, path in paths.items():
            file_size = Path(path).stat().st_size / (1024*1024)  # MB
            print(f"   📄 {key}: {Path(path).name} ({file_size:.1f} MB)")
        print("=" * 100)
        
    except Exception as e:
        total_time = time.time() - start_time
        print("=" * 100)
        print(f"❌ ERROR after {total_time:.1f} seconds:")
        print(f"   {type(e).__name__}: {e}")
        print("=" * 100)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
