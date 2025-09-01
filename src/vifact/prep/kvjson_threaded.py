from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from rapidfuzz.fuzz import partial_ratio
from rank_bm25 import BM25Okapi

# Pre-compile regex patterns for better performance
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKENIZE_PATTERN = re.compile(r"[\wÀ-ỹ]+")

LABEL_MAP_IN = {
    "SUPPORTED": "SUPPORTED",
    "REFUTED": "REFUTED",
    "NEI": "INSUFFICIENT",
    "INSUFFICIENT": "INSUFFICIENT",
}


def split_paragraphs(text: str) -> List[str]:
    paras = PARAGRAPH_SPLIT_PATTERN.split(text.strip())
    paras = [WHITESPACE_PATTERN.sub(" ", p).strip() for p in paras if p.strip()]
    return paras


def chunk_text(text: str, max_tokens: int = 220) -> List[str]:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    return [" ".join(tokens[i:i + max_tokens]) 
            for i in range(0, len(tokens), max_tokens)]


def tokenize_bm25(text: str) -> List[str]:
    text_lower = text.lower()
    words = TOKENIZE_PATTERN.findall(text_lower)
    return words


def process_claim_batch(args):
    """Process a batch of claims using threading"""
    claims_batch, id_to_chunks, bm25, all_doc_ids, doc_text_map, topk_bm25 = args
    
    results = []
    for idx, (key, obj) in claims_batch:
        claim = str(obj.get("claim", "")).strip()
        verdict = str(obj.get("verdict", "INSUFFICIENT")).strip().upper()
        label = LABEL_MAP_IN.get(verdict, "INSUFFICIENT")
        evidence_text = obj.get("evidence")

        # BM25 candidates
        tokens = tokenize_bm25(claim)
        scores = bm25.get_scores(tokens)
        ranked_idx = np.argpartition(scores, -topk_bm25)[-topk_bm25:]
        ranked_idx = ranked_idx[np.argsort(scores[ranked_idx])[::-1]]
        ranked_doc_ids = [all_doc_ids[i] for i in ranked_idx]

        # Gold evidence matching
        gold_ids = []
        if evidence_text and key in id_to_chunks:
            chunks_for_key = id_to_chunks[key]
            if chunks_for_key:
                chunk_texts = [text for _, text in chunks_for_key]
                chunk_ids = [did for did, _ in chunks_for_key]
                scores = [partial_ratio(evidence_text, text) for text in chunk_texts]
                if scores:
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    gold_ids = [chunk_ids[best_idx]]

        evidences = [{"doc_id": did, "text": doc_text_map[did]} for did in ranked_doc_ids]

        rec = {
            "claim_id": key,
            "claim": claim,
            "label": label,
            "evidences": evidences,
        }
        if gold_ids:
            rec["gold_evidence_ids"] = gold_ids
        
        results.append(rec)
    
    return results


def prepare_from_kvjson_threaded(
    in_path: str,
    out_dir: str,
    max_tokens_per_chunk: int = 220,
    topk_bm25: int = 20,
    valid_ratio: float = 0.1,
    seed: int = 42,
    n_threads: Optional[int] = None,
) -> Dict[str, str]:
    """
    Threaded version of prepare_from_kvjson - uses threading instead of multiprocessing
    to avoid pickle issues while still providing parallelization for I/O bound operations
    """
    if n_threads is None:
        import multiprocessing as mp
        n_threads = min(mp.cpu_count(), 8)  # Cap at 8
    
    in_path = str(in_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict] = json.load(f)
    total_items = len(data)
    print(f"[prepare_kvjson_threaded] Loaded {total_items} items from {in_path}")
    print(f"[prepare_kvjson_threaded] Using {n_threads} threads")

    # 1) Build corpus by chunking - sequential for simplicity and memory efficiency
    print("[prepare_kvjson_threaded] Processing documents...")
    corpus_rows = []
    id_to_chunks = {}
    all_doc_ids = []
    all_texts = []
    
    for idx, (key, obj) in enumerate(data.items(), 1):
        ctx = obj.get("context") or ""
        if not ctx:
            doc_id = f"{key}#p00"
            chunks = [(doc_id, "")]
            corpus_rows.append((doc_id, ""))
            all_doc_ids.append(doc_id)
            all_texts.append("")
        else:
            paras = split_paragraphs(ctx)
            chunks = []
            pidx = 0
            for p in paras:
                p_chunks = chunk_text(p, max_tokens=max_tokens_per_chunk)
                for ch in p_chunks:
                    doc_id = f"{key}#p{pidx:02d}"
                    chunks.append((doc_id, ch))
                    corpus_rows.append((doc_id, ch))
                    all_doc_ids.append(doc_id)
                    all_texts.append(ch)
                    pidx += 1
            
            if not chunks:
                doc_id = f"{key}#p00"
                chunks = [(doc_id, ctx.strip())]
                corpus_rows.append((doc_id, ctx.strip()))
                all_doc_ids.append(doc_id)
                all_texts.append(ctx.strip())
                
        id_to_chunks[key] = chunks
        
        if idx % 1000 == 0:
            print(f"[prepare_kvjson_threaded] Processed {idx}/{total_items} documents")

    # Save corpus
    corpus_df = pd.DataFrame({
        "doc_id": all_doc_ids,
        "text": all_texts
    })
    corpus_csv = out / "corpus.csv"
    print(f"[prepare_kvjson_threaded] Built corpus with {len(corpus_rows)} chunks; writing {corpus_csv}")
    corpus_df.to_csv(corpus_csv, index=False)
    doc_text_map = dict(corpus_rows)

    # 2) Build BM25
    print(f"[prepare_kvjson_threaded] Building BM25 over {len(all_texts)} chunks...")
    bm25_corpus_tokens = [tokenize_bm25(text) for text in all_texts]
    bm25 = BM25Okapi(bm25_corpus_tokens)

    # 3) Process claims with threading
    print("[prepare_kvjson_threaded] Processing claims with threading...")
    claims_items = list(enumerate(data.items(), start=1))
    
    # Split data for threading
    chunk_size = max(20, len(claims_items) // (n_threads * 2))
    records = []
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        
        for i in range(0, len(claims_items), chunk_size):
            chunk = claims_items[i:i + chunk_size]
            
            future = executor.submit(
                process_claim_batch,
                (chunk, id_to_chunks, bm25, all_doc_ids, doc_text_map, topk_bm25)
            )
            futures.append(future)
        
        # Collect results
        processed_count = 0
        for future in as_completed(futures):
            chunk_records = future.result()
            records.extend(chunk_records)
            processed_count += len(chunk_records)
            if processed_count % 500 == 0:
                print(f"[prepare_kvjson_threaded] Processed {processed_count}/{total_items} claims")

    print(f"[prepare_kvjson_threaded] Finished processing {len(records)} claims")

    # 4) Train/valid split and save
    random.Random(seed).shuffle(records)
    n_valid = max(1, int(len(records) * valid_ratio))
    valid_recs = records[:n_valid]
    train_recs = records[n_valid:]

    def write_jsonl(path: Path, items: List[Dict]):
        lines = [json.dumps(obj, ensure_ascii=False) for obj in items]
        with open(path, "w", encoding="utf-8") as f:
            f.write('\n'.join(lines) + '\n')

    train_jsonl = out / "train.jsonl"
    valid_jsonl = out / "valid.jsonl"
    write_jsonl(train_jsonl, train_recs)
    write_jsonl(valid_jsonl, valid_recs)

    # 5) Save bm25.json
    bm25_map = {rec["claim"]: [ev["doc_id"] for ev in rec["evidences"]] for rec in records}
    bm25_json = out / "bm25.json"
    with open(bm25_json, "w", encoding="utf-8") as f:
        json.dump(bm25_map, f, ensure_ascii=False)
    
    print(f"[prepare_kvjson_threaded] Wrote outputs to {out}")

    return {
        "corpus_csv": str(corpus_csv),
        "train_jsonl": str(train_jsonl),
        "valid_jsonl": str(valid_jsonl),
        "bm25_json": str(bm25_json),
    }
