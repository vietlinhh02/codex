from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import threading

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

# Global progress tracking
_global_progress = {"processed": 0, "total": 0, "start_time": 0}
_progress_lock = threading.Lock()


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


# Standalone function for multiprocessing - must be at module level
def process_documents_batch(batch_data):
    """Process a batch of documents for chunking"""
    batch, max_tokens_per_chunk = batch_data
    results = {}
    corpus_batch = []
    
    for key, obj in batch:
        ctx = obj.get("context") or ""
        if not ctx:
            doc_id = f"{key}#p00"
            chunks = [(doc_id, "")]
        else:
            paras = split_paragraphs(ctx)
            chunks = []
            pidx = 0
            for p in paras:
                p_chunks = chunk_text(p, max_tokens=max_tokens_per_chunk)
                for ch in p_chunks:
                    doc_id = f"{key}#p{pidx:02d}"
                    chunks.append((doc_id, ch))
                    pidx += 1
            
            if not chunks:
                doc_id = f"{key}#p00"
                chunks = [(doc_id, ctx.strip())]
        
        results[key] = chunks
        corpus_batch.extend(chunks)
    
    return results, corpus_batch


# Standalone function for multiprocessing with progress reporting
def process_claims_batch_with_progress(batch_data):
    """Process a batch of claims for evidence retrieval with progress"""
    batch_id, claims_batch, id_to_chunks_subset, bm25_tokens_data, doc_info, topk_bm25 = batch_data
    all_doc_ids, doc_text_map = doc_info
    
    # Rebuild BM25 in this process
    bm25 = BM25Okapi(bm25_tokens_data)
    
    results = []
    batch_start = time.time()
    
    for local_idx, (idx, (key, obj)) in enumerate(claims_batch):
        claim = str(obj.get("claim", "")).strip()
        verdict = str(obj.get("verdict", "INSUFFICIENT")).strip().upper()
        label = LABEL_MAP_IN.get(verdict, "INSUFFICIENT")
        evidence_text = obj.get("evidence")

        # BM25 candidates - this is the slow part
        tokens = tokenize_bm25(claim)
        scores = bm25.get_scores(tokens)
        
        # Use numpy for fast ranking
        if len(scores) >= topk_bm25:
            ranked_idx = np.argpartition(scores, -topk_bm25)[-topk_bm25:]
            ranked_idx = ranked_idx[np.argsort(scores[ranked_idx])[::-1]]
        else:
            ranked_idx = np.argsort(scores)[::-1]
            
        ranked_doc_ids = [all_doc_ids[i] for i in ranked_idx]

        # Gold evidence matching
        gold_ids = []
        if evidence_text and key in id_to_chunks_subset:
            chunks_for_key = id_to_chunks_subset[key]
            if chunks_for_key:
                chunk_texts = [text for _, text in chunks_for_key]
                chunk_ids = [did for did, _ in chunks_for_key]
                scores_fuzzy = [partial_ratio(evidence_text, text) for text in chunk_texts]
                if scores_fuzzy:
                    best_idx = max(range(len(scores_fuzzy)), key=lambda i: scores_fuzzy[i])
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
        
        # Log every 10 claims within batch
        if (local_idx + 1) % 10 == 0 or (local_idx + 1) == len(claims_batch):
            elapsed = time.time() - batch_start
            rate = (local_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"[Worker-{batch_id}] Processed {local_idx+1}/{len(claims_batch)} claims in batch - {rate:.1f} claims/sec")
    
    batch_time = time.time() - batch_start
    print(f"[Worker-{batch_id}] 🎉 Batch completed in {batch_time:.1f}s - {len(results)} claims processed")
    
    return results


def prepare_from_kvjson_multiprocessing_v2(
    in_path: str,
    out_dir: str,
    max_tokens_per_chunk: int = 220,
    topk_bm25: int = 20,
    valid_ratio: float = 0.1,
    seed: int = 42,
    n_workers: Optional[int] = None,
) -> Dict[str, str]:
    """
    Enhanced multiprocessing version with detailed progress tracking
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count() - 2, 6)  # Leave 2 cores free, max 6
    else:
        n_workers = min(n_workers, 6)  # Force max 6 as requested
    
    start_time = time.time()
    in_path = str(in_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_v2] Starting at {time.strftime('%H:%M:%S')}")
    print(f"[prepare_v2] Using {n_workers} processes (CPU limit: 6)")
    
    with open(in_path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict] = json.load(f)
    total_items = len(data)
    print(f"[prepare_v2] Loaded {total_items} items")
    print(f"[prepare_v2] Config: max_tokens={max_tokens_per_chunk}, topk_bm25={topk_bm25}")

    # Step 1: Process documents in parallel
    print(f"[prepare_v2] Step 1/4: Processing {total_items} documents...")
    step1_start = time.time()
    
    # Split data into batches
    batch_size = max(50, total_items // (n_workers * 3))
    data_items = list(data.items())
    batches = []
    
    for i in range(0, len(data_items), batch_size):
        batch = data_items[i:i + batch_size]
        batches.append((batch, max_tokens_per_chunk))
    
    print(f"[prepare_v2] Created {len(batches)} batches of ~{batch_size} documents each")
    
    id_to_chunks = {}
    corpus_rows = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_documents_batch, batch): i for i, batch in enumerate(batches)}
        
        completed = 0
        for future in as_completed(futures):
            batch_idx = futures[future]
            batch_id_to_chunks, batch_corpus = future.result()
            id_to_chunks.update(batch_id_to_chunks)
            corpus_rows.extend(batch_corpus)
            
            completed += 1
            elapsed = time.time() - step1_start
            print(f"[prepare_v2] Doc batch {completed}/{len(batches)} completed - {elapsed:.1f}s elapsed")
    
    all_doc_ids = [did for did, _ in corpus_rows]
    all_texts = [text for _, text in corpus_rows]
    
    step1_time = time.time() - step1_start
    print(f"[prepare_v2] Step 1 completed in {step1_time:.1f}s - Created {len(corpus_rows)} chunks")

    # Save corpus
    print(f"[prepare_v2] Saving corpus to CSV...")
    corpus_df = pd.DataFrame({
        "doc_id": all_doc_ids,
        "text": all_texts
    })
    corpus_csv = out / "corpus.csv"
    corpus_df.to_csv(corpus_csv, index=False)
    doc_text_map = dict(corpus_rows)
    print(f"[prepare_v2] Corpus saved: {len(corpus_rows)} chunks")

    # Step 2: Build BM25
    print(f"[prepare_v2] Step 2/4: Building BM25 index...")
    step2_start = time.time()
    
    print(f"[prepare_v2] Tokenizing {len(all_texts)} chunks...")
    bm25_corpus_tokens = []
    for i, text in enumerate(all_texts):
        tokens = tokenize_bm25(text)
        bm25_corpus_tokens.append(tokens)
        if (i + 1) % 10000 == 0 or (i + 1) == len(all_texts):
            elapsed = time.time() - step2_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(all_texts) - (i + 1)) / rate if rate > 0 else 0
            print(f"[prepare_v2] Tokenized {i+1}/{len(all_texts)} ({(i+1)/len(all_texts)*100:.1f}%) - "
                  f"{rate:.0f} chunks/sec - ETA: {eta:.0f}s")
    
    print(f"[prepare_v2] Creating BM25 index...")
    bm25 = BM25Okapi(bm25_corpus_tokens)
    step2_time = time.time() - step2_start
    print(f"[prepare_v2] Step 2 completed in {step2_time:.1f}s")

    # Step 3: Process claims in parallel with detailed progress
    print(f"[prepare_v2] Step 3/4: Processing {total_items} claims (THIS IS THE HEAVY STEP)...")
    print(f"[prepare_v2] Expected time: ~{(total_items * len(all_texts) / 1_000_000):.1f} minutes for {total_items} × {len(all_texts)} operations")
    step3_start = time.time()
    
    claims_items = list(enumerate(data.items(), start=1))
    claim_batch_size = max(10, len(claims_items) // (n_workers * 4))  # Smaller batches for better progress tracking
    
    # Prepare batches for claims processing
    claim_batches = []
    for i, start_idx in enumerate(range(0, len(claims_items), claim_batch_size)):
        batch = claims_items[start_idx:start_idx + claim_batch_size]
        # Create subset of id_to_chunks for this batch
        batch_keys = {key for _, (key, _) in batch}
        id_to_chunks_subset = {k: v for k, v in id_to_chunks.items() if k in batch_keys}
        
        claim_batches.append((
            i,  # batch_id for logging
            batch, 
            id_to_chunks_subset, 
            bm25_corpus_tokens,  # Pass tokens instead of BM25 object
            (all_doc_ids, doc_text_map),
            topk_bm25
        ))
    
    print(f"[prepare_v2] Created {len(claim_batches)} claim batches of ~{claim_batch_size} claims each")
    print(f"[prepare_v2] 🕐 Step 3 will take the longest - processing {total_items} claims against {len(all_texts)} chunks...")
    
    records = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_claims_batch_with_progress, batch): i for i, batch in enumerate(claim_batches)}
        
        processed_batches = 0
        processed_claims = 0
        
        for future in as_completed(futures):
            batch_idx = futures[future]
            batch_records = future.result()
            records.extend(batch_records)
            
            processed_batches += 1
            processed_claims += len(batch_records)
            elapsed = time.time() - step3_start
            rate = processed_claims / elapsed if elapsed > 0 else 0
            eta = (total_items - processed_claims) / rate if rate > 0 else 0
            
            print(f"[prepare_v2] ✅ Claim batch {processed_batches}/{len(claim_batches)} DONE - "
                  f"Total claims: {processed_claims}/{total_items} ({processed_claims/total_items*100:.1f}%) - "
                  f"⚡ {rate:.1f} claims/sec - ⏰ ETA: {eta/60:.1f} minutes")

    step3_time = time.time() - step3_start
    print(f"[prepare_v2] Step 3 completed in {step3_time:.1f}s ({step3_time/60:.1f} minutes)")

    # Step 4: Save outputs
    print(f"[prepare_v2] Step 4/4: Saving outputs...")
    step4_start = time.time()
    
    # Train/valid split
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
    print(f"[prepare_v2] Writing {len(train_recs)} train records...")
    write_jsonl(train_jsonl, train_recs)
    print(f"[prepare_v2] Writing {len(valid_recs)} validation records...")
    write_jsonl(valid_jsonl, valid_recs)

    # Save bm25.json
    print(f"[prepare_v2] Saving BM25 mappings...")
    bm25_map = {rec["claim"]: [ev["doc_id"] for ev in rec["evidences"]] for rec in records}
    bm25_json = out / "bm25.json"
    with open(bm25_json, "w", encoding="utf-8") as f:
        json.dump(bm25_map, f, ensure_ascii=False)
    
    step4_time = time.time() - step4_start
    total_time = time.time() - start_time
    
    print(f"[prepare_v2] Step 4 completed in {step4_time:.1f}s")
    print(f"[prepare_v2] 🎉 ALL DONE! Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"[prepare_v2] Performance: {total_items/(total_time/60):.0f} claims/minute")
    print(f"[prepare_v2] Wrote outputs to {out}")

    return {
        "corpus_csv": str(corpus_csv),
        "train_jsonl": str(train_jsonl),
        "valid_jsonl": str(valid_jsonl),
        "bm25_json": str(bm25_json),
    }
