from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import pandas as pd
import numpy as np
from rapidfuzz.fuzz import partial_ratio
from rank_bm25 import BM25Okapi

# Pre-compile regex patterns for better performance
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKENIZE_PATTERN = re.compile(r"[\wÀ-ỹ]+")

# Cache for tokenized claims to avoid repeated tokenization
_tokenize_cache: Dict[str, List[str]] = {}


LABEL_MAP_IN = {
    "SUPPORTED": "SUPPORTED",
    "REFUTED": "REFUTED",
    "NEI": "INSUFFICIENT",
    "INSUFFICIENT": "INSUFFICIENT",
}


def split_paragraphs(text: str) -> List[str]:
    # Split by blank lines first using pre-compiled pattern
    paras = PARAGRAPH_SPLIT_PATTERN.split(text.strip())
    # Normalize whitespace using pre-compiled pattern
    paras = [WHITESPACE_PATTERN.sub(" ", p).strip() for p in paras if p.strip()]
    return paras


def chunk_text(text: str, max_tokens: int = 220) -> List[str]:
    # Simple whitespace token-based chunking - optimized
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    
    # Use list comprehension for better performance
    return [" ".join(tokens[i:i + max_tokens]) 
            for i in range(0, len(tokens), max_tokens)]


def tokenize_bm25(text: str) -> List[str]:
    # Use cache to avoid re-tokenizing the same text
    if text in _tokenize_cache:
        return _tokenize_cache[text]
    
    # Basic tokenization suitable for Vietnamese/Unicode words
    text_lower = text.lower()
    # Use pre-compiled pattern
    words = TOKENIZE_PATTERN.findall(text_lower)
    
    # Cache result if text is short enough (likely to be repeated)
    if len(text) < 1000:  # Cache only shorter texts
        _tokenize_cache[text] = words
    
    return words


def prepare_from_kvjson(
    in_path: str,
    out_dir: str,
    max_tokens_per_chunk: int = 220,
    topk_bm25: int = 20,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Converts your key->object JSON to:
    - corpus.csv (doc_id,text)
    - train.jsonl / valid.jsonl (for Reasoner/Explainer training)
    - bm25.json (claim -> ranked list of doc_id)
    """
    in_path = str(in_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict] = json.load(f)
    total_items = len(data)
    print(f"[prepare_kvjson] Loaded {total_items} items from {in_path}")

    # 1) Build corpus by chunking context into paragraphs/chunks - optimized
    corpus_rows: List[Tuple[str, str]] = []  # (doc_id, text)
    id_to_chunks: Dict[str, List[Tuple[str, str]]] = {}
    
    # Pre-allocate lists for better memory efficiency
    all_doc_ids = []
    all_texts = []
    
    for key, obj in data.items():
        ctx = obj.get("context") or ""
        if not ctx:
            # Handle empty context efficiently
            doc_id = f"{key}#p00"
            chunks = [(doc_id, "")]
            corpus_rows.append((doc_id, ""))
            all_doc_ids.append(doc_id)
            all_texts.append("")
        else:
            paras = split_paragraphs(ctx)
            chunks: List[Tuple[str, str]] = []
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
                # ensure at least one chunk (possibly empty)
                doc_id = f"{key}#p00"
                chunks = [(doc_id, ctx.strip())]
                corpus_rows.append((doc_id, ctx.strip()))
                all_doc_ids.append(doc_id)
                all_texts.append(ctx.strip())
                
        id_to_chunks[key] = chunks

    # Save corpus using pre-built lists
    corpus_df = pd.DataFrame({
        "doc_id": all_doc_ids,
        "text": all_texts
    })
    corpus_csv = out / "corpus.csv"
    print(f"[prepare_kvjson] Built corpus with {len(corpus_rows)} chunks; writing {corpus_csv}")
    corpus_df.to_csv(corpus_csv, index=False)
    # Fast lookup map - use dict comprehension for better performance
    doc_text_map: Dict[str, str] = dict(corpus_rows)

    # 2) Build BM25 over all chunks - optimized tokenization
    print(f"[prepare_kvjson] Tokenizing {len(all_texts)} chunks for BM25...")
    bm25_corpus_tokens = [tokenize_bm25(text) for text in all_texts]
    bm25 = BM25Okapi(bm25_corpus_tokens)
    print(f"[prepare_kvjson] Built BM25 over {len(corpus_rows)} chunks")

    # 3) Build train/valid JSONL with mapped gold evidence when possible - optimized
    records: List[Dict] = []
    
    # Pre-compile claim tokens to avoid repeated tokenization
    claim_tokens_cache: Dict[str, List[str]] = {}
    
    for idx, (key, obj) in enumerate(data.items(), start=1):
        claim = str(obj.get("claim", "")).strip()
        verdict = str(obj.get("verdict", "INSUFFICIENT")).strip().upper()
        label = LABEL_MAP_IN.get(verdict, "INSUFFICIENT")
        evidence_text = obj.get("evidence")

        # BM25 candidates for this claim - use cached tokenization
        if claim not in claim_tokens_cache:
            claim_tokens_cache[claim] = tokenize_bm25(claim)
        tokens = claim_tokens_cache[claim]
        
        scores = bm25.get_scores(tokens)
        # Use numpy-style operations for faster ranking
        ranked_idx = np.argpartition(scores, -topk_bm25)[-topk_bm25:]
        ranked_idx = ranked_idx[np.argsort(scores[ranked_idx])[::-1]]
        ranked_doc_ids = [all_doc_ids[i] for i in ranked_idx]

        # Map gold evidence (string) to best chunk by fuzzy match - optimized
        gold_ids: List[str] = []
        if evidence_text:
            chunks_for_key = id_to_chunks.get(key, [])
            if chunks_for_key:
                # Use vectorized approach for fuzzy matching
                chunk_texts = [text for _, text in chunks_for_key]
                chunk_ids = [did for did, _ in chunks_for_key]
                
                # Calculate all scores at once
                scores = [partial_ratio(evidence_text, text) for text in chunk_texts]
                if scores:
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    gold_ids = [chunk_ids[best_idx]]

        # Build evidences field (use text for all ranked ids) - optimized
        evidences = [{"doc_id": did, "text": doc_text_map[did]} for did in ranked_doc_ids]

        rec = {
            "claim_id": key,
            "claim": claim,
            "label": label,
            "evidences": evidences,
        }
        if gold_ids:
            rec["gold_evidence_ids"] = gold_ids
        records.append(rec)

        # Lightweight progress logging
        if idx % 500 == 0 or idx == total_items:
            print(f"[prepare_kvjson] Processed {idx}/{total_items} claims")

    # Train/valid split
    random.Random(seed).shuffle(records)
    n_valid = max(1, int(len(records) * valid_ratio))
    valid_recs = records[:n_valid]
    train_recs = records[n_valid:]

    def write_jsonl(path: Path, items: List[Dict]):
        # Batch write for better I/O performance
        lines = [json.dumps(obj, ensure_ascii=False) for obj in items]
        with open(path, "w", encoding="utf-8") as f:
            f.write('\n'.join(lines) + '\n')

    train_jsonl = out / "train.jsonl"
    valid_jsonl = out / "valid.jsonl"
    write_jsonl(train_jsonl, train_recs)
    write_jsonl(valid_jsonl, valid_recs)

    # 4) Save bm25.json mapping claim->ranked_doc_ids for quick hybrid fusion
    bm25_map = {rec["claim"]: [ev["doc_id"] for ev in rec["evidences"]] for rec in records}
    bm25_json = out / "bm25.json"
    with open(bm25_json, "w", encoding="utf-8") as f:
        json.dump(bm25_map, f, ensure_ascii=False)
    print(f"[prepare_kvjson] Wrote outputs to {out}")

    # Clear cache to free memory
    _tokenize_cache.clear()

    return {
        "corpus_csv": str(corpus_csv),
        "train_jsonl": str(train_jsonl),
        "valid_jsonl": str(valid_jsonl),
        "bm25_json": str(bm25_json),
    }
