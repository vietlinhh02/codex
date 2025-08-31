from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz.fuzz import partial_ratio
from rank_bm25 import BM25Okapi


LABEL_MAP_IN = {
    "SUPPORTED": "SUPPORTED",
    "REFUTED": "REFUTED",
    "NEI": "INSUFFICIENT",
    "INSUFFICIENT": "INSUFFICIENT",
}


def split_paragraphs(text: str) -> List[str]:
    # Split by blank lines first
    paras = re.split(r"\n\s*\n+", text.strip())
    # Normalize whitespace
    paras = [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]
    return paras


def chunk_text(text: str, max_tokens: int = 220) -> List[str]:
    # Simple whitespace token-based chunking
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = " ".join(tokens[i : i + max_tokens])
        if chunk:
            chunks.append(chunk)
    return chunks


def tokenize_bm25(text: str) -> List[str]:
    # Basic tokenization suitable for Vietnamese/Unicode words
    text = text.lower()
    # Split on non-letter/digit characters; keep Vietnamese letters
    words = re.findall(r"[\wÀ-ỹ]+", text)
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

    # 1) Build corpus by chunking context into paragraphs/chunks
    corpus_rows: List[Tuple[str, str]] = []  # (doc_id, text)
    id_to_chunks: Dict[str, List[Tuple[str, str]]] = {}
    for key, obj in data.items():
        ctx = obj.get("context") or ""
        paras = split_paragraphs(ctx) if ctx else []
        chunks: List[Tuple[str, str]] = []
        pidx = 0
        for p in paras:
            for ch in chunk_text(p, max_tokens=max_tokens_per_chunk):
                doc_id = f"{key}#p{pidx:02d}"
                chunks.append((doc_id, ch))
                corpus_rows.append((doc_id, ch))
                pidx += 1
        if not chunks:
            # ensure at least one chunk (possibly empty)
            doc_id = f"{key}#p00"
            chunks = [(doc_id, ctx.strip())]
            corpus_rows.append((doc_id, ctx.strip()))
        id_to_chunks[key] = chunks

    # Save corpus
    corpus_df = pd.DataFrame(corpus_rows, columns=["doc_id", "text"])
    corpus_csv = out / "corpus.csv"
    corpus_df.to_csv(corpus_csv, index=False)

    # 2) Build BM25 over all chunks
    bm25_corpus_tokens = [tokenize_bm25(t) for _, t in corpus_rows]
    bm25 = BM25Okapi(bm25_corpus_tokens)

    # 3) Build train/valid JSONL with mapped gold evidence when possible
    records: List[Dict] = []
    for key, obj in data.items():
        claim = str(obj.get("claim", "")).strip()
        verdict = str(obj.get("verdict", "INSUFFICIENT")).strip().upper()
        label = LABEL_MAP_IN.get(verdict, "INSUFFICIENT")
        evidence_text = obj.get("evidence")

        # BM25 candidates for this claim
        tokens = tokenize_bm25(claim)
        scores = bm25.get_scores(tokens)
        # rank doc_ids by score
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk_bm25]
        ranked_doc_ids = [corpus_rows[i][0] for i in ranked_idx]

        # Map gold evidence (string) to best chunk by fuzzy match
        gold_ids: List[str] = []
        if evidence_text:
            best_doc = None
            best_score = -1
            # Restrict search to chunks belonging to this sample id (key) for higher precision
            for did, text in id_to_chunks.get(key, []):
                s = partial_ratio(evidence_text, text)
                if s > best_score:
                    best_score = s
                    best_doc = did
            if best_doc is not None:
                gold_ids = [best_doc]

        # Build evidences field (use text for all ranked ids)
        evidences = [{"doc_id": did, "text": corpus_df.loc[corpus_df.doc_id == did, "text"].values[0]} for did in ranked_doc_ids]

        rec = {
            "claim_id": key,
            "claim": claim,
            "label": label,
            "evidences": evidences,
        }
        if gold_ids:
            rec["gold_evidence_ids"] = gold_ids
        records.append(rec)

    # Train/valid split
    import random

    random.Random(seed).shuffle(records)
    n_valid = max(1, int(len(records) * valid_ratio))
    valid_recs = records[:n_valid]
    train_recs = records[n_valid:]

    def write_jsonl(path: Path, items: List[Dict]):
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    train_jsonl = out / "train.jsonl"
    valid_jsonl = out / "valid.jsonl"
    write_jsonl(train_jsonl, train_recs)
    write_jsonl(valid_jsonl, valid_recs)

    # 4) Save bm25.json mapping claim->ranked_doc_ids for quick hybrid fusion
    bm25_map = {rec["claim"]: [ev["doc_id"] for ev in rec["evidences"]] for rec in records}
    bm25_json = out / "bm25.json"
    with open(bm25_json, "w", encoding="utf-8") as f:
        json.dump(bm25_map, f, ensure_ascii=False)

    return {
        "corpus_csv": str(corpus_csv),
        "train_jsonl": str(train_jsonl),
        "valid_jsonl": str(valid_jsonl),
        "bm25_json": str(bm25_json),
    }

