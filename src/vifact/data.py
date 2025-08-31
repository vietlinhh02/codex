from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class CorpusDoc:
    doc_id: str
    text: str
    title: Optional[str] = None
    meta: Optional[Dict] = None


def load_corpus_from_csv(
    path: str, text_col: str = "text", id_col: str = "doc_id", title_col: Optional[str] = None
) -> List[CorpusDoc]:
    df = pd.read_csv(path)
    docs: List[CorpusDoc] = []
    for _, row in df.iterrows():
        doc_id = str(row[id_col])
        text = str(row[text_col])
        title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else None
        docs.append(CorpusDoc(doc_id=doc_id, text=text, title=title))
    return docs


def load_corpus_from_jsonl(
    path: str,
    text_field: str = "text",
    id_field: str = "doc_id",
    title_field: Optional[str] = None,
) -> List[CorpusDoc]:
    docs: List[CorpusDoc] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj[id_field])
            text = str(obj[text_field])
            title = str(obj[title_field]) if title_field and obj.get(title_field) else None
            docs.append(CorpusDoc(doc_id=doc_id, text=text, title=title, meta=obj))
    return docs


def load_claims_from_csv(
    path: str,
    claim_col: str = "claim",
    id_col: str = "claim_id",
    label_col: Optional[str] = None,
    evidence_col: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep_cols = [c for c in [id_col, claim_col, label_col, evidence_col] if c]
    return df[keep_cols].copy()


def load_claims_from_jsonl(
    path: str,
    claim_field: str = "claim",
    id_field: str = "claim_id",
    label_field: Optional[str] = None,
    evidence_field: Optional[str] = None,
) -> pd.DataFrame:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rec = {
                id_field: obj.get(id_field),
                claim_field: obj.get(claim_field),
            }
            if label_field:
                rec[label_field] = obj.get(label_field)
            if evidence_field:
                rec[evidence_field] = obj.get(evidence_field)
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    return df


def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter robust for Vietnamese and punctuation.
    import re

    # Split by period/question/exclamation, keep punctuation
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

