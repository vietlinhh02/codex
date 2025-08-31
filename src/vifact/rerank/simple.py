from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util


class SimpleReranker:
    def __init__(self, cross_encoder_model: str | None = None):
        self.ce = CrossEncoder(cross_encoder_model) if cross_encoder_model else None
        self.be = None if self.ce else SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def rerank(self, claim: str, candidates: List[Tuple[str, str]], top_k: int = 20) -> List[Tuple[str, float]]:
        # candidates: List of (doc_id, text)
        if self.ce is not None:
            pairs = [(claim, text) for _, text in candidates]
            scores = self.ce.predict(pairs).tolist()
            scored = list(zip([doc_id for doc_id, _ in candidates], scores))
        else:
            queries = self.be.encode([claim], normalize_embeddings=True)
            docs = self.be.encode([t for _, t in candidates], normalize_embeddings=True)
            sims = (docs @ queries.T).reshape(-1).tolist()
            scored = list(zip([doc_id for doc_id, _ in candidates], sims))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

