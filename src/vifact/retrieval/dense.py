from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss may not be present in some envs
    faiss = None  # type: ignore


class DenseRetriever:
    def __init__(self, model_name: str, max_seq_length: int = 256, use_gpu: bool = False):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.use_gpu = use_gpu
        self.index = None
        self.doc_ids: List[str] = []

    def build_index(self, texts: List[str], doc_ids: List[str], use_faiss: bool = True):
        self.doc_ids = list(doc_ids)
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        if use_faiss and faiss is not None:
            dim = emb.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(emb.astype(np.float32))
            self.index = index
        else:
            self.index = emb  # fallback: brute-force cosine using numpy

    def save_faiss(self, path: str):
        if faiss is None or not isinstance(self.index, faiss.Index):
            raise ValueError("FAISS index not available")
        faiss.write_index(self.index, path)

    def load_faiss(self, path: str):
        if faiss is None:
            raise ValueError("FAISS not installed")
        self.index = faiss.read_index(path)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        if self.index is None:
            raise ValueError("Index not built")
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        if faiss is not None and isinstance(self.index, faiss.Index):
            D, I = self.index.search(q.astype(np.float32), top_k)
            scores = D[0].tolist()
            ids = [self.doc_ids[i] for i in I[0]]
            return list(zip(ids, scores))
        else:
            emb = self.index  # type: ignore
            sims = (emb @ q.T).reshape(-1)
            idx = np.argsort(-sims)[:top_k]
            return [(self.doc_ids[i], float(sims[i])) for i in idx]

