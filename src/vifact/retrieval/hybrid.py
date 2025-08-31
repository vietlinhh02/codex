from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60, top_k: int = 100) -> List[str]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:top_k]]


def merge_rankings_with_scores(
    rankings: List[List[Tuple[str, float]]], normalize: bool = True, top_k: int = 100
) -> List[Tuple[str, float]]:
    # Optionally min-max normalize each ranking's scores, then sum.
    agg: Dict[str, float] = {}
    for ranked in rankings:
        if not ranked:
            continue
        if normalize:
            vals = [s for _, s in ranked]
            vmin, vmax = min(vals), max(vals)
            span = (vmax - vmin) if vmax > vmin else 1.0
            norm = [(d, (s - vmin) / span) for d, s in ranked]
        else:
            norm = ranked
        for doc_id, s in norm:
            agg[doc_id] = agg.get(doc_id, 0.0) + float(s)
    sorted_docs = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_docs

