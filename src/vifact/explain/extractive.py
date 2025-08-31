from __future__ import annotations

from typing import List

from rapidfuzz.string_metric import jaro_winkler_similarity

from ..data import split_sentences


def extractive_explanation(claim: str, evidence_texts: List[str], max_sentences: int = 2) -> str:
    # Score sentences by similarity to claim, pick top-k unique sentences.
    scored: List[tuple[str, float]] = []
    for ev in evidence_texts:
        for s in split_sentences(ev):
            if len(s) < 5:
                continue
            score = jaro_winkler_similarity(claim, s)
            scored.append((s, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked: List[str] = []
    seen = set()
    for s, _ in scored:
        if s in seen:
            continue
        picked.append(s)
        seen.add(s)
        if len(picked) >= max_sentences:
            break
    if not picked:
        return ""
    return " ".join(picked)

