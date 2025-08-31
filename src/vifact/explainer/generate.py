from __future__ import annotations

from typing import Dict, List, Tuple

from ..data import split_sentences


def assemble_explanation(
    claim: str,
    evidences: List[Dict],
    evidence_attn: List[float] | None = None,
    label: str | None = None,
    counter_flags: List[bool] | None = None,
    max_support: int = 2,
    max_counter: int = 1,
) -> str:
    # Rank evidences by attention if available, else by order
    idxs = list(range(len(evidences)))
    if evidence_attn is not None and len(evidence_attn) == len(evidences):
        idxs.sort(key=lambda i: evidence_attn[i], reverse=True)
    # Collect supporting sentences
    support_parts: List[str] = []
    support_cites: List[str] = []
    for i in idxs:
        doc = evidences[i]
        did = str(doc.get("doc_id", i))
        best = split_sentences(doc["text"])[:1]
        if best:
            support_parts.append(best[0])
            support_cites.append(f"[{did}]")
        if len(support_parts) >= max_support:
            break
    # Optional counter-evidence sentences
    counter_parts: List[str] = []
    counter_cites: List[str] = []
    if counter_flags is not None and any(counter_flags):
        for i in idxs:
            if not counter_flags[i]:
                continue
            doc = evidences[i]
            did = str(doc.get("doc_id", i))
            best = split_sentences(doc["text"])[:1]
            if best:
                counter_parts.append(best[0])
                counter_cites.append(f"[{did}]")
            if len(counter_parts) >= max_counter:
                break

    # Compose explanation
    lines: List[str] = []
    if label == "SUPPORTED":
        lines.append("Kết luận: ĐÚNG.")
    elif label == "REFUTED":
        lines.append("Kết luận: SAI.")
    else:
        lines.append("Kết luận: KHÔNG ĐỦ THÔNG TIN.")
    if support_parts:
        body = " ".join(support_parts)
        cites = " ".join(support_cites)
        lines.append(f"Bằng chứng: {body} {cites}")
    if counter_parts:
        body = " ".join(counter_parts)
        cites = " ".join(counter_cites)
        lines.append(f"Bằng chứng phản bác: {body} {cites}")
    return "\n".join(lines)

