from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


Label3 = {"SUPPORTED": 0, "REFUTED": 1, "INSUFFICIENT": 2}


@dataclass
class ReasonerSample:
    claim_id: str
    claim: str
    evidences: List[Dict[str, Any]]  # each: {doc_id, text}
    label: str
    gold_evidence_ids: Optional[List[str]] = None
    counter_evidence_ids: Optional[List[str]] = None


class ReasonerJSONLDataset(Dataset):
    def __init__(
        self,
        path: str,
        max_evidences: int = 6,
    ) -> None:
        self.data: List[ReasonerSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                evidences = obj.get("evidences") or obj.get("evidence") or []
                evidences = evidences[:max_evidences]
                self.data.append(
                    ReasonerSample(
                        claim_id=str(obj.get("claim_id", obj.get("id", len(self.data)))),
                        claim=str(obj.get("claim")),
                        evidences=evidences,
                        label=str(obj.get("label", "INSUFFICIENT")).upper(),
                        gold_evidence_ids=obj.get("gold_evidence_ids"),
                        counter_evidence_ids=obj.get("counter_evidence_ids"),
                    )
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> ReasonerSample:
        return self.data[idx]


class ReasonerCollator:
    def __init__(self, tokenizer, max_len_claim: int = 128, max_len_evi: int = 192):
        self.tok = tokenizer
        self.max_len_c = max_len_claim
        self.max_len_e = max_len_evi

    def __call__(self, batch: List[ReasonerSample]) -> Dict[str, torch.Tensor]:
        claims = [b.claim for b in batch]
        # tokenize claim
        c = self.tok(
            claims,
            padding=True,
            truncation=True,
            max_length=self.max_len_c,
            return_tensors="pt",
        )
        # tokenize evidences per item (ragged list) then pad to max count in batch
        max_ev = max(len(b.evidences) for b in batch)
        ev_input_ids: List[torch.Tensor] = []
        ev_attn_mask: List[torch.Tensor] = []
        ev_counts: List[int] = []
        ev_doc_ids: List[List[str]] = []
        for b in batch:
            texts = [e["text"] for e in b.evidences]
            ev_doc_ids.append([str(e.get("doc_id", i)) for i, e in enumerate(b.evidences)])
            if texts:
                t = self.tok(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len_e,
                    return_tensors="pt",
                )
                ids, mask = t.input_ids, t.attention_mask
            else:
                # no evidence: create a single empty place-holder
                ids = torch.zeros((1, 2), dtype=torch.long)
                mask = torch.zeros((1, 2), dtype=torch.long)
            ev_counts.append(ids.size(0))
            # pad to max_ev along evidence dimension
            pad_e = max_ev - ids.size(0)
            if pad_e > 0:
                pad_ids = torch.zeros((pad_e, ids.size(1)), dtype=torch.long)
                pad_mask = torch.zeros((pad_e, mask.size(1)), dtype=torch.long)
                ids = torch.cat([ids, pad_ids], dim=0)
                mask = torch.cat([mask, pad_mask], dim=0)
            ev_input_ids.append(ids)
            ev_attn_mask.append(mask)

        ev_input_ids = torch.stack(ev_input_ids, dim=0)  # (B, E, L)
        ev_attn_mask = torch.stack(ev_attn_mask, dim=0)

        labels = torch.tensor([Label3.get(b.label, 2) for b in batch], dtype=torch.long)

        # Build attention supervision if present
        gold_masks = torch.zeros((len(batch), max_ev), dtype=torch.float)
        neg_masks = torch.zeros((len(batch), max_ev), dtype=torch.float)
        for bi, b in enumerate(batch):
            ids = ev_doc_ids[bi]
            if b.gold_evidence_ids:
                for i, did in enumerate(ids):
                    if did in set(b.gold_evidence_ids):
                        gold_masks[bi, i] = 1.0
            if b.counter_evidence_ids:
                for i, did in enumerate(ids):
                    if did in set(b.counter_evidence_ids):
                        neg_masks[bi, i] = 1.0

        return {
            "claim_input_ids": c.input_ids,
            "claim_attention_mask": c.attention_mask,
            "evi_input_ids": ev_input_ids,
            "evi_attention_mask": ev_attn_mask,
            "labels": labels,
            "gold_masks": gold_masks,
            "neg_masks": neg_masks,
            "evi_counts": torch.tensor(ev_counts, dtype=torch.long),
        }

