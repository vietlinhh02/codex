from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RationaleSelector(nn.Module):
    """
    Scores sentences given a claim; trains with binary labels per sentence when available.
    Outputs per-sentence probabilities to serve as faithful extractive rationales.
    """

    def __init__(self, base_model: str = "xlm-roberta-base", hidden: int = 768, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(
        self,
        claim_ids: torch.Tensor,
        claim_mask: torch.Tensor,
        sent_ids: torch.Tensor,
        sent_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Dict:
        # sent_ids: (B, S, L)
        B, S, L = sent_ids.size()
        c = self.encode(claim_ids, claim_mask)  # (B,H)
        flat_ids = sent_ids.view(B * S, L)
        flat_mask = sent_mask.view(B * S, L)
        s = self.encode(flat_ids, flat_mask).view(B, S, -1)
        # Pair features for each sentence
        c_exp = c.unsqueeze(1).expand_as(s)
        feats = torch.cat([c_exp, s, torch.abs(c_exp - s), c_exp * s], dim=-1)
        logits = self.scorer(feats).squeeze(-1)  # (B,S)
        out = {"logits": logits}
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            out["loss"] = loss
        return out


def build_sentence_batches(tokenizer, claims: List[str], evidence_sent_lists: List[List[str]], max_len_claim=128, max_len_sent=96):
    import itertools
    import torch

    B = len(claims)
    max_sents = max(len(sents) if sents else 1 for sents in evidence_sent_lists)
    # Tokenize claims once
    c = tokenizer(claims, padding=True, truncation=True, max_length=max_len_claim, return_tensors="pt")
    # Tokenize sentences per example then pad to max_sents
    all_ids, all_mask = [], []
    for sents in evidence_sent_lists:
        if not sents:
            sents = [""]
        t = tokenizer(sents, padding=True, truncation=True, max_length=max_len_sent, return_tensors="pt")
        ids, mask = t.input_ids, t.attention_mask
        pad = max_sents - ids.size(0)
        if pad > 0:
            pad_ids = torch.zeros((pad, ids.size(1)), dtype=torch.long)
            pad_mask = torch.zeros((pad, mask.size(1)), dtype=torch.long)
            ids = torch.cat([ids, pad_ids], dim=0)
            mask = torch.cat([mask, pad_mask], dim=0)
        all_ids.append(ids)
        all_mask.append(mask)
    sent_ids = torch.stack(all_ids, dim=0)
    sent_mask = torch.stack(all_mask, dim=0)
    return c, sent_ids, sent_mask

