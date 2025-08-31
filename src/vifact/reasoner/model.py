from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MultiHopReasoner(nn.Module):
    def __init__(
        self,
        base_model: str = "xlm-roberta-base",
        hidden: int = 768,
        num_heads: int = 8,
        num_labels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)
        self.hidden = hidden
        self.attn1 = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj_q = nn.Linear(hidden * 2, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        # Sufficiency head for adaptive confidence (predicts if SUPPORTED/REFUTED vs INSUFFICIENT)
        self.suff_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS pooling for xlm-roberta: first token
        cls = out.last_hidden_state[:, 0, :]
        return cls

    def encode_evidences(self, evi_ids: torch.Tensor, evi_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # evi_ids: (B, E, L)
        B, E, L = evi_ids.size()
        flat_ids = evi_ids.view(B * E, L)
        flat_mask = evi_mask.view(B * E, L)
        cls = self.encode(flat_ids, flat_mask)  # (B*E, H)
        cls = cls.view(B, E, -1)  # (B, E, H)
        # Build a per-batch padding mask for evidences (True means pad/ignore)
        pad_mask = (evi_mask.sum(dim=-1) == 0)  # (B, E)
        return cls, pad_mask

    def forward(
        self,
        claim_input_ids: torch.Tensor,
        claim_attention_mask: torch.Tensor,
        evi_input_ids: torch.Tensor,
        evi_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        gold_masks: Optional[torch.Tensor] = None,
        neg_masks: Optional[torch.Tensor] = None,
        evi_counts: Optional[torch.Tensor] = None,
        rationale_lambda: float = 0.3,
        neg_rationale_lambda: float = 0.1,
    ) -> Dict:
        B = claim_input_ids.size(0)
        q0 = self.encode(claim_input_ids, claim_attention_mask)  # (B, H)
        ev_vecs, ev_pad = self.encode_evidences(evi_input_ids, evi_attention_mask)  # (B, E, H), (B,E)

        # Cross-attention hop 1: query is q0 over evidence vectors as a sequence
        q = q0.unsqueeze(1)  # (B,1,H)
        k = ev_vecs  # (B,E,H)
        v = ev_vecs
        ctx1, attn1 = self.attn1(q, k, v, key_padding_mask=ev_pad)  # ctx1: (B,1,H), attn1: (B,1,E)
        ctx1 = ctx1.squeeze(1)
        attn1 = attn1.squeeze(1)  # (B,E)

        # Hop 2: combine q0 and ctx1
        q1 = torch.tanh(self.proj_q(torch.cat([q0, ctx1], dim=-1))).unsqueeze(1)  # (B,1,H)
        ctx2, attn2 = self.attn2(q1, k, v, key_padding_mask=ev_pad)
        ctx2 = ctx2.squeeze(1)
        attn2 = attn2.squeeze(1)  # (B,E)

        # Aggregate features
        def pair_feats(a, b):
            return torch.cat([a, b, torch.abs(a - b), a * b, (a + b) / 2.0], dim=-1)

        feats = pair_feats(q0, ctx2)
        logits = self.classifier(self.dropout(feats))  # (B,3)

        suff_feats = torch.cat([q0, ctx1, ctx2], dim=-1)
        suff_logit = self.suff_head(self.dropout(suff_feats)).squeeze(-1)  # (B,)

        out = {
            "logits": logits,
            "attn1": attn1,
            "attn2": attn2,
            "suff_logit": suff_logit,
        }

        if labels is not None:
            ce = nn.CrossEntropyLoss()
            cls_loss = ce(logits, labels)

            # Sufficiency target: 1 for SUPPORTED/REFUTED, 0 for INSUFFICIENT
            suff_target = (labels != 2).float()
            bce = nn.BCEWithLogitsLoss()
            suff_loss = bce(suff_logit, suff_target)

            loss = cls_loss + 0.5 * suff_loss

            # Rationale regularization if gold masks exist
            if gold_masks is not None and gold_masks.sum() > 0:
                # Normalize gold to distribution across available evidences per item
                gold = gold_masks / (gold_masks.sum(dim=-1, keepdim=True) + 1e-8)
                # Encourage attn2 to match gold (KL or CE)
                # Add epsilon to avoid log(0)
                attn = attn2 + 1e-8
                rat_loss = (gold * (gold.add(1e-8).log() - attn.log())).sum(dim=-1).mean()
                loss = loss + rationale_lambda * rat_loss

            if neg_masks is not None and neg_masks.sum() > 0:
                # Penalize attention mass on known counter-evidence
                penal = (attn2 * neg_masks).sum(dim=-1).mean()
                loss = loss + neg_rationale_lambda * penal

            out["loss"] = loss
        return out


@torch.no_grad()
def reason_infer(
    model: MultiHopReasoner,
    claim: str,
    evidences: list[dict],
    prob_threshold: float = 0.55,
    suff_threshold: float = 0.5,
) -> Dict:
    model.eval()
    tok = model.tokenizer
    device = next(model.parameters()).device
    c = tok(claim, return_tensors="pt", truncation=True, max_length=128).to(device)
    texts = [e["text"] for e in evidences]
    if not texts:
        texts = [""]
    t = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=192)
    evi_ids = t.input_ids.unsqueeze(0).to(device)
    evi_mask = t.attention_mask.unsqueeze(0).to(device)

    out = model(
        claim_input_ids=c.input_ids,
        claim_attention_mask=c.attention_mask,
        evi_input_ids=evi_ids,
        evi_attention_mask=evi_mask,
    )
    probs = torch.softmax(out["logits"], dim=-1)[0]
    pred = int(torch.argmax(probs).item())
    labels = ["SUPPORTED", "REFUTED", "INSUFFICIENT"]
    maxp = float(probs[pred].item())
    suff_p = torch.sigmoid(out["suff_logit"])[0].item()
    final = labels[pred]
    if maxp < prob_threshold or suff_p < suff_threshold:
        final = "INSUFFICIENT"
    return {
        "label": final,
        "probs": {labels[i]: float(probs[i].item()) for i in range(3)},
        "attn": out["attn2"].cpu().numpy().tolist()[0],
        "suff_prob": float(suff_p),
    }

