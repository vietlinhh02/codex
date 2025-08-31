from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_MAP_3WAY = {
    "entailment": "SUPPORTED",
    "contradiction": "REFUTED",
    "neutral": "INSUFFICIENT",
}


class NLIClaimVerifier:
    def __init__(self, model_name: str, max_seq_length: int = 256, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

        # Derive label mapping from model config if possible
        id2label = getattr(self.model.config, "id2label", None)
        self.id2label = {int(k): v for k, v in id2label.items()} if id2label else None
        self.max_len = max_seq_length

    def predict(self, claim: str, evidence_texts: List[str]) -> Tuple[str, List[Tuple[str, float]]]:
        # Returns (final_label, per_evidence_labels_with_scores)
        results: List[Tuple[str, float]] = []
        best_label = "INSUFFICIENT"
        best_score = -1.0

        for ev in evidence_texts:
            inputs = self.tokenizer(
                claim,
                ev,
                max_length=self.max_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # Figure out label names
            if self.id2label:
                labels = [self.id2label[i].lower() for i in range(len(probs))]
            else:
                labels = ["entailment", "neutral", "contradiction"]
            label_scores = dict(zip(labels, probs.tolist()))

            # Map to 3-way output
            mapped_scores = {
                "SUPPORTED": label_scores.get("entailment", 0.0),
                "REFUTED": label_scores.get("contradiction", 0.0),
                "INSUFFICIENT": label_scores.get("neutral", 0.0),
            }
            # Choose the max
            pred_label = max(mapped_scores.items(), key=lambda x: x[1])
            results.append(pred_label)
            if pred_label[1] > best_score:
                best_label, best_score = pred_label

        return best_label, results

