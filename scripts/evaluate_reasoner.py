#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

from vifact.reasoner.model import MultiHopReasoner, reason_infer


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def load_reasoner(ckpt_dir: str) -> MultiHopReasoner:
    cfg_path = Path(ckpt_dir) / "config.json"
    base = "xlm-roberta-base"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            base = json.load(f).get("base_model", base)
    model = MultiHopReasoner(base_model=base)
    state = torch.load(Path(ckpt_dir) / "reasoner.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


def main():
    ap = argparse.ArgumentParser(description="Evaluate ViFact-Reasoner on JSONL valid set")
    ap.add_argument("--valid", required=True, help="Path to valid.jsonl")
    ap.add_argument("--ckpt", required=True, help="Reasoner checkpoint dir")
    ap.add_argument("--topk", type=int, default=6, help="Use top-K evidences from file")
    args = ap.parse_args()

    model = load_reasoner(args.ckpt)
    gold, pred = [], []
    for obj in load_jsonl(args.valid):
        claim = obj["claim"]
        label = obj.get("label", "INSUFFICIENT").upper()
        evidences = obj.get("evidences", [])[: args.topk]
        out = reason_infer(model, claim, evidences)
        pred_label = out["label"].upper()
        gold.append(label)
        pred.append(pred_label)

    labels = ["SUPPORTED", "REFUTED", "INSUFFICIENT"]
    acc = accuracy_score(gold, pred)
    macro_f1 = f1_score(gold, pred, labels=labels, average="macro")
    report = classification_report(gold, pred, labels=labels)
    print(json.dumps({"accuracy": acc, "macro_f1": macro_f1}, indent=2))
    print(report)


if __name__ == "__main__":
    main()

