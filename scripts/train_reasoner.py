#!/usr/bin/env python
import argparse
import json
from pathlib import Path

# Bootstrap repo-local src on sys.path for direct script runs (Windows-friendly)
try:
    import vifact  # type: ignore
except Exception:
    import os, sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src = os.path.join(_root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from vifact.reasoner.dataset import ReasonerCollator, ReasonerJSONLDataset
from vifact.reasoner.model import MultiHopReasoner


def evaluate(model, dl, device):
    model.eval()
    correct = total = 0
    from math import isfinite

    with torch.no_grad():
        for batch in dl:
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device) if torch.is_tensor(batch[k]) else batch[k]
            out = model(
                batch["claim_input_ids"],
                batch["claim_attention_mask"],
                batch["evi_input_ids"],
                batch["evi_attention_mask"],
            )
            probs = torch.softmax(out["logits"], dim=-1)
            pred = probs.argmax(dim=-1)
            total += pred.size(0)
            correct += (pred == batch.get("labels", pred)).sum().item()
    acc = correct / max(1, total)
    return {"accuracy": acc}


def main():
    ap = argparse.ArgumentParser(description="Train ViFact-Reasoner (multi-hop)")
    ap.add_argument("--train", required=True, help="Path to train JSONL")
    ap.add_argument("--valid", required=True, help="Path to valid JSONL")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_evidences", type=int, default=6)
    ap.add_argument("--max_len_claim", type=int, default=128)
    ap.add_argument("--max_len_evi", type=int, default=192)
    ap.add_argument("--rationale_lambda", type=float, default=0.3)
    ap.add_argument("--neg_rationale_lambda", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHopReasoner(base_model=args.base_model)
    model.to(device)

    train_ds = ReasonerJSONLDataset(args.train, max_evidences=args.max_evidences)
    valid_ds = ReasonerJSONLDataset(args.valid, max_evidences=args.max_evidences)
    collate = ReasonerCollator(model.tokenizer, max_len_claim=args.max_len_claim, max_len_evi=args.max_len_evi)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optim = AdamW(model.parameters(), lr=args.lr)
    num_steps = args.epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * num_steps), num_steps)

    best_acc = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_dl, 1):
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device) if torch.is_tensor(batch[k]) else batch[k]
            out = model(
                batch["claim_input_ids"],
                batch["claim_attention_mask"],
                batch["evi_input_ids"],
                batch["evi_attention_mask"],
                labels=batch["labels"],
                gold_masks=batch["gold_masks"],
                neg_masks=batch["neg_masks"],
                evi_counts=batch["evi_counts"],
                rationale_lambda=args.rationale_lambda,
                neg_rationale_lambda=args.neg_rationale_lambda,
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        metrics = evaluate(model, valid_dl, device)
        print(f"epoch {epoch} valid: {metrics}")
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            # save
            torch.save(model.state_dict(), out_dir / "reasoner.pt")
            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"base_model": args.base_model}, f)

    print(f"best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
