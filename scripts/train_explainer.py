#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from vifact.explainer.selector import RationaleSelector, build_sentence_batches
from vifact.data import split_sentences


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_dataset(jsonl_path, max_sents=12):
    claims, sent_lists, labels = [], [], []
    for obj in load_jsonl(jsonl_path):
        claim = obj["claim"]
        evidences = obj.get("evidences", [])
        # Flatten sentences across top evidences
        sents = []
        for ev in evidences:
            sents.extend(split_sentences(ev["text"]))
            if len(sents) >= max_sents:
                break
        sents = sents[:max_sents] if sents else [""]
        # Build labels if available
        gold_spans = obj.get("rationales")  # list of strings or indices
        y = [0] * len(sents)
        if isinstance(gold_spans, list) and gold_spans:
            # match by substring
            for i, s in enumerate(sents):
                for gs in gold_spans:
                    if isinstance(gs, int) and gs == i:
                        y[i] = 1
                        break
                    if isinstance(gs, str) and gs and gs in s:
                        y[i] = 1
                        break
        claims.append(claim)
        sent_lists.append(sents)
        labels.append(y)
    return claims, sent_lists, labels


def collate_fn(tokenizer, batch, max_len_claim=128, max_len_sent=96):
    claims, sent_lists, labels = zip(*batch)
    c, sent_ids, sent_mask = build_sentence_batches(tokenizer, list(claims), list(sent_lists), max_len_claim, max_len_sent)
    import torch

    max_sents = sent_ids.size(1)
    y = torch.zeros((len(batch), max_sents), dtype=torch.float)
    for i, lab in enumerate(labels):
        for j, v in enumerate(lab[:max_sents]):
            y[i, j] = float(v)
    return c.input_ids, c.attention_mask, sent_ids, sent_mask, y


def main():
    ap = argparse.ArgumentParser(description="Train ViFact-Explainer rationale selector")
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_sents", type=int, default=12)
    args = ap.parse_args()

    model = RationaleSelector(base_model=args.base_model)
    tok = model.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = list(zip(*build_dataset(args.train, max_sents=args.max_sents)))
    valid_data = list(zip(*build_dataset(args.valid, max_sents=args.max_sents)))
    # train_data is list of tuples; convert to list of triples for DataLoader
    train_list = [(a, b, c) for a, b, c in zip(*build_dataset(args.train, max_sents=args.max_sents))]
    valid_list = [(a, b, c) for a, b, c in zip(*build_dataset(args.valid, max_sents=args.max_sents))]

    train_dl = DataLoader(train_list, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(tok, b))
    valid_dl = DataLoader(valid_list, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(tok, b))

    optim = AdamW(model.parameters(), lr=args.lr)

    best = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        for step, batch in enumerate(train_dl, 1):
            ci, cm, si, sm, y = batch
            ci, cm, si, sm, y = ci.to(device), cm.to(device), si.to(device), sm.to(device), y.to(device)
            out = model(ci, cm, si, sm, y)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        # simple validation: average AUC-like proxy using threshold 0.5
        model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for batch in valid_dl:
                ci, cm, si, sm, y = batch
                ci, cm, si, sm, y = ci.to(device), cm.to(device), si.to(device), sm.to(device), y.to(device)
                out = model(ci, cm, si, sm)
                probs = torch.sigmoid(out["logits"]) > 0.5
                tp += ((probs == 1) & (y == 1)).sum().item()
                tn += ((probs == 0) & (y == 0)).sum().item()
                fp += ((probs == 1) & (y == 0)).sum().item()
                fn += ((probs == 0) & (y == 1)).sum().item()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        print(f"epoch {epoch} valid rationale F1: {f1:.4f}")
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), out_dir / "explainer.pt")
            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump({"base_model": args.base_model}, f)

    print(f"best rationale F1: {best:.4f}")


if __name__ == "__main__":
    main()

