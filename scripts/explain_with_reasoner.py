#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch

from vifact.reasoner.model import MultiHopReasoner, reason_infer
from vifact.verify.nli_classifier import NLIClaimVerifier
from vifact.explainer.generate import assemble_explanation


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
    ap = argparse.ArgumentParser(description="Explain with ViFact-Reasoner + NLI counter-evidence")
    ap.add_argument("--ckpt", required=True, help="Reasoner checkpoint dir")
    ap.add_argument("--claim", required=True)
    ap.add_argument("--evidences", required=True, help="JSONL of evidences: lines {doc_id,text}")
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    evidences = []
    with open(args.evidences, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            evidences.append(json.loads(line))
            if len(evidences) >= args.topk:
                break

    reasoner = load_reasoner(args.ckpt)
    out = reason_infer(reasoner, args.claim, evidences)
    label = out["label"]
    attn = out["attn"] if isinstance(out["attn"], list) else out["attn"]

    # Flag counter-evidence via NLI per-evidence
    nli = NLIClaimVerifier("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    texts = [e["text"] for e in evidences]
    _, score_list = nli.predict(args.claim, texts)
    counter_flags = [lbl == "REFUTED" for (lbl, _) in score_list]

    explanation = assemble_explanation(
        args.claim,
        evidences,
        evidence_attn=attn[: len(evidences)] if attn else None,
        label=label,
        counter_flags=counter_flags,
    )
    print(json.dumps({"label": label, "explanation": explanation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

