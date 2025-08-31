from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from .config import PipelineConfig
from .data import CorpusDoc
from .retrieval.dense import DenseRetriever
from .retrieval.hybrid import reciprocal_rank_fusion
from .rerank.simple import SimpleReranker
from .verify.nli_classifier import NLIClaimVerifier
from .explain.extractive import extractive_explanation

# Optional Reasoner
try:
    from .reasoner.model import MultiHopReasoner, reason_infer
    _HAS_REASONER = True
except Exception:
    MultiHopReasoner = None  # type: ignore
    reason_infer = None  # type: ignore
    _HAS_REASONER = False


class ViFactPipeline:
    def __init__(self, cfg: PipelineConfig, reasoner_ckpt: Optional[str] = None):
        self.cfg = cfg
        self.dense = DenseRetriever(
            model_name=cfg.retrieval.dense_model, max_seq_length=cfg.retrieval.max_seq_length
        )
        self.reranker = SimpleReranker(cfg.rerank.cross_encoder_model)
        self.verifier = NLIClaimVerifier(cfg.verify.nli_model, max_seq_length=cfg.verify.max_seq_length)
        self.corpus: Dict[str, CorpusDoc] = {}
        self.reasoner = None
        if reasoner_ckpt and _HAS_REASONER:
            try:
                import json, torch, os
                from pathlib import Path

                cfg_path = Path(reasoner_ckpt) / "config.json"
                base = "xlm-roberta-base"
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        base = json.load(f).get("base_model", base)
                m = MultiHopReasoner(base_model=base)  # type: ignore
                state = torch.load(Path(reasoner_ckpt) / "reasoner.pt", map_location="cpu")
                m.load_state_dict(state)
                m.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                self.reasoner = m
            except Exception:
                self.reasoner = None

    def build_dense_index(self, docs: List[CorpusDoc]):
        self.corpus = {d.doc_id: d for d in docs}
        texts = [d.text for d in docs]
        ids = [d.doc_id for d in docs]
        self.dense.build_index(texts, ids, use_faiss=True)

    def retrieve(
        self,
        claim: str,
        bm25_candidates: Optional[List[str]] = None,
    ) -> List[str]:
        dense_scored = self.dense.search(claim, top_k=self.cfg.retrieval.top_k_dense)
        dense_ids = [d for d, _ in dense_scored]
        ranks = [dense_ids]
        if bm25_candidates:
            ranks.append(bm25_candidates[: self.cfg.retrieval.top_k_bm25])
        fused = reciprocal_rank_fusion(ranks, k=self.cfg.retrieval.rrf_k, top_k=self.cfg.rerank.top_k)
        return fused

    def rerank(self, claim: str, doc_ids: List[str]) -> List[Tuple[str, float]]:
        candidates = [(doc_id, self.corpus[doc_id].text) for doc_id in doc_ids if doc_id in self.corpus]
        return self.reranker.rerank(claim, candidates, top_k=self.cfg.rerank.top_k)

    def verify(self, claim: str, top_docs: List[str]) -> Tuple[str, List[Tuple[str, float]]]:
        texts = [self.corpus[d].text for d in top_docs if d in self.corpus]
        return self.verifier.predict(claim, texts)

    def explain(self, claim: str, top_docs: List[str]) -> str:
        texts = [self.corpus[d].text for d in top_docs if d in self.corpus]
        if self.cfg.explain.method == "extractive":
            return extractive_explanation(claim, texts, max_sentences=self.cfg.explain.max_sentences)
        return ""

    def run(self, claim: str, bm25_candidates: Optional[List[str]] = None) -> Dict:
        retrieved = self.retrieve(claim, bm25_candidates=bm25_candidates)
        reranked = self.rerank(claim, retrieved)
        top_docs = [d for d, _ in reranked]
        # If Reasoner is available, use it for label; else NLI
        if self.reasoner is not None and _HAS_REASONER:
            evidences = [{"doc_id": d, "text": self.corpus[d].text} for d in top_docs if d in self.corpus]
            ri = reason_infer(self.reasoner, claim, evidences)  # type: ignore
            label = ri["label"]
            evidence_scores = [(evidences[i]["doc_id"], float(ri.get("attn", [0.0] * len(evidences))[i] if ri.get("attn") else 0.0)) for i in range(len(evidences))]
            explanation = extractive_explanation(claim, [e["text"] for e in evidences], max_sentences=self.cfg.explain.max_sentences)
        else:
            label, evidence_scores = self.verify(claim, top_docs)
            explanation = self.explain(claim, top_docs)
        return {
            "claim": claim,
            "retrieved": retrieved,
            "reranked": reranked,
            "label": label,
            "evidence_scores": evidence_scores,
            "explanation": explanation,
        }
