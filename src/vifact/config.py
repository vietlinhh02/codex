from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalConfig:
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_seq_length: int = 256
    use_faiss_gpu: bool = False
    faiss_nprobe: int = 16
    top_k_dense: int = 50
    top_k_bm25: int = 50
    rrf_k: int = 60  # Reciprocal Rank Fusion parameter


@dataclass
class RerankConfig:
    # Optional cross-encoder. If None, falls back to cosine rerank.
    cross_encoder_model: Optional[str] = None  # e.g. "jinaai/jina-reranker-v2-base-multilingual"
    batch_size: int = 32
    top_k: int = 20


@dataclass
class VerifyConfig:
    nli_model: str = (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    max_seq_length: int = 256
    batch_size: int = 16


@dataclass
class ExplainConfig:
    method: str = "extractive"  # "extractive" or "none"
    max_sentences: int = 2


@dataclass
class PipelineConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)

