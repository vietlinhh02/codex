**Dataset Guide — ISE-DSC01 (Train) and Corpus Prep**

- **Goal:** Use your ISE-DSC01 train split to build a retrieval corpus and evaluate the pipeline.

**What We Assume**
- You currently have only `ise-dsc01-train` and you already built a BM25 that can return similar documents for a query.
- ISE-DSC01 fields vary by release; adapt column/field names accordingly.

**Two Artifacts You Need**
- Retrieval corpus (CSV or JSONL): a table of candidate evidence texts.
  - Required columns: `doc_id`, `text` (optionally `title`).
  - If your train file contains long articles, split into paragraphs or sentences and assign synthetic IDs like `art123#p05`.
- Claims file (optional): a list of claims to process.
  - JSONL lines with fields `{ "claim_id": ..., "claim": ... }`, or a CSV with a `claim` column.

**Reasoner/Explainer Training JSONL**
- For Module 2–3 training, build a JSONL combining claims and their candidate evidences:
```
{
  "claim_id": "c001",
  "claim": "Vắc-xin X được phê duyệt năm 2020.",
  "label": "SUPPORTED",
  "evidences": [
    {"doc_id": "art001#p02", "text": "..."},
    {"doc_id": "art050#p03", "text": "..."}
  ],
  "gold_evidence_ids": ["art001#p02"],              // optional, for rationale regularization
  "counter_evidence_ids": ["art050#p03"],            // optional, penalize attention here
  "rationales": ["câu chứa năm 2020" ]               // optional, sentence spans or indices
}
```
- Cách tạo nhanh từ ISE-DSC01 train + BM25:
  - Lấy mỗi claim trong train, chạy BM25 để lấy top-K `doc_id` từ corpus bạn đã tạo.
  - Ghép `text` tương ứng từ corpus -> mảng `evidences`.
  - `label`: dùng nhãn train (nếu có). Nếu không có, có thể tạo weak labels từ luật hoặc bỏ qua training.
  - Nếu có gold evidence: điền `gold_evidence_ids` để bật Rationale Regularization.

**From ISE-DSC01 Train → Corpus**
- If the train set contains pairs or triples like `(query/claim, doc/evidence, label)`, extract all `doc/evidence` as rows in the corpus.
- If only article-level text exists, pre-split into chunks (200–400 tokens per chunk is a good target).

Example (CSV corpus):
```
doc_id,title,text
art001#p01,Title of A,"Đoạn văn 1 ..."
art001#p02,Title of A,"Đoạn văn 2 ..."
```

**Running With BM25**
- Save BM25 results per-claim as a JSON mapping where keys are raw claim strings and values are ranked lists of `doc_id` present in the corpus:
```
{
  "Vắc-xin X được phê duyệt năm 2020.": ["art001#p02", "art050#p03", ...],
  "...": [ ... ]
}
```
- Then call:
```
!python scripts/run_pipeline.py \
  --corpus corpus.csv --format csv \
  --claims claims.jsonl \
  --bm25 bm25.json \
  --topk 10 \
  --out results.jsonl
```

**Colab Tips**
- Use Drive for persistence: `from google.colab import drive; drive.mount('/content/drive')`.
- Install once per session: `!pip -q install -r requirements.txt`.
- If FAISS save/load fails, rebuild the dense index by running `run_pipeline.py` directly; it internally builds in-memory index each run.

**Common Pitfalls**
- ID mismatch: Ensure BM25 `doc_id`s exactly match the `doc_id` column in the corpus. Any mismatch will be dropped during fusion.
- Oversized docs: Without chunking, retrieval and NLI may underperform due to truncation. Always chunk long texts.
- Language coverage: The default dense model and NLI are multilingual. If performance lags, consider Vietnamese-specialized models.
- Non-UTF8 input: Normalize encoding to UTF-8 to avoid tokenizer errors.

**Colab Training Commands**
- Reasoner: `!python scripts/train_reasoner.py --train train.jsonl --valid valid.jsonl --out_dir ./ckpt/reasoner`
- Explainer: `!python scripts/train_explainer.py --train train.jsonl --valid valid.jsonl --out_dir ./ckpt/explainer`

**Adapting to Your Train Schema**
- If your train data is JSONL with keys like `{"id":..., "query":..., "doc":..., "label":...}`:
  - Build corpus from `doc` values.
  - Build claims from `query` values.
  - Map labels to {SUPPORTED/REFUTED/INSUFFICIENT} if applicable for evaluation.

**Evaluation (Optional)**
- Retrieval: compute Recall@k on gold evidence IDs if available.
- Verification: compare predicted labels against gold labels (Accuracy, Macro-F1).
- Explanation: start with extractive; human spot-check for plausibility and faithfulness.
