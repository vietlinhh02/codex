**ViFact — Colab-Ready Vietnamese Fact-Checking Scaffold**

- **Mục tiêu:** Cung cấp bộ khung huấn luyện và suy luận cho hệ AFC tiếng Việt: Truy hồi (BM25 + Dense), Re-rank, Suy luận đa bước (Reasoner), Xác minh (NLI), và Giải thích trung thực (Explainer) có gán nguồn.
- **Trạng thái:** Hoạt động với corpus CSV/JSONL, tích hợp ngay BM25 bạn đã có. Có script huấn luyện Module 2 (Reasoner) và Module 3 (Explainer).

**Đã Triển Khai**
- **Retrieval lai:** Dense bi-encoder + nhập top‑K BM25, hợp nhất bằng Reciprocal Rank Fusion (RRF).
- **Rerank:** Cross-encoder tùy chọn; mặc định cosine fallback nhanh.
- **Verify (NLI):** Bộ phân loại NLI đa ngữ → {SUPPORTED, REFUTED, INSUFFICIENT}.
- **Module 2 — Reasoner:** Multi-hop reasoning (2 tầng cross-attention), Adaptive Confidence (sufficiency head), Rationale Regularization (gold/negative evidences).
- **Module 3 — Explainer:** Rationale selector (huấn luyện tách câu) + lắp ráp giải thích có trích dẫn nguồn và hiển thị “bằng chứng phản bác”.

**Cấu Trúc Thư Mục**
- `src/vifact/`
  - `config.py`: cấu hình model/tham số.
  - `data.py`: loader corpus/claims (CSV/JSONL), tiện ích tách câu.
  - `retrieval/`: `dense.py` (FAISS), `hybrid.py` (RRF).
  - `rerank/simple.py`: cross-encoder optional hoặc cosine fallback.
  - `verify/nli_classifier.py`: NLI 3‑nhãn.
  - `pipeline.py`: pipeline end‑to‑end (retrieval → rerank → verify → explain extractive).
  - `explain/extractive.py`: giải thích extractive nhanh.
  - `reasoner/`: `dataset.py`, `model.py` (MultiHopReasoner + inference helper).
  - `explainer/`: `selector.py` (rationale selector), `generate.py` (lắp ráp giải thích + citations + phản bác).
- `scripts/`
  - `run_pipeline.py`: chạy end‑to‑end cho 1 claim hoặc file claims.
  - `build_index.py`: build và lưu FAISS + doc_ids (tùy chọn).
  - `train_reasoner.py`: train Module 2 (Reasoner).
  - `train_explainer.py`: train Module 3 (Explainer selector).
  - `explain_with_reasoner.py`: inference dùng Reasoner + NLI để sinh giải thích có gán nguồn.
  - `prepare_ise_kvjson.py`: chuyển dữ liệu KV JSON (dạng bạn đưa) → `corpus.csv`, `train.jsonl`, `valid.jsonl`, `bm25.json`.
- `docs/DATASET.md`: hướng dẫn chuẩn bị corpus/claims và JSONL train cho Module 2–3.

**Cài Đặt Nhanh Trên Colab**
- Cài thư viện và cấu hình đường dẫn:
```
!pip -q install -r requirements.txt
import sys; sys.path.append('src')
```
- Chuẩn bị corpus (CSV/JSONL) có cột `doc_id`, `text` (tùy chọn `title`). Nên chunk bài dài 200–400 token/đoạn.

**Chạy Pipeline Cơ Bản (Retrieval → Verify → Explain)**
- Một claim, không BM25:
```
!python scripts/run_pipeline.py \
  --corpus corpus.csv --format csv \
  --claim "Vắc-xin X được phê duyệt năm 2020." \
  --topk 10
```
- Có BM25 (fusion RRF): tạo `bm25.json` dạng `{"<claim>": ["doc1","doc2", ...]}` rồi:
```
!python scripts/run_pipeline.py \
  --corpus corpus.csv --format csv \
  --claim "..." \
  --bm25 bm25.json --topk 10
```
- Batch claims từ JSONL/CSV và lưu kết quả:
```
!python scripts/run_pipeline.py \
  --corpus corpus.csv --format csv \
  --claims claims.jsonl \
  --bm25 bm25.json \
  --topk 10 \
  --out results.jsonl
```

**Chuẩn Bị Dữ Liệu Huấn Luyện (Module 2–3)**
- JSONL huấn luyện/validation, mỗi dòng:
```
{
  "claim_id": "c001",
  "claim": "Vắc-xin X được phê duyệt năm 2020.",
  "label": "SUPPORTED",                           // SUPPORTED | REFUTED | INSUFFICIENT
  "evidences": [                                   // top-K từ BM25/dense của bạn
    {"doc_id": "art001#p02", "text": "..."},
    {"doc_id": "art050#p03", "text": "..."}
  ],
  "gold_evidence_ids": ["art001#p02"],            // optional: regularize attention
  "counter_evidence_ids": ["art050#p03"],          // optional: penalize attention
  "rationales": ["câu chứa '2020'"]                // optional: câu hoặc index câu
}
```
- Cách tạo nhanh từ ISE‑DSC01 train: với mỗi claim → lấy top‑K doc_id bằng BM25 của bạn → join text từ corpus → điền `evidences`; nếu có nhãn, điền `label` và `gold_evidence_ids`.

**Dành riêng cho dữ liệu của bạn (KV JSON)**
- Dữ liệu dạng:
```
{
  "7125": {"context": "...", "claim": "...", "verdict": "SUPPORTED|REFUTED|NEI", "evidence": "..."|null, "domain": "..."},
  "18829": {...},
  ...
}
```
- Chuyển đổi và tạo BM25 tự động (rank-bm25) + corpus/chunks:
```
!python scripts/prepare_ise_kvjson.py \
  --input ise_train.json \
  --out_dir ./prepared \
  --max_tokens_per_chunk 220 \
  --topk_bm25 20 \
  --valid_ratio 0.1
```
- Kết quả:
  - `prepared/corpus.csv`: corpus đã chunk (doc_id, text)
  - `prepared/train.jsonl`, `prepared/valid.jsonl`: cho Module 2–3
  - `prepared/bm25.json`: mapping claim → list doc_id top‑K (phục vụ hybrid)

Sau bước này bạn có thể:
- Train Reasoner: `!python scripts/train_reasoner.py --train ./prepared/train.jsonl --valid ./prepared/valid.jsonl --out_dir ./ckpt/reasoner`
- Train Explainer: `!python scripts/train_explainer.py --train ./prepared/train.jsonl --valid ./prepared/valid.jsonl --out_dir ./ckpt/explainer`
- Chạy pipeline hybrid (dense+BM25):
```
!python scripts/run_pipeline.py \
  --corpus ./prepared/corpus.csv --format csv \
  --claims ./prepared/train.jsonl \
  --bm25 ./prepared/bm25.json \
  --topk 10 \
  --out results.jsonl
```

**Huấn Luyện Module 2 — Reasoner**
```
!python scripts/train_reasoner.py \
  --train train.jsonl \
  --valid valid.jsonl \
  --out_dir ./ckpt/reasoner \
  --base_model xlm-roberta-base \
  --epochs 3 --batch_size 4 --lr 2e-5 \
  --max_evidences 6 --max_len_claim 128 --max_len_evi 192
```
- Kết quả: `./ckpt/reasoner/reasoner.pt` và `config.json`.
- Inference nhanh: dùng `reason_infer(...)` trong `src/vifact/reasoner/model.py` (đã có adaptive confidence).

**Huấn Luyện Module 3 — Explainer (Rationale Selector)**
```
!python scripts/train_explainer.py \
  --train train.jsonl \
  --valid valid.jsonl \
  --out_dir ./ckpt/explainer \
  --base_model xlm-roberta-base \
  --epochs 3 --batch_size 8 --lr 2e-5 --max_sents 12
```
- Mục tiêu: dự đoán xác suất mỗi câu là rationale → phục vụ giải thích trung thực.

**Sinh Giải Thích Có Gán Nguồn và Phản Bác**
- Dùng Reasoner đã train + NLI để đánh dấu counter-evidence, rồi lắp ghép giải thích:
```
!python scripts/explain_with_reasoner.py \
  --ckpt ./ckpt/reasoner \
  --claim "..." \
  --evidences evidences.jsonl \
  --topk 6
```
- `evidences.jsonl`: mỗi dòng `{"doc_id":"...","text":"..."}` (thường lấy từ retrieval/rerank top‑K).

**Tuỳ Chỉnh Mô Hình**
- `src/vifact/config.py`:
  - Dense: `dense_model` (mặc định `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
  - Rerank: `cross_encoder_model` (ví dụ `jinaai/jina-reranker-v2-base-multilingual`).
  - NLI: `verify.nli_model` (mặc định mDeBERTa multilingual).
  - Explain: số câu extractive.

**Mẹo & Tránh Lỗi**
- Chunking: luôn chia nhỏ văn bản dài để tránh truncation và tăng recall.
- Khớp ID: `doc_id` giữa BM25, corpus và JSONL train phải khớp tuyệt đối.
- Colab/FAISS: nếu lưu/đọc FAISS lỗi, cứ build index lại trong runtime.
- GPU: giảm `max_evidences`, `max_len_evi` khi thiếu VRAM.
- Nhãn: chuẩn hoá nhãn về {SUPPORTED, REFUTED, INSUFFICIENT}.

**Lộ Trình Đề Xuất (Từ ISE‑DSC01 Train)**
- Tạo `corpus.csv` (chunk bài, `doc_id`,`text`).
- Chạy BM25 của bạn → sinh `bm25.json` (claim → list doc_id).
- Tạo `train.jsonl`/`valid.jsonl` theo schema (nếu có gold evidence thì điền).
- Train Reasoner/Explainer → dùng `explain_with_reasoner.py` để sinh nhãn + giải thích có citation.

Xem thêm chi tiết chuẩn bị dữ liệu trong `docs/DATASET.md`.
