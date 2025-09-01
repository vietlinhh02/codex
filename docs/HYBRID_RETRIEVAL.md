# Hybrid Retrieval (BM25 + Dense) — Quick Guide

Mục tiêu: Xây dựng và chạy truy hồi lai (Hybrid Retrieval) kết hợp BM25 (sparse) và dense (SentenceTransformers + FAISS) để cải thiện recall/precision cho bằng chứng tiếng Việt.

## Chuẩn Bị
- Python 3.10+, cài phụ thuộc: `pip install -r requirements.txt`
- Thêm đường dẫn `src/` khi chạy notebook: `import sys; sys.path.append('src')`
- Dữ liệu Warmup (KV JSON): `Dataset/ise-dsc01-warmup.json`

## 1) Tạo Corpus + BM25 Candidates
Chuyển đổi KV JSON thành corpus và tập train/valid (đồng thời tạo gợi ý ứng viên BM25 cho Hybrid):

```
python scripts/prepare_ise_kvjson.py \
  --input Dataset/ise-dsc01-warmup.json \
  --out_dir prepared/warmup \
  --topk_bm25 50 \
  --valid_ratio 0.1
```

Đầu ra chính:
- `prepared/warmup/corpus.csv` — chứa các chunk `doc_id,text`
- `prepared/warmup/train.jsonl`, `prepared/warmup/valid.jsonl`
- `prepared/warmup/bm25.json` — map `claim` → danh sách `doc_id` gợi ý (BM25)

## 2) Chạy Nhanh Bằng CLI (Khuyến nghị)
Pipeline đã tích hợp Dense + nhập ứng viên BM25 và hợp nhất bằng RRF.

- Truy hồi cho 1 claim:
```
python scripts/run_pipeline.py \
  --corpus prepared/warmup/corpus.csv --format csv \
  --claim "<viết claim của bạn>" \
  --bm25 prepared/warmup/bm25.json \
  --topk 10
```

- Truy hồi hàng loạt và lưu từng claim (preview kết quả + file evidences):
```
python scripts/retrieve_evidences.py \
  --corpus prepared/warmup/corpus.csv --format csv \
  --claims prepared/warmup/valid.jsonl \
  --bm25 prepared/warmup/bm25.json \
  --out_dir outputs/warmup \
  --topk 10
```

Gợi ý: `--topk` là số evidences sau hợp nhất (RRF) và rerank.

## 3) Xây Hybrid “thủ công” (Notebook/Colab)
Khi muốn tùy biến sâu (weighted-sum, đánh giá nhanh):

- Cài gói cần thêm (nếu chạy Colab):
```
!pip install -q sentence-transformers faiss-cpu rank-bm25 underthesea pandas tqdm
```

- Tải corpus và xây BM25:
```
import pandas as pd
from rank_bm25 import BM25Okapi

# Đọc corpus
corpus_path = "prepared/warmup/corpus.csv"
df = pd.read_csv(corpus_path)
doc_ids = df["doc_id"].astype(str).tolist()
docs = df["text"].astype(str).tolist()

# Tokenizer TV (fallback là tách theo khoảng trắng)
try:
    from underthesea import word_tokenize
    tokenize = lambda s: word_tokenize(s, format="text").split()
except Exception:
    tokenize = lambda s: s.split()

tokenized_docs = [tokenize(d) for d in docs]
bm25 = BM25Okapi(tokenized_docs)
```

- Xây dense + FAISS (khuyên dùng `intfloat/multilingual-e5-base`):
```
import numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base", device=("cuda" if torch.cuda.is_available() else "cpu"))

def e5_doc_embed(texts, bs=64):
    texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, batch_size=bs, show_progress_bar=True, normalize_embeddings=True).astype("float32")

doc_embs = e5_doc_embed(docs)
index = faiss.IndexFlatIP(doc_embs.shape[1])
index.add(doc_embs)

def e5_query_embed(queries):
    queries = [f"query: {q}" for q in queries]
    return model.encode(queries, batch_size=32, show_progress_bar=False, normalize_embeddings=True).astype("float32")
```

- Hợp nhất Hybrid (RRF hoặc weighted-sum):
```
import numpy as np

def retrieve_hybrid(query, topk=10, method="rrf", alpha=0.6, bm25_k=200, dense_k=200):
    # Sparse
    b_scores = bm25.get_scores(tokenize(query))
    b_top = np.argsort(-b_scores)[:bm25_k]
    b_rank = {int(i): r for r, i in enumerate(b_top, 1)}
    # Dense
    qv = e5_query_embed([query])
    ds, di = index.search(qv, dense_k)
    ds, di = ds[0], di[0]
    d_rank = {int(i): r for r, i in enumerate(di, 1)}
    # Fusion
    cand = set(b_top.tolist()) | set(di.tolist())
    fused = {}
    if method == "rrf":
        K = 60
        for i in cand:
            fused[i] = 1/(K + b_rank.get(int(i), 10**9)) + 1/(K + d_rank.get(int(i), 10**9))
    else:  # weighted-sum với min-max norm
        bmin, bmax = float(b_scores.min()), float(b_scores.max())
        norm_b = lambda s: (s - bmin)/(bmax - bmin + 1e-9)
        dmap = {int(di[j]): float(ds[j]) for j in range(len(di))}
        dvals = list(dmap.values()) if dmap else [0.0]
        dmin, dmax = float(min(dvals)), float(max(dvals))
        norm_d = lambda s: (s - dmin)/(dmax - dmin + 1e-9)
        for i in cand:
            sb = norm_b(float(b_scores[int(i)]))
            sd = norm_d(float(dmap.get(int(i), dmin)))
            fused[i] = alpha*sd + (1 - alpha)*sb
    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [(doc_ids[i], docs[i]) for i, _ in top]
```

- Đánh giá nhanh recall/precision@K nếu có `prepared/warmup/valid.jsonl`:
```
import json, tqdm

def load_valid(path):
    items = []
    for line in open(path, "r", encoding="utf-8"):
        if not line.strip():
            continue
        j = json.loads(line)
        claim = j.get("claim") or j.get("statement")
        gold = j.get("gold_evidence_ids") or []
        if claim and gold:
            items.append((claim, set(map(str, gold))))
    return items

valid_items = load_valid("prepared/warmup/valid.jsonl")

def eval_retrieval(valid_items, topk=10, **kwargs):
    precs, recs = [], []
    for claim, gold in tqdm.tqdm(valid_items):
        pred_ids = [i for i,_ in retrieve_hybrid(claim, topk=topk, **kwargs)]
        inter = set(pred_ids) & set(gold)
        precs.append(len(inter)/max(1,len(pred_ids)))
        recs.append(len(inter)/max(1,len(gold)))
    return {
        "mean_precision@k": round(sum(precs)/len(precs),4),
        "mean_recall@k": round(sum(recs)/len(recs),4)
    }

# So sánh
eval_retrieval(valid_items, topk=10, method="rrf", bm25_k=200, dense_k=200)
```

## 4) Lưu/Load Index (tiết kiệm thời gian)
```
import json, pickle, os
os.makedirs("ckpt/indexes", exist_ok=True)

# FAISS + mapping ids
import faiss, numpy as np
faiss.write_index(index, "ckpt/indexes/dense.faiss")
np.save("ckpt/indexes/doc_embs.npy", doc_embs)
with open("ckpt/indexes/doc_ids.json", "w", encoding="utf-8") as f:
    json.dump(doc_ids, f, ensure_ascii=False)

# BM25 (pickle)
with open("ckpt/indexes/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)
```

## 5) Tinh Chỉnh & Mẹo
- RRF: ổn định, ít cần chuẩn hoá; bắt đầu `bm25_k=dense_k=200`, `topk=10/20`.
- Weighted-sum: thử `alpha=0.6` (ưu tiên dense), tinh chỉnh theo valid.
- Tiết kiệm tải model trên Colab: đặt cache HF `os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"` trước khi import model.
- Thiếu `faiss-cpu`: cài `pip install faiss-cpu` (Colab thường có sẵn package tương thích).
- Hết VRAM/RAM: giảm `--topk`, chiều dài câu, batch size khi encode.
- Ký tự tiếng Việt: dữ liệu cần UTF‑8; đường dẫn không có ký tự đặc biệt nếu gặp lỗi I/O.

## Câu Hỏi Thường Gặp
- Tôi đã chạy xong `prepare_ise_kvjson.py`, tiếp theo chạy danh sách bước “Chuẩn bị corpus → BM25 → Dense → Hybrid → Đánh giá → Lưu index” là đủ chưa?
  - Có. Bạn có thể chọn đường tắt dùng `scripts/run_pipeline.py` (đã kết hợp Hybrid + RRF) hoặc làm thủ công theo mục 3 để tuỳ chỉnh và đánh giá chi tiết.

---
Nếu cần, có thể bổ sung script đánh giá nhanh `scripts/eval_retrieval.py` để tự động tính precision/recall@K trên `valid.jsonl`.

