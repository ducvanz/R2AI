# Legal RAG Vietnam — Provision-Level Neural Reranking

Hệ thống hỏi đáp pháp luật doanh nghiệp Việt Nam với **Neural Reranking ở mức Provision**, được lấy cảm hứng từ bài báo:
> *"Neural reranking for UK statutory retrieval: Provision-level evaluation and an open distilled model"* — Artificial Intelligence and Law, 2026. https://link.springer.com/article/10.1007/s10506-025-09501-6

---

## Kiến trúc hệ thống

```
Câu hỏi người dùng
        │
        ▼
┌─────────────────────┐
│  Stage 1: Retrieval  │  BM25 (sparse) + Dense (sentence-transformers)
│  Top-K candidates    │  → Trả về ~50–100 provision candidates
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2: Reranking  │  Cross-Encoder Neural Reranker (mô hình multilingual)
│  Provision-level     │  → Score từng (query, provision) pair
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 3: Output     │  Top-5 provisions với law_id + article_title
└─────────────────────┘
```

---

## Cấu trúc thư mục

```
legal-rag-vn/
├── data/
│   └── data.parquet               # Dữ liệu pháp điển (input)
├── src/
│   ├── 01_preprocess.py           # Tiền xử lý & lập chỉ mục dữ liệu
│   ├── 02_retriever.py            # Stage 1: BM25 + Dense Retrieval
│   ├── 03_reranker.py             # Stage 2: Neural Cross-Encoder Reranking
│   ├── 04_pipeline.py             # Pipeline đầy đủ: query → results
│   ├── 05_evaluate.py             # Đánh giá nDCG, MRR (nếu có ground truth)
│   └── utils.py                   # Hàm tiện ích chung
├── models/                        # Cache mô hình tải về
├── results/                       # Kết quả truy vấn
├── requirements.txt
└── README.md
```

---

## Pipeline chi tiết — Thứ tự chạy

### Bước 1 — Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Bước 2 — Đặt dữ liệu
```
Đặt file data.parquet vào thư mục data/
```

### Bước 3 — Tiền xử lý & Index
```bash
python src/01_preprocess.py
```
- Đọc `data/data.parquet`
- Tạo `provision_text` kết hợp title + content cho retrieval
- Xây dựng **BM25 index** (lưu `data/bm25_index.pkl`)
- Tạo **Dense embeddings** bằng `intfloat/multilingual-e5-base` (lưu `data/dense_index.npy`)

### Bước 4 — Kiểm tra retriever đơn lẻ
```bash
python src/02_retriever.py --query "Doanh nghiệp nhỏ và vừa cần điều kiện gì?"
```

### Bước 5 — Chạy pipeline đầy đủ (Retrieval + Reranking)
```bash
python src/04_pipeline.py --query "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ?"
```

**Output mẫu:**
```
Top provisions:
1. [04/2017/QH14] Luật Hỗ trợ DNNVV | Điều 4 | score=0.94
2. [04/2017/QH14] Luật Hỗ trợ DNNVV | Điều 5 | score=0.91
3. [80/2021/NĐ-CP] Nghị định hướng dẫn | Điều 5 | score=0.87
```

### Bước 6 (Tuỳ chọn) — Đánh giá nDCG & MRR
```bash
python src/05_evaluate.py --eval_file data/eval_queries.json
```

---

## Các mô hình sử dụng

| Vai trò | Mô hình | Ghi chú |
|---|---|---|
| Dense Retrieval | `intfloat/multilingual-e5-base` | Tối ưu tiếng Việt, miễn phí |
| Neural Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder, nhanh |
| Reranker nâng cao | `BAAI/bge-reranker-v2-m3` | Multilingual, mạnh hơn |
| LLM (tuỳ chọn) | `gemma3:4b` qua Ollama | Giải thích kết quả |

---

## Thiết kế theo bài báo

Bài báo đề xuất pipeline **2 giai đoạn**:
1. **First-stage retrieval** (BM25 hoặc dense) → lấy Top-K ứng viên
2. **Neural reranking** (cross-encoder) → xếp hạng lại chính xác hơn

Khác với bài báo (UK legislation), hệ thống này:
- Thêm **Hybrid Retrieval** (BM25 + Dense, RRF fusion) để tăng recall cho tiếng Việt
- Sử dụng mô hình **multilingual** thay vì English-only
- Đánh giá ở mức **provision** (điều luật cụ thể), không phải document

---

## Yêu cầu hệ thống

- Python 3.9+
- RAM: tối thiểu 8GB (16GB khuyến nghị cho toàn bộ dataset)
- GPU: tuỳ chọn (CPU vẫn chạy được, chậm hơn ~5–10x)
- Ollama (nếu dùng gemma3:4b cho LLM step)