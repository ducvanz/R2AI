# Legal RAG Vietnam — Provision-Level Neural Reranking

Hệ thống hỏi đáp pháp luật doanh nghiệp Việt Nam với **Neural Reranking ở mức Provision**, được lấy cảm hứng từ bài báo:
> *"Neural reranking for UK statutory retrieval: Provision-level evaluation and an open distilled model"* — Artificial Intelligence and Law, 2026. https://link.springer.com/article/10.1007/s10506-025-09501-6

---

## Kiến trúc hệ thống

```
 Legal Documents
        ↓
 Chunk (Clauses)
        ↓
 ┌───────────────┐
 │ BM25 or Dense │     # Recall Retrieval
 └───────────────┘
        ↓
     RRF Fusion        # Re-Rank ~ Tổng hợp các độ đo trước đó
        ↓
     Top 1000          # Candiates Pool
        ↓
 ┌───────────────┐
 │ Cross Encoder │     # Precision Retrieval
 └───────────────┘         
        ↓
      Top 7            # Final Contextual
        ↓
 ┌───────────────┐
 │ LLM Answering │     # Re-Rank ~ Precision Retrieval
 └───────────────┘  

```
---


## Cấu trúc thư mục
```
src
 ├── retrieval_pipeline
 |      ├── Retrieval Query         # wrapper cho truy vấn gốc
 |      ├── Retrieval Layer         # Interfaces cho các lớp BM25, Dense,...
 |      └── Retrieval Pipeline      # Pipeline cuối cùng
 |
 ├── recall_retriever
 |      ├── BM25 Layer        
 |      ├── Dense Layer
 |      └── RRF Fusion
 |
 └── precision_retriever
        └── CrossEncoder
```

** Lưu ý: [requirements.txt](requirements.txt) chỉ dành cho hệ điều hành Linux - Ubuntu =))

## Giải thích cơ chế:

### BM25 (Recall)

Cho:
    $$q : query \\ D : document$$
Thì:
    $$
    BM25(q,D)=\sum_{t \in q}​ IDF(t) \cdot \frac{f(t,D) \cdot (k_1 + 1)}{​(f(t,D) + k_1 (1−b+b\frac{|D|}{len(D).mean}​)}
    $$

Trong đó:
    $$
    f(t, D) : \text{tần suất term xuất hiện trong document}
    $$

### Dense Retrieval (Recall)

Sử dụng một mô hình Sentence Transformer (ví dụ như `intfloat/multilingual-e5-small`) để embedding document (chuyển toàn bộ một tài liệu thành chuỗi vector nhúng). Sau đó giảm chiều và lưu vào một Vector Database (Ở đây được lưu trữ và indexing bởi `FAISS`). Sau đó search trên cơ sở dữ liệu đó dựa trên $\text{cosine similarity}$.

Vấn đề nhận thấy:
- Các mô hình này không phản ánh nội dung văn bản đủ chi tiết. Điếm chỉ giao đồng trong khoảng $0.78 - 0.90$, trong đó $0.78$ là điểm sàn cho đoạn văn bản trống `""`, cho thấy bias lớn (không rõ do đặc điểm phương pháp hay do mô hình xử lý Tiếng Việt kém).
- Thời gian build/fit cho CSDL khá lâu. Cơ chế chỉ lọc lấy nội dung ở node lá (leaf) hiện tại làm mất nội dung của `title`.

### Hierarchy Corpus (Data Chunking):

- Ý tưởng: Phân lớp tài liệu (Điều luật `Article`) thành các form nhỏ hơn (Khoản `Clause` và Điểm `Point`). Nếu lý tưởng, khi chỉ xử lý phần văn bản ở các nhánh là `leaf`, kỹ thuật Retrieval có thể trỏ tới vị trí chính xác của phần nội dung được đánh giá cao (các trường `comment`), và normalize được độ dài văn bản (đối với `BM25`).
- Thực tế implement:
       - Bị mất thông tin `title`
       - Performance không cải thiện so với trước khi chunk.

