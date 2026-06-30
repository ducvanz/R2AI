# Legal RAG Vietnam — Provision-Level Neural Reranking

Dự án phát triển hệ thống hỏi đáp (QA) tự động cho văn bản pháp luật doanh nghiệp Việt Nam, áp dụng mô hình **Retrieval-Augmented Generation (RAG)** kết hợp với **Neural Reranking ở mức điều/khoản (Provision-Level)**. 

---

## 🏗️ Kiến trúc tổng quan hệ thống (System Architecture)

Dưới đây là luồng xử lý của hệ thống RAG từ lúc nhận văn bản pháp luật thô cho đến khi sinh câu trả lời cho người dùng:

```text
       Văn bản Pháp luật (Legal Documents)
                       │
                       ▼
       Chia nhỏ văn bản (Chunks/Clauses)
                       │
                       ▼
             ┌───────────────────┐
             │ BM25 & Dense E5   │  ◄── (Recall - Tìm kiếm sơ bộ)
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │    RRF Fusion     │  ◄── (Hybrid Scoring - Kết hợp điểm số)
             └─────────┬─────────┘
                       │
                       ▼
            Top 100 as Recall Candidates Pool
                       │
                       ▼
             ┌───────────────────┐
             │   Cross-Encoder   │  ◄── (Rerank - Tái xếp hạng chính xác)
             └─────────┬─────────┘
                       │
                       ▼
             Normalizer & Selection    ◄── (Threshold or Top 5)
                       │
                       ▼
             ┌───────────────────┐
             │   Generative LLM  │  ◄── (QA - Tạo câu trả lời cuối cùng)
             └───────────────────┘
```

---

## 📁 Cấu trúc thư mục dự án (Project Directory Structure)

Mô hình phân bổ mã nguồn và dữ liệu thực tế trong project (đã loại bỏ các thư mục tạm và không cần thiết):

```text
.
├── data/                   # Thư mục lưu trữ dữ liệu pháp luật
│   ├── crawl/                  # Dữ liệu thu thập từ các trang văn bản pháp luật
│   ├── processed/              # Dữ liệu đã qua làm sạch và tiền xử lý
│   ├── qa/                     # Bộ dữ liệu câu hỏi - đáp (Question-Answering) để kiểm thử
│   └── raw/                    # Dữ liệu thô ban đầu
│
├── results/                # Kết quả thực nghiệm và đánh giá pipeline
│   └── runs/                   # Lưu tạm kết quả đầu ra
│
├── saved_model/            # Lưu trữ các mô hình và chỉ mục tìm kiếm cục bộ
│   ├── phapdien/               # Cơ sở dữ liệu và chỉ mục cho Bộ Pháp điển
│   └── vbpl/                   # Cơ sở dữ liệu và chỉ mục cho Văn bản Pháp luật (VBPL)
│    
└── src/                        # Mã nguồn triển khai các mô hình và pipeline
    ├── preprocess_parquet.py      # Tiền xử lý dữ liệu sang định dạng Parquet
    ├── data_presentation.py       # Các hàm hiển thị và format kết quả
    ├── recall_retriever.py        # Module tìm kiếm sơ bộ (Recall Retrieval: BM25 & Dense)
    ├── precision_retriever.py     # Module tái xếp hạng độ chính xác cao (Precision Reranking & HyDE)
    ├── retrieval_pipeline.py      # Pipeline tích hợp
    └── visualize.py               # Trực quan hóa kết quả và vẽ biểu đồ hiệu năng

```

---

## ⚙️ Hướng dẫn cài đặt & Chạy dự án

### 1. Cài đặt các thư viện cần thiết
Bạn có thể cài đặt toàn bộ dependencies thông qua file `requirements.txt` :
Lưu ý: file này chỉ dành cho hệ điều hành Ubuntu.
```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công theo từng nhóm công cụ chính dưới đây:
```bash
# Quản lý và xử lý dữ liệu
pip install pandas numpy beautifulsoup4 lxml fastparquet pyarrow

# Học máy và Tìm kiếm thông tin
pip install torch transformers sentence-transformers rank-bm25 faiss-cpu

# LLM local và Tiện ích khác
pip install llama-cpp-python tqdm ipykernel
```

> [!NOTE]
> * Chi tiết về các module mã nguồn trong `src/` và cách sử dụng các lớp/hàm được mô tả cụ thể trong [src/README.md](src/README.md).
> * Chi tiết về cấu trúc dữ liệu thô, dữ liệu sau cào và các bước xử lý dữ liệu trong `data/` được mô tả cụ thể trong [data/README.md](data/README.md).

---

## 📓 Danh sách các Notebook hiện có

Dưới đây là 4 notebook chính được sử dụng để xây dựng và thử nghiệm hệ thống. Hãy điền mô tả ngắn cho từng notebook bên dưới:

1. **`Set_PhapDien_preprocess.ipynb`**
   * *Mô tả:* Xử lý định dạng sau khi tải về cho dataset Pháp Điển. Không cần chạy lại nếu đã xử lý xong.
2. **`Set_VBPL_crawl.ipynb`**
   * *Mô tả:* Crawl dữ liệu và xử lý định dạng thô cho tập dữ liệu VBPL dựa trên dataset `tmquan/vpbpl-vn`.
3. **`Test_Retrieval_pipeline.ipynb`**
   * *Mô tả:* Prototype thử nghiệm pipeline RAG trên môi trường local..
4. **`Kaggle_Retrieval_pipeline.ipynb`**
   * *Mô tả:* Phiên bản pipeline chạy trực tiếp trên Kaggle với GPU. Hãy đảm bảo dataset đã được upload dữ liệu trong thư mục `src/` và `data/`.

---

## 🛠️ Công cụ sử dụng (Dependencies)

Dự án được xây dựng dựa trên các công cụ và thư viện mã nguồn mở phổ biến:

* **Quản lý dữ liệu**:
  * `pandas`: Thư viện xử lý và phân tích dữ liệu dạng bảng.
  * `numpy`: Hỗ trợ tính toán số học và ma trận hiệu năng cao.
* **Xử lý dữ liệu**:
  * `json`: Đọc/ghi cấu trúc dữ liệu JSON dùng để trao đổi kết quả.
  * `Beautiful Soup (bs4)`: Phân tích cú pháp và trích xuất dữ liệu từ trang HTML khi crawl.
* **Xây dựng và sử dụng mô hình**:
  * `transformers` (bao gồm `Sentence Transformers`, `Cross Encoder`, `AutoTokenizer`): Thư viện lõi từ Hugging Face phục vụ cho việc nhúng từ và chạy reranker.
  * `llama-cpp-python`: Hỗ trợ chạy các mô hình ngôn ngữ lớn (LLM) định dạng GGUF tối ưu hóa trên khi chạy trên Kaggle.
  * `bm25s`: Triển khai thuật toán tìm kiếm từ khóa BM25.
  * `FAISS`: Thư viện tìm kiếm vector tương đồng mật độ cao cực nhanh của Meta.

* Cùng một số dependencies nhỏ khác. Như `re` (xử lý regex/ string pattern), `ABC` (viết interfaces),... 
---

## 🤖 Các mô hình ngôn ngữ & Nhúng đã sử dụng (Models Used)

Hệ thống RAG sử dụng các mô hình pre-trained chuyên biệt để đạt hiệu quả tối ưu cho tiếng Việt:

1. **Bi-Encoder (Recall/Dense Retrieval)**: `"keepitreal/vietnamese-sbert"`
   * Dùng để chuyển đổi các đoạn văn bản (chunks/provisions) thành vector nhúng nhằm thực hiện tìm kiếm ngữ nghĩa sơ bộ.
2. **Reranker (Precision Reranking)**: `"BAAI/bge-reranker-v2-m3"`
   * Mô hình Cross-Encoder đa ngôn ngữ mạnh mẽ, dùng để chấm điểm độ liên quan trực tiếp giữa câu hỏi và danh sách văn bản gợi ý từ Recall layer.
3. **LLM cho HyDE (Hypothetical Document Embeddings)**: `Qwen 7B`
   * Sử dụng để sinh ra văn bản giả định từ câu hỏi gốc của người dùng nhằm tăng độ phủ (recall) khi tìm kiếm ngữ nghĩa.
4. **LLM Generator (Sinh câu trả lời QA)**: `N/A`

--- 

## 🤖 Thành viên tham gia (Contributors):
1. [Nguyễn Văn Thịnh](github.com/TNoLisme): Module QA sau khi retrieval hoàn tất (nếu có :v).
2. [Mai Đức Văn](github.com/ducvanz): Thiết kế pipeline và all-in LLM.
3. [Nguyễn Trường Sơn](github.com/Nostagi): xách nước bổ cam.