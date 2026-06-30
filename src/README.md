# Source Code Documentation (`src/`)

Thư mục này chứa mã nguồn chính của hệ thống RAG phục vụ tìm kiếm và trả lời câu hỏi pháp luật. Dưới đây là chi tiết chức năng của từng file mã nguồn:

## 📋 Danh sách các file và chức năng

### 1. `retrieval_pipeline.py`
* **Vai trò**: Điểm tích hợp chính của toàn bộ hệ thống RAG (luồng kết hợp Recall và Precision).
* **Chức năng**:  
  * Định nghĩa lớp điều phối luồng tìm kiếm từ khâu nhận câu hỏi, gọi các bộ recall (BM25, Dense).
  * Trực tiếp thực hiện RRF Fusion với các cột điểm được trả về từ `recall_retriever` 
  * Gửi danh sách ứng viên `top_re_rank` qua bộ Cross-Encoder Rerank để xếp hạng lại và trả về ngữ cảnh tốt nhất.
* **Bổ sung**:
  * Module này cũng kèm theo các interface định hình các thành phần có trong cấu trúc hệ thống.

### 2. `recall_retriever.py`
* **Vai trò**: Tìm kiếm sơ bộ với độ phủ cao (Recall Retrieval).
* **Chức năng**:
  * Tích hợp thuật toán BM25 truyền thống (`bm25s`) để truy vấn từ khóa.
  * Tích hợp Dense Retrieval sử dụng mô hình Bi-Encoder  cùng cơ sở dữ liệu vector `FAISS` để tìm kiếm ngữ nghĩa.
  * Khuyến nghị sử dụng mô hình `keepitreal/vietnamese-sbert`.
  * Kết quả đầu ra dựa trên interface `RecallRetriever`, là một DataFrame chứa điểm và phụ chú cho từng văn bản trong corpus. Nếu văn bản không được tính điểm sẽ có giá trị NaN.

### 3. `precision_retriever.py`
* **Vai trò**: Tái xếp hạng độ chính xác cao (Precision Reranking).
* **Chức năng**:
  * Tải và cấu hình mô hình Cross-Encoder. Khuyến nghị sử dụng mô hình `BAAI/bge-reranker-v2-m3`.
  * Thực hiện việc tính toán điểm số tương tác trực tiếp (Query-Document interaction score) giữa câu hỏi và danh sách văn bản gợi ý để chọn lọc ra 3 kết quả tốt nhất.
  * Kết quả đầu ra dựa trên interface `PrecisionRetriever` là một DataFrame chứa điểm và phụ chú cho từng văn bản `top_re_rank` được gửi tới bởi `retrieval_pipeline`.
  * Tạm thời bao hàm cả module `HyDE Processor`.

### 4. `preprocess_parquet.py`
* **Vai trò**: Tiền xử lý dữ liệu.
* **Chức năng**:
  * Load và save dữ liệu từ các file định dạng JSON/CSV/PARQUET.
  * Một số utils khác cho tiền xử lý dữ liệu.
  * Thực hiện làm sạch văn bản, chuẩn hóa cấu trúc phân cấp (Điều, Khoản, Điểm) và lưu trữ dưới dạng định dạng cột Parquet (`.parquet`) tối ưu hóa tốc độ đọc ghi.

### 5. `data_presentation.py`
* **Vai trò**: Định dạng quản lý văn bản (corpus) và ánh xạ điểm số.
* **Chức năng**:
  * Format và lưu trữ corpus dưới dạng flatten array để sử dụng trực tiếp cho các module Retriever.
  * Tích hợp cấu trúc cây để hỗ trợ quản lý chunking.
  * Thực hiện ánh xạ từ node văn bản sang điểm số và ngược lại.

### 6. `visualize.py`
* **Vai trò**: Trực quan hóa dữ liệu và kết quả đánh giá.
* **Chức năng**:
  * Hỗ trợ vẽ các biểu đồ phân phối điểm số, độ dài câu hỏi/văn bản, hoặc so sánh hiệu năng (độ chính xác, thời gian phản hồi) giữa các cấu hình tìm kiếm khác nhau.
