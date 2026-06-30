# Data Directory Documentation (`data/`)

Thư mục này chứa toàn bộ dataset phục vụ cho hệ thống RAG, từ dữ liệu thô thu thập được từ các nguồn trực tuyến cho đến dữ liệu đã qua tiền xử lý, định dạng và lưu trữ.

## 📁 Cấu trúc và mô tả chi tiết các thư mục con

### 1. `raw/`
* **Mô tả**: Nơi chứa dữ liệu thô (raw data) ban đầu được import trực tiếp hoặc tải từ các nguồn chính thống mà chưa qua bất kỳ bước làm sạch nào.
* **Nội dung**: Hiện tại, thư mục này chỉ chứa dataset gốc được download từ [Cổng thông tin điện tử pháp điển](https://phapdien.moj.gov.vn/Pages/home.aspx). Dataset `tmquan/vbpl-vn` có thể được download trực tiếp từ [Hugging Face](https://huggingface.co/datasets/tmquan/vbpl-vn).

### 2. `crawl/`
* **Mô tả**: Lưu trữ dữ liệu thu thập trực tiếp (crawler) từ trang [vbpl.vn](https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/1), dựa trên mẫu biểu từ `tmquan/vbpl-vn`.
* **Mục đích**: Lưu lại dữ liệu gốc để tái tiền xử lý khi cần thiết.
* **Lưu ý**: Hiện tại chưa có tính năng cập nhật dữ liệu tự động, do đó cần phải chạy thủ công tại Jupyter Notebook `../Set_VBPL_crawl.ipynb`.

### 3. `processed/`
* **Mô tả**: Dữ liệu đã qua quy trình tiền xử lý (làm sạch văn bản, chuẩn hóa cấu trúc phân cấp Điều/Khoản/Điểm, bóc tách các trường thông tin quan trọng) của Jupyter Notebooks ([Set_VBPL_processed.ipynb](../Set_VBPL_processed.ipynb), [Set_PhapDien_preprocess.ipynb](../Set_PhapDien_preprocess.ipynb)). Cấu trúc chi tiết được trình bày bên dưới.   

### 4. `qa/`
* **Mô tả**: Bộ dữ liệu câu hỏi - đáp (Question - Answer pairs) mẫu được xây dựng thủ công hoặc tự động sinh, dùng cho việc kiểm tra chất lượng tìm kiếm (Retrieval Evaluation) và kiểm định câu trả lời của mô hình ngôn ngữ lớn (LLM Evaluation).

---

## 📊 Cấu trúc dữ liệu đã tiền xử lý (`processed/`)

Dữ liệu sau khi xử lý được lưu trữ dưới dạng JSON/Parquet với cấu trúc phẳng (flat) biểu diễn thông tin chi tiết từng **Điều** (Article) trong văn bản pháp luật. Dưới đây là chi tiết các trường thông tin dựa trên file dữ liệu mẫu [SAMPLE_PROCESSED.json](processed/SAMPLE_PROCESSED.json):

| Tên trường (Key) | Kiểu dữ liệu | Mô tả | Ví dụ từ mẫu |
| :--- | :--- | :--- | :--- |
| `docs_code` | `String` | Số hiệu/ký hiệu của văn bản pháp quy. | `"139/2002/QĐ-TTG"` |
| `docs_title` | `String` | Tên đầy đủ hoặc tiêu đề của văn bản pháp quy. | `"Quyết định số 139/2002/QĐ-TTg"` |
| `article_index` | `String` | Số thứ tự/chỉ mục của Điều. | `"Điều 1"`, `"Điều 2"` |
| `article_title` | `String` | Tiêu đề đầy đủ của Điều hoặc câu dẫn mở đầu Điều. | `"Điều 2.2.QĐ.2.2. Đối tượng được hưởng chế độ hỗ trợ khám, chữa bệnh theo Quyết định này gồm:"` |
| `source_note_text` | `String` | Ghi chú về nguồn gốc, ngày ban hành, hiệu lực và các văn bản sửa đổi, bổ sung liên quan. | `"(Điều 1 Quyết định số 139/2002/QĐ-TTg Về việc khám, chữa bệnh...)"` |
| `source_links` | `String` | Chuỗi URL liên kết tham chiếu gốc trên VBPL (href). | `"http://vbpl.vn/..."` |
| `topic_title` | `String` | Lĩnh vực lớn hoặc chủ đề chính của văn bản. | `"Bảo hiểm"` |
| `subject_title` | `String` | Chủ đề cụ thể hơn/chủ đề con của văn bản. | `"Bảo hiểm y tế"` |
| `content_text` | `Dict` | Nội dung văn bản của Điều. Định dạng thay đổi tùy thuộc vào cấu trúc điều (đơn đoạn hoặc phân đoạn khoản/điểm). | (Xem cấu trúc chi tiết bên dưới) |
| `references` | `List[Dict]` | Danh sách các văn bản có liên quan đến Điều hiện tại (nếu dataset thô gốc có cung cấp trường thông tin này) | `['id': 1, 'docs_code': '24/LĐ-NĐ']`|
| `content_word_count` | `Integer` | Số lượng từ trong phần nội dung của Điều. | `38` hoặc `124` |
| `content_clause_count` | `Integer` | Số lượng khoản hoặc phân đoạn con được bóc tách từ Điều. | `0` hoặc `4` |
| `__index_level_0__` | `Integer` | Chỉ mục gốc (loc) của dòng dữ liệu trong DataFrame của pandas. | `977` |

### Cấu trúc chi tiết của trường `content_text`

Trường `content_text` biểu diễn nội dung phân cấp của một **Điều luật**. 
Cấu trúc phân cấp của văn bản pháp quy Việt Nam thường đi theo thứ tự: **Điều -> Khoản -> Điểm**. 

Hệ thống biểu diễn cấu trúc này dưới dạng một cấu trúc cây đệ quy (recursive tree structure):
* **Nút lá (Leaf Node)**: Nếu một cấp (Điều, Khoản hoặc Điểm) không chứa bất kỳ phân đoạn/cấp con nào bên dưới nó, toàn bộ nội dung văn bản của cấp đó được đặt trong một dictionary có khóa `"text"` duy nhất (không có các khóa `title` hay `content`).
* **Nút nhánh (Branch Node)**: Nếu một cấp chứa các cấp con trực thuộc (ví dụ: Điều chứa các Khoản con, hoặc Khoản chứa các Điểm con):
  - Khóa `"title"` chứa tiêu đề hoặc câu dẫn mở đầu của cấp đó (phần nội dung văn bản đứng trước các cấp con trực thuộc).
  - Khóa `"content"` chứa một danh sách (List) các dictionary con đại diện cho cấp tiếp theo (Khoản hoặc Điểm), mỗi phần tử con lại được parse đệ quy theo cùng quy tắc này.

#### Các ví dụ minh họa:

**1. Điều luật không có Khoản con (Nút lá ở cấp Điều)**
```json
{
  "text": "Cáccơ sở khám, chữa bệnh Nhà nước từ trạm y tế xã đến bệnh viện và viện có giườngbệnh tuyến Trung ương thực hiện chế độ khám, chữa bệnh cho người nghèo theo quyđịnh tại Quyết định này."
}
```

**2. Điều luật có Khoản con, nhưng Khoản không có Điểm con (Điều là nút nhánh, Khoản là nút lá)**
```json
{
  "title": "Điều 2.2.QĐ.2.2. Đối tượng được hưởng chế độ hỗ trợ khám, chữa bệnh theo Quyết định này gồm:",
  "content": [
    {
      "text": "1. Người thuộc hộ nghèo theo quy định hiện hành của Thủ tướng Chính phủ về chuẩn hộ nghèo."
    },
    {
      "text": "2. Đồng bào dân tộc thiểu số đang sinh sống ở xã, phường, thị trấn thuộc vùng khó khăn theo quy định tại Quyết định số 30/2007/QĐ-TTg ngày 05 tháng 3 năm 2007 của Thủ tướng Chính phủ."
    }
  ]
}
```

**3. Điều luật có Khoản con, và Khoản có Điểm con (Điều và Khoản là nút nhánh, Điểm là nút lá)**
```json
{
  "title": "Điều 3. Phương thức hỗ trợ chi phí khám, chữa bệnh:",
  "content": [
    {
      "title": "1. Hỗ trợ tiền ăn cho các đối tượng quy định tại Khoản 1 và 2 Điều 2 khi điều trị nội trú:",
      "content": [
        {
          "text": "a) Đối với người thuộc hộ nghèo..."
        },
        {
          "text": "b) Đối với đồng bào dân tộc thiểu số..."
        }
      ]
    }
  ]
}
```

---

## 💾 Lưu ý về Đọc/Ghi dữ liệu (Serialization & Deserialization)

Do đặc thù lưu trữ của các định dạng bảng biểu (như lưu trữ ra DataFrame dưới định dạng `.parquet`, `.csv` hoặc `.json`), các trường có cấu trúc phức tạp dạng Object/Dict hoặc List của Object như **`content_text`** và **`references`** cần được xử lý chuyển đổi dữ liệu:

* **Khi lưu dữ liệu (Serialization)**: Bắt buộc phải chuyển đổi cấu trúc Object/List sang dạng chuỗi ký tự JSON (JSON string) bằng phương thức `json.dumps()` trước khi tiến hành lưu để đảm bảo dữ liệu ghi vào file được toàn vẹn cấu trúc và tránh các lỗi định dạng của thư viện pandas/pyarrow.
* **Khi đọc dữ liệu (Deserialization)**: Bắt buộc phải giải mã ngược lại từ chuỗi JSON thành các đối tượng Python gốc (Dict/List) tương ứng bằng phương thức `json.loads()` trước khi thực hiện các tác vụ truy vấn hoặc tiền xử lý khác.

> [!TIP]
> Chi tiết các bước cài đặt và xử lý ngoại lệ an toàn cho quá trình đọc/ghi này đã được định nghĩa tại hai hàm `save()` và `load()` trong file source code [preprocess_parquet.py](../src/preprocess_parquet.py).
