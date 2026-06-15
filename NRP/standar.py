"""
normalize_results.py
--------------------
Chuẩn hoá kết quả tra cứu pháp điển sang định dạng Q&A chuẩn.

Đầu vào : input.json  – danh sách các object tìm kiếm (mảng JSON)
Đầu ra  : output.json – danh sách Q&A chuẩn

Sử dụng model local Ollama (gemma3:4b) để tổng hợp câu trả lời.

Yêu cầu:
    - Ollama đang chạy:  ollama serve
    - Model đã pull:     ollama pull gemma3:4b

"""

import json
import re
import sys
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Cấu hình Ollama
# ---------------------------------------------------------------------------
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

# ---------------------------------------------------------------------------
# Registry động: tự xây dựng từ dữ liệu đầu vào, không hardcode tên luật
#
# Tên đầy đủ được ghép tự động:
#   loại văn bản (từ đuôi mã)  +  mã số  +  subject_title (từ dữ liệu)
#
# Ví dụ:
#   code="80/2021/NĐ-CP", subject_title="Hỗ trợ doanh nghiệp nhỏ và vừa"
#   → "Nghị định 80/2021/NĐ-CP Hỗ trợ doanh nghiệp nhỏ và vừa"
# ---------------------------------------------------------------------------

# Quy tắc nhận diện loại văn bản theo đuôi mã số
_CODE_TYPE_RULES: list[tuple[str, str]] = [
    ("TTLT", "Thông tư liên tịch"),   # phải trước "TT" để không bị nuốt
    ("QH",   "Luật"),
    ("NĐ",   "Nghị định"),
    ("TT",   "Thông tư"),
    ("QĐ",   "Quyết định"),
    ("CT",   "Chỉ thị"),
    ("CV",   "Công văn"),
]

# Cache động được điền ở bước scan trước khi xử lý
_law_registry: dict[str, str] = {}


def _detect_doc_type(code: str) -> str:
    """Nhận diện loại văn bản từ mã số."""
    upper = code.upper()
    for suffix, label in _CODE_TYPE_RULES:
        if suffix in upper:
            return label
    return "Văn bản"


def _register_law(code: str, subject_title: str = "") -> None:
    """Đăng ký văn bản vào registry (chỉ lần đầu gặp)."""
    if not code or code in _law_registry:
        return
    doc_type = _detect_doc_type(code)
    name = f"{doc_type} {code}"
    if subject_title:
        name = f"{name} {subject_title.strip()}"
    _law_registry[code] = name


def law_full_name(code: str) -> str:
    """Trả về tên đầy đủ từ registry; tự sinh nếu chưa có."""
    if code in _law_registry:
        return _law_registry[code]
    doc_type = _detect_doc_type(code)
    return f"{doc_type} {code}"


def scan_and_register(data: list[dict]) -> None:
    """
    Duyệt toàn bộ dữ liệu một lần để xây dựng registry trước khi xử lý.
    Lấy subject_title từ field cùng tên trong mỗi result.
    """
    for search_obj in data:
        for r in search_obj.get("results", []):
            code          = r.get("law_code", "").strip()
            subject_title = r.get("subject_title", "").strip()
            _register_law(code, subject_title)

    print(f"  → Đã đăng ký {len(_law_registry)} văn bản pháp luật từ dữ liệu.")


# ---------------------------------------------------------------------------
# Trích số điều từ article_title
# ---------------------------------------------------------------------------
_DIEU_FALLBACK_RE = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)


def extract_article_number(article_title: str) -> str:
    """
    Ví dụ:
      'Điều 12.3.LQ.5. Nguyên tắc hỗ trợ...' → 'Điều 5'
      'Điều 12.3.NĐ.3.16. Điều kiện vay vốn'  → 'Điều 16'
    """
    parts = article_title.split(".")
    for part in reversed(parts):
        stripped = part.strip().split()[0] if part.strip() else ""
        if stripped.isdigit():
            return f"Điều {stripped}"
    m = _DIEU_FALLBACK_RE.search(article_title)
    return f"Điều {m.group(1)}" if m else article_title


# ---------------------------------------------------------------------------
# Gọi Ollama local để tổng hợp câu trả lời
# ---------------------------------------------------------------------------

def synthesize_answer(question: str, passages: list[dict]) -> str:
    """Dùng gemma3:4b qua Ollama để tổng hợp câu trả lời từ các đoạn pháp lý."""
    k=0
    context_parts = []
    for p in passages:
        k+=1
        if (k> 5):
            break
        context_parts.append(
            f"[{p['law_code']} – {p['article_title']}]\n{p['content_text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "Bạn là chuyên gia tư vấn luật của doanh nghiệp. "
        "Dựa trên các đoạn trích pháp luật dưới đây, hãy trả lời câu hỏi "
        "Chỉ trả lời nội dung, không thêm lời mở đầu hay kết thúc.\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Các đoạn trích pháp luật:\n{context}\n\n"
        "Yêu cầu:\n"
        "- Trả lời trực tiếp câu hỏi, không lặp lại câu hỏi.\n"
        "- Đề cập số điều / nghị định khi cần thiết để người đọc tra cứu.\n"
        "- Trẳ lời ngắn gọn, chỉ trả lời nội dung trực tiếp hỗ trợ cho người hỏi, bỏ qua các thông tin không liên quan.\n"
    )

    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Không thể kết nối Ollama tại {OLLAMA_URL}.\n"
            "Hãy chắc chắn Ollama đang chạy: ollama serve\n"
            f"Chi tiết lỗi: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Xây dựng một bản ghi Q&A từ một object tìm kiếm
# ---------------------------------------------------------------------------

def build_qa_record(idx: int, search_obj: dict) -> dict:
    question = search_obj.get("query", "").strip()
    results  = search_obj.get("results", [])

    # --- relevant_docs (deduplicated, giữ thứ tự xuất hiện) -----------------
    seen_docs: set[str] = set()
    relevant_docs: list[str] = []
    for r in results:
        code = r.get("law_code", "").strip()
        if code and code not in seen_docs:
            seen_docs.add(code)
            relevant_docs.append(f"{code}|{law_full_name(code)}")

    # --- relevant_articles ---------------------------------------------------
    seen_arts: set[str] = set()
    relevant_articles: list[str] = []
    for r in results:
        code   = r.get("law_code", "").strip()
        title  = r.get("article_title", "").strip()
        art_no = extract_article_number(title)
        key    = f"{code}|{art_no}"
        if key not in seen_arts:
            seen_arts.add(key)
            relevant_articles.append(f"{code}|{law_full_name(code)}|{art_no}")

    # --- answer (tổng hợp bằng gemma3:4b qua Ollama) ------------------------
    passages = [
        {
            "law_code":      r.get("law_code", ""),
            "article_title": r.get("article_title", ""),
            "content_text":  r.get("content_text", ""),
        }
        for r in results
        if r.get("content_text")
    ]
    answer = synthesize_answer(question, passages) if passages else ""

    return {
        "id":                idx,
        "question":          question,
        "answer":            answer,
        "relevant_docs":     relevant_docs,
        "relevant_articles": relevant_articles,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "../results/all.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "results.json"

    print(f"Đọc dữ liệu từ : {input_file}")
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    # Hỗ trợ cả dạng list và dict đơn
    if isinstance(data, dict):
        data = [data]

    # Bước 1: scan toàn bộ để xây registry tên luật động
    print("Quét văn bản pháp luật trong dữ liệu...")
    scan_and_register(data)

    print(f"Tổng số câu hỏi : {len(data)}")
    print(f"Model sử dụng   : {OLLAMA_MODEL} (Ollama local)\n")
    k = 0

    output: list[dict] = []
    for i, search_obj in enumerate(data, start=1):
        if i<507:
            continue
        query_preview = search_obj.get("query", "")[:70]
        print(f"[{i}/{len(data)}] {query_preview}...")

        record = build_qa_record(i, search_obj)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    print(f"\nHoàn thành! Kết quả đã được ghi vào: {output_file}")


if __name__ == "__main__":
    main()