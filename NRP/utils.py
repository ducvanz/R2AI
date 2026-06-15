"""
utils.py — Hàm tiện ích chung cho toàn bộ pipeline
"""

import re
import unicodedata
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Provision Text Builder ─────────────────────────────────────────────────

def build_provision_text(row: dict, mode: str = "full") -> str:
    """
    Tạo văn bản provision để đưa vào retrieval / reranking.
    
    mode="full"   → title + chapter + source_note + content (dùng cho indexing)
    mode="short"  → title + content (dùng để hiển thị)
    mode="rerank" → title + content (dùng cho cross-encoder, ngắn gọn hơn)
    """
    parts = []
    
    if row.get("topic_title"):
        parts.append(f"Chủ đề: {row['topic_title']}")
    if row.get("article_title"):
        parts.append(f"Điều: {row['article_title']}")
    if mode == "full":
        if row.get("chapter_title"):
            parts.append(f"Chương: {row['chapter_title']}")
        if row.get("source_note_text"):
            parts.append(f"Nguồn: {row['source_note_text']}")
    if row.get("content_text"):
        parts.append(row["content_text"])
    
    return " | ".join(p.strip() for p in parts if p.strip())


def build_provision_id(row: dict) -> str:
    """
    Tạo ID định danh duy nhất cho một provision, dạng:
    subject_id::article_title
    """
    return f"{row.get('subject_id', '')}::{row.get('article_title', '')}"


def format_result(row: dict, score: float, rank: int) -> dict:
    """
    Format kết quả trả về người dùng.
    """
    # Trích xuất số luật từ source_note_text nếu có
    law_code = extract_law_code(row.get("source_note_text", ""))
    
    return {
        "rank": rank,
        "score": round(float(score), 4),
        "law_code": law_code,
        "topic_title": row.get("topic_title", ""),
        "subject_title": row.get("subject_title", ""),
        "article_title": row.get("article_title", ""),
        "chapter_title": row.get("chapter_title", ""),
        "content_text": row.get("content_text", ""),
        "source_note_text": row.get("source_note_text", ""),
        "source_url": row.get("source_url", ""),
    }


def extract_law_code(source_note: str) -> str:
    """
    Trích xuất mã luật từ source_note_text.
    Ví dụ: '(Điều 1 Luật số 32/2004/QH11 ...)' → '32/2004/QH11'
    """
    if not source_note:
        return ""
    # Patterns: QH, NĐ-CP, TT, QĐ...
    patterns = [
        r'\d+/\d{4}/QH\d+',     # Luật Quốc hội: 04/2017/QH14
        r'\d+/\d{4}/NĐ-CP',     # Nghị định: 80/2021/NĐ-CP
        r'\d+/\d{4}/TT-\w+',    # Thông tư: 05/2020/TT-BKHĐT
        r'\d+/\d{4}/QĐ-\w+',    # Quyết định
        r'số \d+/\d{4}/\w+',    # fallback
    ]
    for pat in patterns:
        m = re.search(pat, source_note)
        if m:
            return m.group(0).replace("số ", "")
    return ""


def normalize_text(text: str) -> str:
    """
    Chuẩn hoá Unicode (NFC) và bỏ khoảng trắng thừa.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def print_results(results: list[dict], query: str = "") -> None:
    """In kết quả đẹp ra console."""
    if query:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
    
    for r in results:
        print(f"\n[#{r['rank']}] Score: {r['score']:.4f}")
        if r.get("law_code"):
            print(f"  📋 Mã luật  : {r['law_code']}")
        print(f"  📚 Chủ đề   : {r.get('topic_title','')}")
        print(f"  📄 Điều luật: {r.get('article_title','')}")
        if r.get("chapter_title"):
            print(f"  📂 Chương   : {r.get('chapter_title','')}")
        print(f"  📝 Nội dung : {r.get('content_text','')[:200]}...")
        if r.get("source_url"):
            print(f"  🔗 URL      : {r.get('source_url','')}")