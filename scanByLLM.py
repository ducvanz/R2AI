"""
rewrite_articles.py
--------------------
Đọc file .parquet chứa dữ liệu pháp điển Việt Nam,
ghép context đầy đủ (chủ đề + đề mục + chương + điều),
rồi dùng LLM local (Ollama gemma3:4b) viết lại thành
văn bản có ngữ cảnh hoàn chỉnh.

Kết quả lưu vào rewritten_articles.json với cấu trúc:
{
  "Điều 25.10.NĐ.1.": "Phạm vi điều chỉnh của luật về ...",
  ...
}

Yêu cầu:
  pip install pandas pyarrow requests tqdm
  Ollama đang chạy: ollama serve
  Model: ollama pull gemma3:4b
"""

import json
import sys
import time
import textwrap
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
#  CẤU HÌNH — chỉnh sửa 2 biến này trước khi chạy
# ═══════════════════════════════════════════════════════════════

PARQUET_FILE  = "phapdien_cache.parquet"   # đường dẫn file parquet
OUTPUT_FILE   = "rewritten_articles2.json"

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "gemma3:4b"

# Giới hạn số điều luật xử lý (None = tất cả)
LIMIT         = None

# Nếu True: bỏ qua điều đã có trong OUTPUT_FILE (tiếp tục từ lần chạy trước)
RESUME        = True

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        # Tự tìm trong thư mục hiện tại
        found = sorted(Path(".").glob("**/*.parquet"))
        if not found:
            print(f"[LỖI] Không tìm thấy file: {path}")
            sys.exit(1)
        p = found[0]
        print(f"[INFO] Dùng file tìm thấy: {p}")
    df = pd.read_parquet(p)
    print(f"[INFO] Đọc xong {len(df):,} bản ghi từ {p.name}")
    return df


def load_existing(output_path: str) -> dict:
    p = Path(output_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Resume: đã có {len(data):,} điều trong {p.name}")
        return data
    return {}


def save_json(data: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def call_llm(prompt: str, retries: int = 3) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            print("\n[LỖI] Không kết nối được Ollama. Đảm bảo 'ollama serve' đang chạy.")
            sys.exit(1)
        except Exception as e:
            if attempt == retries:
                print(f"\n[WARN] LLM thất bại sau {retries} lần: {e}")
                return ""
            time.sleep(2 * attempt)
    return ""


# ═══════════════════════════════════════════════════════════════
#  BUILD CONTEXT — ghép đầy đủ ngữ cảnh cho 1 điều luật
# ═══════════════════════════════════════════════════════════════

def build_context(row: pd.Series) -> str:
    """
    Ghép các trường thành 1 đoạn văn mô tả đầy đủ ngữ cảnh của điều luật.
    """
    parts = []

    subject_num   = row.get("subject_number", "")
    subject_title = row.get("subject_title", "")
    topic_num     = row.get("topic_number", "")
    topic_title   = row.get("topic_title", "")
    chapter_title = row.get("chapter_title", "")
    article_title = row.get("article_title", "")
    content       = row.get("content_text", "")
    source        = row.get("source_note_text", "")

    if subject_title:
        parts.append(f"Lĩnh vực pháp luật (Chủ đề {subject_num}): {subject_title}")
    if topic_title:
        parts.append(f"Đề mục {topic_num}: {topic_title}")
    if chapter_title:
        parts.append(f"Chương: {chapter_title}")
    if article_title:
        parts.append(f"Điều luật: {article_title}")
    if content:
        parts.append(f"Nội dung điều luật: {content}")
    if source:
        parts.append(f"Căn cứ pháp lý: {source}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  PROMPT — yêu cầu LLM viết lại
# ═══════════════════════════════════════════════════════════════

REWRITE_PROMPT_TEMPLATE = """
Bạn là chuyên gia pháp luật Việt Nam. Dựa vào thông tin đầy đủ dưới đây,
hãy viết lại TÊN của điều luật thành một câu mô tả ngắn gọn, súc tích,
có đầy đủ ngữ cảnh để người đọc hiểu ngay điều luật này quy định về gì,
trong lĩnh vực gì, áp dụng cho đối tượng nào.

Yêu cầu:
- Bắt đầu bằng mã điều luật (ví dụ: "Điều 25.10.NĐ.1.")
- Tiếp theo là dấu hai chấm và phần mô tả có ngữ cảnh
- Độ dài: 1-3 câu, không quá 200 từ
- Ngôn ngữ: tiếng Việt, rõ ràng, chính xác
- Chỉ trả lời phần mô tả, không giải thích thêm

--- THÔNG TIN ĐẦY ĐỦ ---
{context}
--- HẾT THÔNG TIN ---

Kết quả:
""".strip()


def rewrite_article(row: pd.Series) -> str:
    context = build_context(row)
    prompt  = REWRITE_PROMPT_TEMPLATE.format(context=context)
    result  = call_llm(prompt)
    return result


# ═══════════════════════════════════════════════════════════════
#  EXTRACT ARTICLE CODE — lấy mã điều luật từ article_title
# ═══════════════════════════════════════════════════════════════

def extract_code(article_title: str) -> str:
    """
    Ví dụ: 'Điều 25.10.NĐ.1. Phạm vi điều chỉnh'
    → 'Điều 25.10.NĐ.1.'
    """
    import re
    # Pattern: Điều X.Y.Z.N. (mã điều luật)
    m = re.match(r"(Điều\s+[\d\.A-ZĐÔƯƠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴA-zđôươắằẳẵặấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+\.+)", article_title)
    if m:
        return m.group(1).strip()
    # Fallback: dùng toàn bộ title làm key
    return article_title.strip()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    df = load_parquet(PARQUET_FILE)

    # Loại bỏ bản ghi không có nội dung
    df = df[df["content_text"].notna() & (df["content_text"].str.strip() != "")]
    df = df[df["article_title"].notna()].reset_index(drop=True)
    print(f"[INFO] Số điều luật có nội dung: {len(df):,}")

    if LIMIT:
        df = df.head(LIMIT)
        print(f"[INFO] Giới hạn xử lý: {LIMIT} điều")

    # Load kết quả cũ nếu RESUME
    results = load_existing(OUTPUT_FILE) if RESUME else {}
    already_done = set(results.keys())

    # Lọc ra những điều chưa xử lý
    pending = df[~df["article_title"].apply(lambda t: extract_code(t) in already_done)]
    print(f"[INFO] Cần xử lý: {len(pending):,} điều (bỏ qua {len(already_done):,} đã có)\n")

    if pending.empty:
        print("[INFO] Tất cả điều luật đã được xử lý. Xem kết quả trong:", OUTPUT_FILE)
        return

    save_every = 10  # Lưu file sau mỗi N điều (tránh mất dữ liệu)
    count = 0

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Đang xử lý"):
        code    = extract_code(row["article_title"])

        rewrite = False; #rewrite_article(row)

        if rewrite:
            results[code] = rewrite
        else:
            # Nếu LLM thất bại, giữ lại title gốc + nội dung
            results[code] = f"{row.get('article_title','')} — {row.get('content_text','')[:200]}"

        count += 1
        if count % save_every == 0:
            save_json(results, OUTPUT_FILE)
            tqdm.write(f"  [Checkpoint] Đã lưu {len(results):,} điều → {OUTPUT_FILE}")

    # Lưu lần cuối
    save_json(results, OUTPUT_FILE)

    print(f"\n{'='*60}")
    print(f"✅ Hoàn tất! Tổng cộng {len(results):,} điều luật đã được viết lại.")
    print(f"   Kết quả lưu tại: {OUTPUT_FILE}")
    print(f"{'='*60}")

    # In preview 3 điều đầu tiên
    print("\n📋 Preview 3 điều đầu tiên:\n")
    for i, (code, text) in enumerate(list(results.items())[:3], 1):
        print(f"  {i}. {code}")
        print(f"     {textwrap.fill(text, width=70, subsequent_indent='     ')}")
        print()


if __name__ == "__main__":
    main()