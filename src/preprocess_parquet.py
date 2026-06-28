import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Union, Optional
import json

# -----------------------
# Utils
# -----------------------

def parse_article(text: str, level="article") -> Dict[str, Any]:
    """
    Parse nội dung của một Điều.
    Thứ tự: article -> clause -> point
    """
    text = text.strip()

    def normalize(segment: str) -> str:
        return re.sub(r"\s+", " ", segment).strip()

    def parse_from_matches(text: str, matches: list) -> tuple[
        Optional[str],
        Union[str, list[str]]
    ]:
        """
        Generic parser dựa trên danh sách matches truyền vào.
        """
        # Không tìm thấy matches
        if not matches:
            return None, normalize(text)

        # Title = phần trước match đầu tiên
        title = normalize(text[: matches[0].start()])
        contents = []

        for i, match in enumerate(matches):
            start = match.start()
            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )
            contents.append(normalize(text[start:end]))

        return title, contents

    def filter_sequential_numbers(matches):
        """
        Chỉ giữ lại chuỗi bắt đầu từ 1, 2, 3, 4,...
        """
        if not matches:
            return []

        result = []
        expected = 1

        for match in matches:
            number = int(match.group(1))
            if number == expected:
                result.append(match)
                expected += 1

        return result

    # Bước 1: Tìm kiếm matches theo từng cấp độ
    if level == "article":
        raw_matches = list(re.finditer(r'(?<!\S)(\d+)\.\s', text)) # CLAUSE PATTERN: 1. 2. 3.
        matches = filter_sequential_numbers(raw_matches)

    elif level == "clause":
        matches = list(re.finditer(r'(?<!\S)[a-z]\)\s', text)) # POINT PATTERN a) b) c)

    else:
        # Tới cấp point hoặc sâu hơn thì chỉ trả về text
        return {"text": normalize(text)}

    # Bước 2: Parse nội dung dựa trên matches đã tìm được
    title, content = parse_from_matches(text, matches)

    # Nếu không bóc tách được cấp con nào, trả về toàn bộ text
    if title is None:
        return {"text": content}

    # Bước 3: Đệ quy xuống cấp độ con
    if level == "article":
        return {
            "title": title,
            "content": [
                parse_article(item, "clause") for item in content
            ]
        }

    # level == "clause"
    return {
        "title": title,
        "content": [
            parse_article(item, "point") for item in content
        ]
    }

def restore_article(node) -> str:
    """Revert `parse_article`"""

    if isinstance(node, str):
        return node.strip()

    if "text" in node:
        return node["text"].strip()

    parts = []

    title = node.get("title", "").strip()

    if title:
        parts.append(title)

    for child in node.get("content", []):
        child_text = restore_article(child)

        if child_text:
            parts.append(child_text)

    return "\n".join(parts)


def parse_source_note(text: str) -> Dict[str, str]:
    """
    Parse:
        Điều xx ...
        Tên văn bản ...
        Mã văn bản ...

    Ví dụ:
        Điều 8 Thông tư số 14/2020/TT-BKHĐT,
        có hiệu lực...

    =>
        article_index = Điều 8
        legal_type   = Thông tư
        docs_code    = 14/2020/TT-BKHĐT
        docs_title   = Thông tư số 14/2020/TT-BKHĐT
    """

    if not text:
        return {
            "article_index": "",
            "legal_type": "",
            "docs_code": "",
            "docs_title": ""
        }

    # --------------------------------------------------
    # 1. NORMALIZE
    # --------------------------------------------------
    def normalize(s: str) -> str:
        s = s.strip().strip("()")

        # nhiều khoảng trắng -> 1
        s = re.sub(r"\s+", " ", s)

        # 08 / 2017 / TT-BTNMT
        # -> 08/2017/TT-BTNMT
        s = re.sub(r"\s*/\s*", "/", s)

        # NĐ - CP
        # -> NĐ-CP
        s = re.sub(r"\s*-\s*", "-", s)

        # dấu phẩy
        s = re.sub(r"\s*,\s*", ", ", s)

        return s.strip()

    # --------------------------------------------------
    # 2. EXTRACT CODE
    # --------------------------------------------------
    def extract_code(s: str) -> Optional[re.Match]:

        patterns = [
            # 08/2017/TT-BTNMT
            r"\d+/\d+/[A-ZĐ0-9\-]+(?:-[A-ZĐ0-9]+)*",

            # 206/QĐ-TTg
            r"\d+/[A-ZĐ0-9\-]+(?:-[A-ZĐ0-9]+)*",

            # 23-L/CTN
            r"\d+-[A-ZĐ]+/[A-ZĐ0-9]+",

            # 370-HĐBT
            r"\d+-[A-ZĐ]+"
        ]

        candidates = []

        for pattern in patterns:
            for match in re.finditer(
                pattern,
                s,
                re.IGNORECASE
            ):

                value = match.group(0)

                # bỏ ngày tháng
                if re.fullmatch(
                    r"\d{1,2}/\d{1,2}/\d{4}",
                    value
                ):
                    continue

                candidates.append(match)

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x.start()
        )

        return candidates[0]

    # --------------------------------------------------
    # 3. EXTRACT TITLE
    # --------------------------------------------------
    def extract_title(s: str, code_match: re.Match) -> str:

        title = s[:code_match.end()]
        title = title.strip(" ,.")

        return title

    # --------------------------------------------------
    # 4. EXTRACT LEGAL TYPE
    # --------------------------------------------------
    def extract_legal_type(s: str, code_match: re.Match) -> str:
        # Lấy phần chuỗi nằm ngay trước docs_code
        prefix = s[:code_match.start()].strip(" ,.")
        
        # Cắt bỏ chữ "số" (nếu có) ở cuối phần prefix, không phân biệt hoa thường
        legal_type = re.sub(r"(?i)\s+số\s*$", "", prefix).strip()
        
        # Viết hoa chữ cái đầu tiên (hàm capitalize() tự động viết hoa chữ đầu và viết thường phần còn lại)
        if legal_type:
            return legal_type.capitalize()
            
        return ""

    # ==================================================
    # PIPELINE
    # ==================================================

    text = normalize(text)

    article_match = re.search(
        r"Điều\s+\d+[A-Za-z]*",
        text,
        re.IGNORECASE
    )

    if not article_match:
        return {
            "article_index": "",
            "legal_type": "",
            "docs_code": "",
            "docs_title": ""
        }

    article_index = article_match.group(0)

    remain = text[
        article_match.end():
    ].strip()

    code_match = extract_code(remain)

    if code_match is None:
        return {
            "article_index": article_index,
            "legal_type": "",
            "docs_code": "",
            "docs_title": ""
        }

    docs_code = code_match.group(0).upper()

    docs_title = extract_title(
        remain,
        code_match
    )
    
    legal_type = extract_legal_type(
        remain,
        code_match
    )

    return {
        "article_index": article_index,
        "legal_type": legal_type,
        "docs_code": docs_code,
        "docs_title": docs_title
    }


# -----------------------
# Save & Load with json dumps
# -----------------------

def save(df: pd.DataFrame, file_path: str) -> None: 
    
    # Tạo bản sao để không làm hỏng cấu trúc object của df gốc
    df_to_save = df.copy()

    # Hàm helper chuyển đổi sang JSON an toàn, tránh lỗi pd.notna trên list/array
    def safe_dump(x):
        # Nếu đã là list, dict, tuple -> dump ra string
        if isinstance(x, (list, dict, tuple)):
            return json.dumps(x, ensure_ascii=False)
        # Nếu là string -> giữ nguyên (tránh bị dump 2 lần thành '"chuỗi"')
        if isinstance(x, str):
            return x
        # Check null/NaN an toàn (dùng pd.api.types.is_scalar để chặn list/array)
        if pd.api.types.is_scalar(x) and pd.isna(x):
            return None
        # Nếu vô tình là numpy array
        if isinstance(x, np.ndarray):
            return json.dumps(x.tolist(), ensure_ascii=False)
        # Các trường hợp còn lại
        return json.dumps(x, ensure_ascii=False)

    # Xử lý content_text
    if 'content_text' in df_to_save.columns:
        df_to_save['content_text'] = df_to_save['content_text'].apply(safe_dump)

    # Xử lý references nếu có tồn tại
    if 'references' in df_to_save.columns:
        df_to_save['references'] = df_to_save['references'].apply(safe_dump)

    # Lưu file
    if file_path.endswith('parquet'):
        df_to_save.to_parquet(file_path)
    elif file_path.endswith('csv'):
        # Lưu CSV nên bỏ index để lúc load không bị dư cột Unnamed: 0
        df_to_save.to_csv(file_path, index=False) 
    else:
        df_to_save.to_json(file_path, orient='records', force_ascii=False)

    print(f"Đã lưu dữ liệu vào {file_path}")


def load(file_path: str) -> pd.DataFrame:
    # Hỗ trợ load theo đúng định dạng đã lưu
    if file_path.endswith('parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_json(file_path)

    # Hàm helper load từ JSON string một cách an toàn
    def safe_load(x):
        # File Parquet có thể tự động giữ nguyên định dạng list/dict, nếu thế thì khỏi cần load
        if isinstance(x, (list, dict, tuple)):
            return x
        # Chỉ parse JSON nếu x là chuỗi
        if isinstance(x, str):
            try:
                return json.loads(x)
            except (json.JSONDecodeError, TypeError):
                # Nếu chuỗi không phải là chuẩn JSON, trả về nguyên bản chuỗi đó
                return x
        return x

    # Load lại cấu trúc object cho content_text
    if "content_text" in df.columns:
        df["content_text"] = df["content_text"].apply(safe_load)

    # Load lại cấu trúc object cho references nếu có
    if "references" in df.columns:
        df["references"] = df["references"].apply(safe_load)

    return df
