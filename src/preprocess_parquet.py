import pandas as pd
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
        Điều 8 Thông tư số 14 /2020/TT-BKHĐT,
        có hiệu lực...

    =>
        article_index = Điều 8
        docs_code    = 14/2020/TT-BKHĐT
        docs_title   = Thông tư số 14/2020/TT-BKHĐT
    """

    if not text:
        return {
            "article_index": "",
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
            "docs_code": "",
            "docs_title": ""
        }

    docs_code = code_match.group(0).upper()

    docs_title = extract_title(
        remain,
        code_match
    )

    return {
        "article_index": article_index,
        "docs_code": docs_code,
        "docs_title": docs_title
    }


# -----------------------
# Pre-process full pipeline
# -----------------------

def preProcess(raw: pd.DataFrame, save_path: str=None, fix_path: str=None) -> pd.DataFrame :
    """
    Đọc thông tin từ dataset.
    Thực hiện extract metadata từ cột `source_note_text`.
    Thực hiện parse (phân nhỏ) nội dung các Điều `content_text`.
    """
    metadata = (raw["source_note_text"]
                .apply(parse_source_note)
                .apply(pd.Series))
    
    final = pd.concat(
        [raw, metadata],
        axis=1
    )

    final['content_text'] = final['content_text'].apply(parse_article)
    final['content_clause_count'] = final['content_text'].apply(lambda x: 
                                                                len(x.get('content', []))
                                                                )

    final = final[['docs_code', 'docs_title', 'article_index', 'article_title', 'source_note_text', 'source_links', 'topic_title', 'subject_title', 'content_text', 'content_word_count', 'content_clause_count']]

    if fix_path is not None:
        fix_df = pd.read_csv(fix_path, index_col=0)
        fix_df["content_text"] = (
            fix_df["content_text"]
            .apply(json.loads)
        )
        final.update(fix_df)
        print(f"Update {len(fix_df)} samples thủ công.")

    if save_path is not None:
        save(final, save_path)

    return final

# -----------------------
# I/O
# -----------------------

def load_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Đọc file parquet và chỉ giữ lại các cột cần thiết.
    """

    return pd.read_parquet(parquet_path)

def save(df: pd.DataFrame, file_path:str) -> None : 
    df['content_text'] = df['content_text'].apply(
        json.dumps
    )

    if file_path.endswith('parquet') :
        df.to_parquet(file_path)
    elif file_path.endswith('csv') :
        df.to_csv(file_path)
    else :
        df.to_json(file_path)

    print(f"Đã lưu dữ liệu vào {file_path}")

def load(file_path:str) -> pd.DataFrame :
    df = pd.read_parquet(file_path)

    df["content_text"] = (
        df["content_text"]
        .apply(json.loads)
    )

    return df
