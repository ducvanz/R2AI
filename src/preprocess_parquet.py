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
    Parse source_note_text.

    Example
    -------
    Input:
        (Điều 8 Thông tư số 14/2020/TT-BKHĐT,
         có hiệu lực thi hành kể từ ngày 25/02/2021)

    Output:
        {
            "docs_title":
                "14/2020/TT-BKHĐT|Thông tư số 14/2020/TT-BKHĐT",
            "article_index":
                "Điều 8"
        }
    """

    if not text:
        return {
            "docs_title": "",
            "article_index": ""
        }

    text = text.strip().strip("()")

    # chỉ lấy phần trước dấu phẩy đầu tiên
    head = text.split(",", 1)[0].strip()

    article_match = re.search(
        r"Điều\s+\d+[A-Za-z]*",
        head
    )

    if not article_match:
        return {
            "docs_title": "",
            "article_index": ""
        }

    article_index = article_match.group(0)

    doc_title = head[article_match.end():].strip()

    code_match = re.search(
        r"\d+/\d+/[^\s,]+",
        doc_title
    )

    docs_title = ""

    if code_match:
        doc_code = code_match.group(0)
        docs_title = f"{doc_code}|{doc_title}"
    else:
        docs_title = doc_title

    return {
        "docs_title": docs_title,
        "article_index": article_index
    }


# -----------------------
# Pre-process full pipeline
# -----------------------

def preProcess(raw: pd.DataFrame, save_path: str=None) -> pd.DataFrame :
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

    final['metadata'] = final['docs_title'] + '|' + final['article_index']

    final = final.drop(columns="source_note_text")

    if save_path is not None:
        save(final, save_path)

    return final


# -----------------------
# I/O
# -----------------------

def load_parquet(parquet_path: str,
                          columns = ["topic_title",
                                     "subject_title",
                                     "article_title",
                                     "content_text",
                                     "content_word_count",
                                     "source_note_text"] ) -> pd.DataFrame:
    """
    Đọc file parquet và chỉ giữ lại các cột cần thiết.
    """

    df = pd.read_parquet(parquet_path)

    # Chỉ giữ các cột tồn tại trong file
    available_cols = [c for c in columns if c in df.columns]

    return df[available_cols].copy()

def save(df: pd.DataFrame, file_path:str) -> None : 
    df['content_text'] = df['content_text'].apply(
        lambda x: json.dumps(
            x,
            ensure_ascii=False
        )
    )

    df.to_parquet(file_path)

    print(f"Đã lưu dữ liệu vào {file_path}")

def load(file_path:str) -> pd.DataFrame :
    df = pd.read_parquet(file_path)

    df["content_text"] = (
        df["content_text"]
        .apply(json.loads)
    )

    return df
