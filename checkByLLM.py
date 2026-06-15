import json
import requests

# Cấu hình endpoint của Ollama chạy local
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"  # Thay đổi tên model của bạn nếu cần (ví dụ: gemma2:4b hoặc gemma:4b)

def evaluate_relevance(query, article_title, content_text):

    prompt = f"""Bạn là một trợ lý pháp lý chuyên nghiệp. Hãy đánh giá xem đoạn văn bản luật dưới đây có chứa thông tin giúp trả lời câu hỏi hay không.

Câu hỏi: "{query}"

Văn bản luật:
- Tiêu đề: {article_title}
- Nội dung: {content_text}

Yêu cầu:
Chỉ trả lời duy nhất từ "YES" nếu văn bản bắt buộc để trả lời câu hỏi, hoặc "NO" nếu văn bản KHÔNG liên quan đến câu hỏi. Không giải thích gì thêm.

Trả lời (YES/NO):"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if response.status_code == 200:
            result_text = response.json().get("response", "").strip().upper()
            if "YES" in result_text:
                return True
            elif "NO" in result_text:
                return False
            else:
                print(f" Phản hồi không chuẩn xác từ LLM: '{result_text}'. Sẽ tự động quét từ khóa.")
                return "YES" in result_text
        else:
            print(f" Lỗi kết nối tới Ollama (Status {response.status_code}): {response.text}")
            return True
    except Exception as e:
        print(f" Lỗi khi gọi LLM: {e}")
        return True


def save_progress(filtered_data, output_file_path):
    """Ghi toàn bộ kết quả đã xử lý ra file."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            json.dump(filtered_data, out_f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f" Lỗi khi ghi file tiến trình: {e}")


def filter_json_data(input_file_path, output_file_path):
    """
    Đọc file JSON, duyệt qua từng câu hỏi và kết quả để lọc bằng LLM local.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Không thể đọc file đầu vào: {e}")
        return

    filtered_data = []

    for index, item in enumerate(data, 1):
        query = item.get("query", "")
        results = item.get("results", [])
        
        print(f"\n[{index}/{len(data)}] Đang xử lý câu hỏi: '{query[:50]}...'")
        print(f" Tổng số kết quả ban đầu cần đánh giá: {len(results)}")

        useful_results = []
        useful_law_references = []

        for res in results:
            article_title = res.get("article_title", "")
            content_text = res.get("content_text", "")
            law_code = res.get("law_code", "")
            
            is_useful = evaluate_relevance(query, article_title, content_text)
            
            if is_useful:
                useful_results.append(res)
                for ref in item.get("law_references", []):
                    if law_code in ref and article_title[:30] in ref:
                        if ref not in useful_law_references:
                            useful_law_references.append(ref)
                
                print(f"  --> [GIỮ LẠI]")
            else:
                print(f"  --> [LOẠI BỎ]")

        if not useful_law_references and useful_results:
            for res in useful_results:
                matched_ref = next((ref for ref in item.get("law_references", []) if res.get("law_code") in ref), None)
                if matched_ref and matched_ref not in useful_law_references:
                    useful_law_references.append(matched_ref)

        new_item = {
            "id": item.get("id"),
            "query": query,
            "num_results": len(useful_results),
            "law_references": useful_law_references,
            "results": useful_results
        }
        filtered_data.append(new_item)

        # >>> Lưu file sau mỗi query hoàn thành <
        save_progress(filtered_data, output_file_path)
        print(f" [Đã lưu tiến trình: {index}/{len(data)} queries]")

    print(f"\n[Thành công] Đã hoàn thành lọc và lưu kết quả tại: {output_file_path}")


# ===================================================
# THIẾT LẬP ĐƯỜNG DẪN ĐỂ CHẠY CHƯƠNG TRÌNH
# ===================================================
if __name__ == "__main__":

    file_dau_vao = "results/all.json"
    file_sau_loc = "results/query_results_filter.json"
    
    filter_json_data(file_dau_vao, file_sau_loc)