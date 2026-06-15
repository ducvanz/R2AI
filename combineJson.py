import json

def merge_json_files(file_paths, output_path):
    """
    Gộp danh sách các file JSON theo trường 'id' và lưu lại.
    """
    merged_data = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    item_id = item.get('id')
                    if item_id is None:
                        continue
                    
                    # Nếu ID chưa tồn tại, thêm mới vào dictionary
                    if item_id not in merged_data:
                        merged_data[item_id] = {
                            "id": item_id,
                            "query": item.get("query", ""),
                            "num_results": item.get("num_results", 0),
                            "law_references": list(item.get("law_references", [])),
                            "results": list(item.get("results", []))
                        }
                    else:
                        target = merged_data[item_id]
                        
                        # 1. Gộp law_references (tránh trùng)
                        for ref in item.get("law_references", []):
                            if ref not in target["law_references"]:
                                target["law_references"].append(ref)
                        
                        # 2. Gộp results (tránh trùng dựa trên law_code và article_title)
                        existing_keys = {(r.get("law_code"), r.get("article_title")) for r in target["results"]}
                        for res in item.get("results", []):
                            key = (res.get("law_code"), res.get("article_title"))
                            if key not in existing_keys:
                                target["results"].append(res)
                                existing_keys.add(key)
                        
                        # 3. Cập nhật số lượng kết quả
                        target["num_results"] = len(target["results"])
                        
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")

    # Ghi toàn bộ dữ liệu ra file mới (giữ nguyên thứ tự đọc được, không sắp xếp)
    try:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(list(merged_data.values()), out_f, ensure_ascii=False, indent=4)
        print(f"Đã gộp thành công {len(file_paths)} file. Kết quả lưu tại: {output_path}")
    except Exception as e:
        print(f"Lỗi khi ghi file kết quả: {e}")


# ===================================================
# ĐIỀN ĐƯỜNG DẪN CÁC FILE CỦA BẠN TẠI ĐÂY
# ===================================================
if __name__ == "__main__":
    
    # 1. Điền danh sách các file JSON cần nối vào đây
    danh_sach_file_can_gop = [
        "results/query_results.json",
        "results/query_results11.json",
        "results/query_results12.json",
        "results/query_results13.json",
        "results/query_results2.json",
        "results/query_results21.json",
        "results/query_results22.json",
        "results/query_results225.json",
        "results/query_results23.json"
    ]
    
    # 2. Điền đường dẫn file kết quả muốn xuất ra
    file_dau_ra = "results/all.json"
    
    # 3. Gọi hàm xử lý gộp
    merge_json_files(danh_sach_file_can_gop, file_dau_ra)