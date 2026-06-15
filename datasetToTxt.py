import os
import re
import ast
import pandas as pd

def load_data(filepath):
    """
    Đọc tệp tin Parquet chứa dữ liệu pháp điển Việt Nam.
    """
    print(f"=== 1. ĐANG ĐỌC FILE DỮ LIỆU: {filepath} ===")
    if not os.path.exists(filepath):
        print(f"❌ Lỗi: Không tìm thấy tệp tin '{filepath}'.")
        print("Vui lòng đặt tệp tin Parquet cùng thư mục với file script này hoặc truyền đường dẫn chính xác.")
        return None
    
    try:
        # Đọc file parquet bằng pandas (yêu cầu thư viện 'pyarrow' hoặc 'fastparquet')
        df = pd.read_parquet(filepath)
        print(f"✅ Đọc dữ liệu thành công!")
        print(f" - Tổng số bản ghi (dòng): {df.shape[0]}")
        print(f" - Số lượng cột thuộc tính: {df.shape[1]}")
        print("\nCác cột dữ liệu và kiểu dữ liệu tương ứng:")
        for col, dtype in zip(df.columns, df.dtypes):
            print(f" • {col:<20} | {dtype}")
        print("-" * 60)
        return df
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi đọc file Parquet: {e}")
        print("Mẹo: Hãy cài đặt thư viện hỗ trợ bằng lệnh: pip install pandas pyarrow")
        return None

def visualize_taxonomy_tree(df, num_samples=3):
    """
    Trực quan hóa cấu trúc phân cấp pháp lý dưới dạng cây phân nhánh (Terminal Tree).
    """
    print(f"\n=== 2. TRỰC QUAN HÓA CẤU TRÚC PHÂN CẤP (Hiển thị {num_samples} bản ghi mẫu) ===")
    
    # Lấy ra các dòng dữ liệu không bị khuyết thiếu cấu trúc chính
    clean_df = df.dropna(subset=['subject_title', 'topic_title', 'chapter_title', 'article_title']).head(num_samples)
    
    if clean_df.empty:
        print("Không có đủ dữ liệu hợp lệ để dựng cây phân cấp.")
        return

    for idx, (_, row) in enumerate(clean_df.iterrows(), 1):
        print(f"\n[BẢN GHI MẪU SỐ {idx}]")
        print("└── 📂 BỘ PHÁP ĐIỂN VIỆT NAM")
        print(f"    └── 📁 Chủ đề {row['subject_number']}: {row['subject_title']}")
        print(f"        └── 🗂️ Đề mục {row['topic_number']}: {row['topic_title']}")
        print(f"            └── 📖 {row['chapter_title']}")
        print(f"                └── 📜 {row['article_title']}")
        
        # Rút gọn nội dung điều luật để hiển thị đẹp mắt trên màn hình terminal
        content_preview = row['content_text'] if pd.notna(row['content_text']) else ""
        if len(content_preview) > 80:
            content_preview = content_preview[:80] + "..."
            
        print(f"                    ├── 📝 Nội dung sơ lược: \"{content_preview}\"")
        print(f"                    ├── 🔢 Thống kê: {row['content_word_count']} từ | {row['content_char_len']} ký tự")
        print(f"                    └── 🔗 URL gốc: {row['source_url']}")
    print("-" * 60)

def extract_cross_references(row):
    """
    Bóc tách danh sách các điều luật liên quan từ trường related_note_text bằng Regex.
    """
    related_text = row.get('related_note_text', '')
    if not isinstance(related_text, str) or pd.isna(related_text):
        return []
    
    # Tìm kiếm các mẫu có định dạng: "Điều X.X.LQ.X" hoặc tương đương
    # Ví dụ: "Điều 1.12.LQ.11", "Điều 1.11.LQ.1", v.v.
    pattern = r"(Điều\s+\d+\.\d+\.[A-Za-z0-9\.]+[^;)]*)"
    matches = re.findall(pattern, related_text)
    
    # Làm sạch các khoảng trắng dư thừa
    cleaned_references = [match.strip() for match in matches]
    return cleaned_references

def analyze_relationships(df):
    """
    Phân tích các mối quan hệ liên kết chéo giữa các điều khoản luật.
    """
    print("\n=== 3. PHÂN TÍCH QUAN HỆ CHÉO (CROSS-REFERENCES) ===")
    
    # Lọc các dòng có chứa thông tin liên quan
    related_df = df[df['related_note_text'].notna() & (df['related_note_text'] != '')]
    print(f"Tìm thấy {len(related_df)}/{len(df)} bản ghi có chứa điều khoản liên quan.")
    
    sample_limit = min(5, len(related_df))
    if sample_limit > 0:
        print(f"Chi tiết liên kết của {sample_limit} bản ghi đầu tiên:")
        for idx, (_, row) in enumerate(related_df.head(sample_limit).iterrows(), 1):
            refs = extract_cross_references(row)
            print(f"\n👉 {idx}. Điều luật gốc: '{row['article_title']}'")
            print(f"   Liên kết tới {len(refs)} điều khoản khác:")
            if refs:
                for ref in refs:
                    print(f"   └── Link 🔗: {ref}")
            else:
                print("   └── (Không bóc tách được chi tiết, nội dung thô: " + row['related_note_text'][:100] + "...)")
    print("-" * 60)

def parse_source_links(links_str):
    """
    Hàm bổ trợ giúp chuyển đổi trường dữ liệu source_links (dạng chuỗi thô của Python list/dict)
    sang danh sách dữ liệu có thể thao tác được trong Python một cách an sau.
    """
    if not isinstance(links_str, str) or pd.isna(links_str):
        return []
    try:
        # Sử dụng ast.literal_eval để chuyển dạng chuỗi có nháy đơn '[{...}]' sang list trong Python an toàn
        return ast.literal_eval(links_str)
    except Exception:
        return []

def analyze_sources(df):
    """
    Trích xuất và phân tích nguồn gốc văn bản ban hành điều luật.
    """
    print("\n=== 4. TRÍCH XUẤT VÀ KIỂM TRA VĂN BẢN NGUỒN ===")
    
    sample_row = df.dropna(subset=['source_links']).first_valid_index()
    if sample_row is not None:
        row = df.loc[sample_row]
        links = parse_source_links(row['source_links'])
        
        print(f"Ví dụ kiểm tra nguồn gốc của điều luật: '{row['article_title']}'")
        print(f"📜 Ghi chú văn bản gốc: {row['source_note_text']}")
        print(f"🌐 Danh sách liên kết cơ sở dữ liệu Luật quốc gia:")
        for idx, link in enumerate(links, 1):
            print(f"   [{idx}] Văn bản: {link.get('text')}")
            print(f"       Liên kết trực tiếp: {link.get('href')}")
    else:
        print("Không tìm thấy thông tin trường 'source_links' để phân tích mẫu.")
    print("-" * 60)

def export_full_taxonomy_to_txt(df, output_txt_path="cau_truc_phap_dien.txt"):
    """
    Xây dựng toàn bộ cây phân cấp từ bộ dữ liệu Parquet và xuất ra tệp tin .txt trực quan.
    """
    print(f"\n=== 5. ĐANG KHỞI TẠO VÀ XUẤT TOÀN BỘ CÂY PHÂN CẤP RA FILE: {output_txt_path} ===")
    
    # Loại bỏ các dòng bị khuyết cấu trúc chính để sơ đồ được nhất quán
    clean_df = df.dropna(subset=['subject_title', 'topic_title', 'chapter_title', 'article_title'])
    
    if clean_df.empty:
        print("❌ Không có dữ liệu hợp lệ để xuất cây phân cấp.")
        return

    # 1. Nhóm dữ liệu thành cấu trúc cây dạng lồng nhau (Nested Dictionary)
    tree_dict = {}
    for _, row in clean_df.iterrows():
        # Định dạng nhãn cho từng cấp
        subj = f"Chủ đề {row['subject_number']}: {row['subject_title']}"
        topic = f"Đề mục {row['topic_number']}: {row['topic_title']}"
        chap = row['chapter_title']
        art = row['article_title']
        
        # Thêm dần vào cây lồng nhau
        if subj not in tree_dict:
            tree_dict[subj] = {}
        if topic not in tree_dict[subj]:
            tree_dict[subj][topic] = {}
        if chap not in tree_dict[subj][topic]:
            tree_dict[subj][topic][chap] = set() # Sử dụng set() để tự động khử trùng lặp các điều luật giống nhau
        
        tree_dict[subj][topic][chap].add(art)

    # 2. Duyệt qua cây vừa tạo để sinh chuỗi văn bản phân cấp trực quan
    lines = []
    lines.append("📂 BỘ PHÁP ĐIỂN VIỆT NAM")
    
    subjects = sorted(tree_dict.keys())
    for i, subj in enumerate(subjects):
        subj_is_last = (i == len(subjects) - 1)
        subj_prefix = "└── " if subj_is_last else "├── "
        lines.append(f"{subj_prefix}📁 {subj}")
        
        # Thiết lập khoảng lùi đầu dòng cho các cấp con tiếp theo
        subj_child_indent = "    " if subj_is_last else "│   "
        
        topics = sorted(tree_dict[subj].keys())
        for j, topic in enumerate(topics):
            topic_is_last = (j == len(topics) - 1)
            topic_prefix = "└── " if topic_is_last else "├── "
            lines.append(f"{subj_child_indent}{topic_prefix}🗂️ {topic}")
            
            topic_child_indent = subj_child_indent + ("    " if topic_is_last else "│   ")
            
            chapters = sorted(tree_dict[subj][topic].keys())
            for k, chap in enumerate(chapters):
                chap_is_last = (k == len(chapters) - 1)
                chap_prefix = "└── " if chap_is_last else "├── "
                lines.append(f"{topic_child_indent}{chap_prefix}📖 {chap}")
                
                chap_child_indent = topic_child_indent + ("    " if chap_is_last else "│   ")
                
                articles = sorted(list(tree_dict[subj][topic][chap]))
                for l, art in enumerate(articles):
                    art_is_last = (l == len(articles) - 1)
                    art_prefix = "└── " if art_is_last else "├── "
                    lines.append(f"{chap_child_indent}{art_prefix}📜 {art}")
                    
    # 3. Ghi toàn bộ nội dung cây phân cấp vào tệp .txt với encoding UTF-8
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✅ Đã xuất sơ đồ thành công! File lưu tại: {os.path.abspath(output_txt_path)}")
        print(f" - Tổng số nhánh/dòng vẽ cây: {len(lines)}")
    except Exception as e:
        print(f"❌ Không thể lưu file .txt do lỗi: {e}")
    print("-" * 60)

def main():
    # Tên mặc định của tệp Parquet
    # Bạn hãy thay thế tên file bên dưới bằng đường dẫn thực tế đến tệp của bạn (Ví dụ: "C:/data/phapdien.parquet")
    file_name = "data/data_business.parquet" 
    
    # Tiến hành thực thi các bước phân tích dữ liệu
    df = load_data(file_name)
    if df is not None:
        visualize_taxonomy_tree(df, num_samples=2)
        analyze_relationships(df)
        analyze_sources(df)
        
        # GỌI HÀM XUẤT TOÀN BỘ CÂY PHÂN CẤP RA FILE TXT
        export_full_taxonomy_to_txt(df, "cau_truc_phap_dien_full.txt")
        
        print("\n🎉 PHÂN TÍCH VÀ XUẤT DỮ LIỆU HOÀN TẤT!")
        print("Tệp tin 'cau_truc_phap_dien_full.txt' đã được tạo ra trong cùng thư mục với mã nguồn này.")

if __name__ == "__main__":
    main()