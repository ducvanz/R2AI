"""
filter_business.py
-------------------
Lọc nhanh bộ dữ liệu pháp điển theo 2 tầng:

  TẦNG 1 — Đề mục (topic_title):
    Nếu đề mục bị loại → toàn bộ điều con bên trong loại luôn
    Nếu đề mục được giữ → giữ toàn bộ điều con, không cần xét tiếp

  TẦNG 2 — Điều luật (article_title + content_text):
    Chỉ áp dụng khi đề mục chưa rõ ràng (không khớp tầng 1)

Không dùng LLM → chạy tức thì dù dataset triệu bản ghi.
"""

from pathlib import Path
import pandas as pd

# ═══════════════════════════════════════════════════════════════
#  CẤU HÌNH
# ═══════════════════════════════════════════════════════════════

PARQUET_FILE   = "phapdien_cache.parquet"
OUTPUT_PARQUET = "data_business.parquet"
OUTPUT_REPORT  = "filter_report.txt"

# ═══════════════════════════════════════════════════════════════
#  TẦNG 1A — LOẠI TOÀN BỘ ĐỀ MỤC (+ mọi điều con bên trong)
#  Khớp vào subject_title hoặc topic_title
# ═══════════════════════════════════════════════════════════════

TOPIC_EXCLUDE = [
    # Quân sự / Quốc phòng
    "quốc phòng", "quân sự", "quân đội", "dân quân tự vệ",
    "dự bị động viên", "động viên quốc phòng",
    "biên phòng", "bộ đội biên phòng",
    "cảnh vệ", "bảo vệ chủ quyền",
    "chiến tranh", "phòng thủ dân sự",
    "quân nhân", "sĩ quan", "hạ sĩ quan",
    "xuất ngũ", "phục viên",
    "chiến sĩ", "lực lượng vũ trang",
    "nghĩa vụ quân sự",
    "tham gia chiến tranh bảo vệ tổ quốc",
    "làm nhiệm vụ quốc tế",
    "giúp bạn lào", "căm-pu-chia",

    # An ninh / Công an thuần túy
    "an ninh quốc gia",
    "công an nhân dân",
    "cảnh sát nhân dân",
    "tình báo", "phản gián",
    "bảo vệ bí mật nhà nước",
    "phòng chống khủng bố",

    # Đảng / Chính trị
    "đảng cộng sản",
    "tổ chức cơ sở đảng",
    "kỷ luật đảng", "đảng viên",
    "bầu cử đại biểu quốc hội",
    "bầu cử đại biểu hội đồng",
    "tổ chức quốc hội",
    "tổ chức hội đồng nhân dân",
    "mặt trận tổ quốc",
    "tổ chức chính trị - xã hội",

    # Người có công / Cựu chiến binh
    "người có công",
    "cựu chiến binh",
    "thương binh", "liệt sĩ",
    "bà mẹ việt nam anh hùng",
    "anh hùng lực lượng vũ trang",
    "chế độ ưu đãi người có công",
    "chế độ đối với đối tượng tham gia",

    # Tư pháp hình sự / Giam giữ
    "thi hành án hình sự",
    "trại giam", "trại tạm giam",
    "tạm giữ hình sự",
    "phạm nhân", "người chấp hành án phạt tù",
    "cải tạo không giam giữ",
    "đặc xá", "xóa án tích", "quản chế",
    "giáo dục tại xã phường thị trấn",
    "đưa vào trường giáo dưỡng",
    "đưa vào cơ sở giáo dục bắt buộc",
    "đưa vào cơ sở cai nghiện",

    # Tôn giáo / Tín ngưỡng
    "tôn giáo", "tín ngưỡng",
    "cơ sở tôn giáo", "chức sắc tôn giáo",

    # Hộ tịch thuần túy
    "hộ tịch", "khai sinh", "khai tử",
    "nuôi con nuôi", "giám hộ",

    # Thiên tai / Cứu hộ thuần túy
    "phòng chống thiên tai",
    "tìm kiếm cứu nạn", "cứu hộ cứu nạn",

    # Di tích / Di sản phi thương mại
    "di tích lịch sử - văn hóa",
    "di sản văn hóa phi vật thể",
    "bảo tồn di sản",
]

# ═══════════════════════════════════════════════════════════════
#  TẦNG 1B — GIỮ TOÀN BỘ ĐỀ MỤC (+ mọi điều con bên trong)
#  Khớp vào subject_title hoặc topic_title
# ═══════════════════════════════════════════════════════════════

TOPIC_KEEP = [
    # Chủ thể kinh doanh
    "doanh nghiệp", "công ty", "hợp tác xã",
    "hộ kinh doanh", "liên doanh", "tập đoàn kinh tế",
    "doanh nhân", "khởi nghiệp", "ươm tạo",

    # Thương mại
    "thương mại", "thương nhân",
    "kinh doanh", "xuất khẩu", "nhập khẩu",
    "xuất nhập khẩu", "ngoại thương",
    "thương mại điện tử",
    "nhượng quyền thương mại",
    "hội chợ triển lãm thương mại",

    # Tài chính - Ngân hàng - Đầu tư
    "đầu tư", "vốn đầu tư",
    "tín dụng", "ngân hàng",
    "chứng khoán", "thị trường chứng khoán",
    "bảo hiểm", "kiểm toán", "kế toán",
    "quỹ đầu tư", "thanh toán",

    # Thuế - Hải quan
    "thuế thu nhập doanh nghiệp",
    "thuế giá trị gia tăng",
    "thuế xuất khẩu", "thuế nhập khẩu",
    "thuế tiêu thụ đặc biệt",
    "quản lý thuế", "hải quan",
    "phí và lệ phí",

    # Lao động
    "lao động", "việc làm",
    "bảo hiểm xã hội", "bảo hiểm thất nghiệp",
    "tiền lương", "tiền công",
    "an toàn vệ sinh lao động",
    "quan hệ lao động",

    # Sở hữu trí tuệ
    "sở hữu trí tuệ",
    "quyền tác giả", "nhãn hiệu", "sáng chế",
    "kiểu dáng công nghiệp",

    # Đất đai kinh doanh / Khu công nghiệp
    "khu công nghiệp", "khu kinh tế",
    "khu chế xuất", "khu công nghệ cao",
    "kinh doanh bất động sản",

    # Cạnh tranh - Tiêu dùng
    "cạnh tranh", "bảo vệ người tiêu dùng",
    "chống độc quyền", "chống bán phá giá",
    "quảng cáo", "khuyến mại",

    # Đấu thầu - Mua sắm
    "đấu thầu", "mua sắm công",

    # Phá sản
    "phá sản",

    # Đăng ký kinh doanh
    "đăng ký kinh doanh", "đăng ký doanh nghiệp",

    # Khoa học - Công nghệ
    "khoa học và công nghệ",
    "công nghệ thông tin",
    "chuyển giao công nghệ",
    "đổi mới sáng tạo",
    "công nghiệp hỗ trợ",

    # Tiêu chuẩn - Chất lượng - ATTP
    "tiêu chuẩn", "quy chuẩn kỹ thuật",
    "chất lượng sản phẩm hàng hóa",
    "an toàn thực phẩm",

    # Vận tải hàng hóa
    "vận tải", "logistics",
    "giao nhận hàng hóa",

    # Xây dựng / Đấu thầu (góc DN)
    "xây dựng công trình",
    "hợp đồng xây dựng",
    "nhà thầu",

    # Năng lượng (góc DN)
    "điện lực", "năng lượng tái tạo",
    "kinh doanh điện",

    # Môi trường (góc DN)
    "giấy phép môi trường",
    "đánh giá tác động môi trường",
]

# ═══════════════════════════════════════════════════════════════
#  TẦNG 2 — LỌC TỪNG ĐIỀU LUẬT (chỉ khi đề mục chưa rõ)
# ═══════════════════════════════════════════════════════════════

ARTICLE_KEEP = [
    "doanh nghiệp", "công ty", "hợp tác xã",
    "hộ kinh doanh", "kinh doanh", "thương mại",
    "đầu tư", "vốn", "thuế", "hải quan",
    "lao động", "người lao động", "tiền lương",
    "bảo hiểm xã hội", "bảo hiểm y tế",
    "sở hữu trí tuệ", "nhãn hiệu", "sáng chế",
    "đấu thầu", "phá sản", "giải thể",
    "đăng ký kinh doanh", "giấy phép kinh doanh",
    "xuất khẩu", "nhập khẩu",
    "chứng khoán", "ngân hàng", "tín dụng",
    "cạnh tranh", "người tiêu dùng",
    "quảng cáo", "khuyến mại",
    "khu công nghiệp", "khu kinh tế",
    "kinh doanh bất động sản",
    "an toàn thực phẩm", "chất lượng hàng hóa",
    "tiêu chuẩn", "quy chuẩn",
    "công nghệ thông tin", "chuyển đổi số",
    "vận tải hàng hóa", "logistics",
    "nhà đầu tư", "vốn góp",
    "hợp đồng kinh tế", "hợp đồng thương mại",
    "phân phối", "bán lẻ", "đại lý",
    "nhập khẩu hàng hóa", "xuất khẩu hàng hóa",
]

# ═══════════════════════════════════════════════════════════════
#  HÀM TIỆN ÍCH
# ═══════════════════════════════════════════════════════════════

def norm(text) -> str:
    return str(text).lower().strip() if pd.notna(text) else ""

def hit(text: str, keywords: list) -> bool:
    t = norm(text)
    return any(kw.lower() in t for kw in keywords)

# ═══════════════════════════════════════════════════════════════
#  PHÂN LOẠI TỪNG HÀNG
# ═══════════════════════════════════════════════════════════════

def classify(row) -> str:
    """
    Nhãn trả về:
      'keep_topic'    → đề mục chắc chắn liên quan DN → giữ cả đề mục
      'keep_article'  → đề mục chưa rõ, điều luật liên quan DN
      'drop_topic'    → đề mục chắc chắn không liên quan → loại cả đề mục
      'drop_article'  → đề mục không rõ, điều luật cũng không liên quan
    """
    topic_text = norm(row.get("subject_title", "")) + " " + norm(row.get("topic_title", ""))

    # Tầng 1a: Đề mục chắc chắn LOẠI → loại ngay cả điều con
    if hit(topic_text, TOPIC_EXCLUDE):
        return "drop_topic"

    # Tầng 1b: Đề mục chắc chắn GIỮ → giữ ngay cả điều con
    if hit(topic_text, TOPIC_KEEP):
        return "keep_topic"

    # Tầng 2: Đề mục chưa rõ → xét nội dung điều luật
    article_text = norm(row.get("article_title", "")) + " " + norm(row.get("content_text", ""))
    if hit(article_text, ARTICLE_KEEP):
        return "keep_article"

    return "drop_article"

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    p = Path(PARQUET_FILE)
    if not p.exists():
        found = sorted(Path(".").glob("**/*.parquet"))
        if not found:
            print(f"[LỖI] Không tìm thấy file: {PARQUET_FILE}")
            return
        p = found[0]
        print(f"[INFO] Dùng file: {p}")

    print(f"Đọc {p.name} ...")
    df = pd.read_parquet(p)
    total = len(df)
    print(f"Tổng bản ghi gốc: {total:,}\n")

    print("Phân loại 2 tầng ...")
    df["_label"] = df.apply(classify, axis=1)
    counts = df["_label"].value_counts()

    df_keep = df[df["_label"].str.startswith("keep")].drop(columns=["_label"])
    df_drop = df[df["_label"].str.startswith("drop")].copy()

    kept  = len(df_keep)
    drop  = len(df_drop)
    pct   = kept / total * 100 if total else 0

    df_keep.to_parquet(OUTPUT_PARQUET, index=False)

    # Thống kê đề mục
    def topic_table(frame, n=40):
        return (
            frame.groupby(["subject_title", "topic_title"])
            .size().reset_index(name="cnt")
            .sort_values("cnt", ascending=False).head(n)
        )

    kept_topics = topic_table(df_keep)
    drop_topics = topic_table(df_drop)

    lines = [
        "=" * 70,
        "  BÁO CÁO LỌC DỮ LIỆU PHÁP ĐIỂN — CHỦ ĐỀ DOANH NGHIỆP",
        "=" * 70,
        f"  File gốc     : {p.name}  ({total:,} bản ghi)",
        f"  File kết quả : {OUTPUT_PARQUET}",
        "",
        f"  ✅ Giữ lại   : {kept:,}  ({pct:.1f}%)",
        f"  ❌ Loại bỏ   : {drop:,}  ({100-pct:.1f}%)",
        "",
        "── PHÂN TẦNG CHI TIẾT ───────────────────────────────────────────",
        f"  keep_topic    đề mục chắc chắn liên quan DN    : {counts.get('keep_topic', 0):>8,}",
        f"  keep_article  điều luật cụ thể liên quan DN    : {counts.get('keep_article', 0):>8,}",
        f"  drop_topic    đề mục chắc chắn không liên quan : {counts.get('drop_topic', 0):>8,}",
        f"  drop_article  điều luật không liên quan        : {counts.get('drop_article', 0):>8,}",
        "",
        "── ĐỀ MỤC ĐƯỢC GIỮ (top 40) ────────────────────────────────────",
    ]
    for _, r in kept_topics.iterrows():
        s = str(r["subject_title"])[:32]
        t = str(r["topic_title"])[:32]
        lines.append(f"  [{r['cnt']:>5,}]  {s:<34} › {t}")

    lines += ["", "── ĐỀ MỤC BỊ LOẠI (top 40) ─────────────────────────────────────"]
    for _, r in drop_topics.iterrows():
        s = str(r["subject_title"])[:32]
        t = str(r["topic_title"])[:32]
        lines.append(f"  [{r['cnt']:>5,}]  {s:<34} › {t}")

    lines.append("=" * 70)
    report = "\n".join(lines)

    print("\n" + report)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Đã lưu: {OUTPUT_PARQUET}")
    print(f"📋 Báo cáo: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()