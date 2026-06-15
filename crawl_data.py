"""
Phân tích và Trực quan hóa Dữ liệu Pháp điển Việt Nam
Dataset: tmquan/phapdien-moj-gov-vn (Bộ Tư pháp)

Cột thực tế:
  subject_id, topic_id, topic_number, topic_title,
  subject_number, subject_title, article_anchor, article_title,
  chapter_title, source_note_text, source_links, related_note_text,
  content_text, content_char_len, content_word_count,
  source_url, scraped_at

Yêu cầu:
    pip install datasets pandas matplotlib seaborn wordcloud tqdm

Chạy:
    python phapdien_analysis.py
"""

import warnings, sys
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # không cần GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False
    print("[INFO] wordcloud chưa cài – bỏ qua WordCloud.  pip install wordcloud")

# ── Thư mục đầu ra ──────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("output_charts")
CACHE_FILE  = Path("phapdien_cache.parquet")   # file cache local
OUTPUT_DIR.mkdir(exist_ok=True)

PALETTE = ["#D62728","#1F77B4","#2CA02C","#FF7F0E",
           "#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi":     120,
})

# ════════════════════════════════════════════════════════════════════════════
# 1. TẢI / ĐỌC DỮ LIỆU
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print("  PHÂN TÍCH DỮ LIỆU PHÁP ĐIỂN VIỆT NAM")
print(f"{'='*62}\n")

if CACHE_FILE.exists():
    print(f"[1/5] Tìm thấy cache → đọc từ  {CACHE_FILE}  (bỏ qua download)")
    df = pd.read_parquet(CACHE_FILE)
    print(f"      Đã đọc {len(df):,} bản ghi từ cache.\n")
else:
    print("[1/5] Chưa có cache – tải từ HuggingFace …")
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[LỖI] Chưa cài datasets:  pip install datasets")

    try:
        ds = load_dataset("tmquan/phapdien-moj-gov-vn", trust_remote_code=True)
    except Exception as e:
        sys.exit(f"[LỖI] Không tải được dataset: {e}")

    frames = []
    for split_name, split_data in ds.items():
        tmp = split_data.to_pandas()
        tmp["split"] = split_name
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)

    # --- Lưu cache (chuyển list/array → string để parquet không lỗi) -------
    df_save = df.copy()
    for col in df_save.columns:
        if df_save[col].dtype == object:
            sample = df_save[col].dropna().head(5)
            if any(isinstance(v, (list, np.ndarray)) for v in sample):
                df_save[col] = df_save[col].apply(
                    lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x
                )
    df_save.to_parquet(CACHE_FILE, index=False)
    print(f"      ✓  Đã lưu cache → {CACHE_FILE}")
    print(f"      Tổng bản ghi: {len(df):,}\n")

# ════════════════════════════════════════════════════════════════════════════
# 2. CHUẨN HOÁ / ĐẶT TÊN CỘT
# ════════════════════════════════════════════════════════════════════════════
print("[2/5] Chuẩn hoá dữ liệu …")

# Ánh xạ cột thực tế của dataset này
COL_TOPIC_TITLE   = "topic_title"      # Chủ đề lớn  (Đề mục)
COL_SUBJECT_TITLE = "subject_title"    # Chủ đề con  (Vấn đề)
COL_CHAPTER       = "chapter_title"    # Chương
COL_ARTICLE       = "article_title"    # Điều
COL_CONTENT       = "content_text"     # Nội dung điều luật
COL_CHAR_LEN      = "content_char_len" # Độ dài ký tự
COL_WORD_COUNT    = "content_word_count"
COL_SCRAPED       = "scraped_at"       # Ngày crawl

# Chuyển kiểu số an toàn
for num_col in [COL_CHAR_LEN, COL_WORD_COUNT]:
    if num_col in df.columns:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

# Chuyển cột list/array → string để tránh lỗi nunique / groupby
for col in df.columns:
    if df[col].dtype == object:
        sample = df[col].dropna().head(10)
        if any(isinstance(v, (list, np.ndarray)) for v in sample):
            df[col] = df[col].apply(
                lambda x: "; ".join(map(str, x)) if isinstance(x, (list, np.ndarray)) else x
            )

# Trích năm từ scraped_at
if COL_SCRAPED in df.columns:
    df["_year"] = pd.to_datetime(df[COL_SCRAPED], errors="coerce").dt.year

print(f"  Cột có sẵn : {list(df.columns)}")
print(f"  Bản ghi    : {len(df):,}\n")

# ════════════════════════════════════════════════════════════════════════════
# 3. THỐNG KÊ TỔNG QUAN
# ════════════════════════════════════════════════════════════════════════════
print("[3/5] Thống kê tổng quan …\n")
print(f"{'─'*50}")
print(f"  Tổng số điều / khoản luật        : {len(df):>8,}")

n_topics   = df[COL_TOPIC_TITLE].nunique()   if COL_TOPIC_TITLE   in df.columns else "N/A"
n_subjects = df[COL_SUBJECT_TITLE].nunique() if COL_SUBJECT_TITLE in df.columns else "N/A"
n_chapters = df[COL_CHAPTER].nunique()       if COL_CHAPTER       in df.columns else "N/A"

print(f"  Số đề mục  (topic)                : {n_topics:>8,}")
print(f"  Số vấn đề  (subject)              : {n_subjects:>8,}")
print(f"  Số chương  (chapter)              : {n_chapters:>8,}")

if COL_CHAR_LEN in df.columns:
    print(f"  Độ dài TB (ký tự / điều)         : {df[COL_CHAR_LEN].mean():>10,.0f}")
    print(f"  Độ dài TB (từ   / điều)          : {df[COL_WORD_COUNT].mean():>10,.0f}")

print(f"{'─'*50}\n")

# ════════════════════════════════════════════════════════════════════════════
# 4. VẼ BIỂU ĐỒ
# ════════════════════════════════════════════════════════════════════════════
print("[4/5] Vẽ biểu đồ …")
saved = []

def savefig(fig, name):
    p = OUTPUT_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    print(f"  ✓  {p}")

# ── 4a. Top 20 Đề mục (topic_title) ─────────────────────────────────────────
vc_topic = df[COL_TOPIC_TITLE].fillna("Không rõ").value_counts().head(20)
fig, ax = plt.subplots(figsize=(11, 8))
colors = PALETTE * (len(vc_topic)//len(PALETTE)+1)
bars = ax.barh(vc_topic.index[::-1], vc_topic.values[::-1], color=colors[:len(vc_topic)])
for bar, val in zip(bars, vc_topic.values[::-1]):
    ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
            f"{val:,}", va="center", fontsize=8)
ax.set_xlabel("Số điều / khoản")
ax.set_title("Top 20 Đề Mục Pháp Điển (topic_title)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
savefig(fig, "01_top_de_muc.png")

# ── 4b. Top 20 Vấn đề (subject_title) ───────────────────────────────────────
vc_subj = df[COL_SUBJECT_TITLE].fillna("Không rõ").value_counts().head(20)
fig, ax = plt.subplots(figsize=(11, 8))
sns.barplot(x=vc_subj.values, y=vc_subj.index, palette="Blues_r", ax=ax)
ax.set_xlabel("Số điều / khoản")
ax.set_title("Top 20 Vấn Đề Pháp Điển (subject_title)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
fig.tight_layout()
savefig(fig, "02_top_van_de.png")

# ── 4c. Phân bố số điều theo đề mục (Treemap giả lập bằng bar) ──────────────
vc_full = df[COL_TOPIC_TITLE].fillna("Không rõ").value_counts()
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(range(len(vc_full)), vc_full.values,
       color=[PALETTE[i % len(PALETTE)] for i in range(len(vc_full))])
ax.set_xticks([])
ax.set_ylabel("Số điều / khoản")
ax.set_title(f"Phân Bố Số Điều Luật Trên {len(vc_full)} Đề Mục Pháp Điển")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax.grid(axis="y", alpha=0.3)
# Ghi nhãn top 5
for i, (idx, val) in enumerate(vc_full.head(5).items()):
    short = idx[:30]+"…" if len(idx)>30 else idx
    ax.text(i, val+10, short, rotation=20, fontsize=7, ha="left")
fig.tight_layout()
savefig(fig, "03_phan_bo_de_muc.png")

# ── 4d. Histogram độ dài ký tự ───────────────────────────────────────────────
if COL_CHAR_LEN in df.columns:
    vals = df[COL_CHAR_LEN].dropna()
    p99  = vals.quantile(0.99)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(vals[vals<=p99], bins=80, color=PALETTE[1],
                 edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Số ký tự")
    axes[0].set_ylabel("Số điều")
    axes[0].set_title("Phân Bố Độ Dài Văn Bản (≤ P99)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))

    axes[1].hist(vals[vals<=p99], bins=80, color=PALETTE[2],
                 edgecolor="white", alpha=0.85, cumulative=True, density=True)
    axes[1].set_xlabel("Số ký tự")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF Độ Dài Văn Bản")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))

    fig.tight_layout()
    savefig(fig, "04_histogram_do_dai.png")

# ── 4e. Box-plot độ dài theo top 10 đề mục ──────────────────────────────────
if COL_CHAR_LEN in df.columns:
    top10_topics = df[COL_TOPIC_TITLE].value_counts().head(10).index
    sub = df[df[COL_TOPIC_TITLE].isin(top10_topics)].copy()
    # rút gọn nhãn
    sub["_short_topic"] = sub[COL_TOPIC_TITLE].apply(
        lambda x: x[:35]+"…" if len(str(x))>35 else x)
    order = sub.groupby("_short_topic")[COL_CHAR_LEN].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=sub, x="_short_topic", y=COL_CHAR_LEN,
                order=order, palette="Set2", ax=ax,
                flierprops={"marker":".", "markersize":2, "alpha":0.3})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("Số ký tự")
    ax.set_title("Phân Bố Độ Dài Theo Top 10 Đề Mục")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    fig.tight_layout()
    savefig(fig, "05_boxplot_de_muc.png")

# ── 4f. Heatmap: Top-15 đề mục × Top-10 vấn đề ──────────────────────────────
top15t = df[COL_TOPIC_TITLE].value_counts().head(15).index
top10s = df[COL_SUBJECT_TITLE].value_counts().head(10).index
heat = (df[df[COL_TOPIC_TITLE].isin(top15t) & df[COL_SUBJECT_TITLE].isin(top10s)]
        .groupby([COL_TOPIC_TITLE, COL_SUBJECT_TITLE])
        .size().unstack(fill_value=0))
if not heat.empty:
    # rút gọn nhãn
    heat.index   = [x[:30]+"…" if len(x)>30 else x for x in heat.index]
    heat.columns = [x[:28]+"…" if len(x)>28 else x for x in heat.columns]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heat, cmap="YlOrRd", linewidths=0.4,
                annot=True, fmt="d", ax=ax,
                cbar_kws={"label":"Số điều"})
    ax.set_xlabel("Vấn đề (subject)")
    ax.set_ylabel("Đề mục (topic)")
    ax.set_title("Heatmap: Đề Mục × Vấn Đề – Số Điều Luật")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    savefig(fig, "06_heatmap_topic_subject.png")

# ── 4g. Pie chart: tỷ lệ điều có / không có source_links ────────────────────
if "source_links" in df.columns:
    has_link = df["source_links"].apply(
        lambda x: bool(x) and str(x) not in ("", "[]", "nan", "None")
    )
    counts = has_link.value_counts()
    labels = ["Có nguồn tham chiếu", "Không có nguồn"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%",
           colors=[PALETTE[2], PALETTE[7]], startangle=90,
           wedgeprops={"edgecolor":"white","linewidth":2})
    ax.set_title("Tỷ Lệ Điều Luật Có Nguồn Tham Chiếu")
    savefig(fig, "07_pie_source_links.png")

# ── 4h. Biểu đồ số điều theo số từ (word count bucket) ──────────────────────
if COL_WORD_COUNT in df.columns:
    wc_vals = df[COL_WORD_COUNT].dropna()
    bins  = [0, 50, 100, 200, 400, 800, float("inf")]
    labels_b = ["≤50", "51–100", "101–200", "201–400", "401–800", ">800"]
    df["_wc_bucket"] = pd.cut(wc_vals, bins=bins, labels=labels_b)
    bucket_counts = df["_wc_bucket"].value_counts().reindex(labels_b)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels_b, bucket_counts.values,
                  color=[PALETTE[i] for i in range(len(labels_b))])
    for bar, val in zip(bars, bucket_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Số từ trong điều luật")
    ax.set_ylabel("Số điều")
    ax.set_title("Phân Nhóm Điều Luật Theo Số Từ")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "08_word_count_bucket.png")

# ── 4i. WordCloud tiêu đề đề mục ─────────────────────────────────────────────
if HAS_WC:
    blob = " ".join(df[COL_TOPIC_TITLE].dropna().astype(str).tolist())
    stops = {"về","của","và","các","trong","theo","số","ngày","năm","quy",
             "định","ban","hành","thực","hiện","việt","nam","luật","điều"}
    wc = WordCloud(width=1400, height=700, background_color="white",
                   colormap="RdYlBu", stopwords=stops,
                   max_words=200, collocations=False).generate(blob)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("WordCloud – Từ Khoá Đề Mục Pháp Điển", fontsize=16, pad=12)
    fig.tight_layout()
    savefig(fig, "09_wordcloud_topic.png")

# ── 4j. Bảng CSV tóm tắt ─────────────────────────────────────────────────────
rows = []
for col in df.columns:
    if col.startswith("_"): continue
    try:
        n_null = int(df[col].isna().sum())
        n_uniq = int(df[col].nunique())
        top_v  = str(df[col].value_counts().index[0])[:60] if n_uniq else ""
    except Exception:
        n_null, n_uniq, top_v = -1, -1, "N/A"
    rows.append({"Cột": col, "Kiểu": str(df[col].dtype),
                 "Null": n_null, "Duy nhất": n_uniq, "Giá trị phổ biến nhất": top_v})
csv_path = OUTPUT_DIR / "00_tom_tat.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
saved.append(csv_path)
print(f"  ✓  {csv_path}")

# ════════════════════════════════════════════════════════════════════════════
# 5. TỔNG KẾT
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[5/5] Hoàn tất!\n{'='*62}")
print("  FILE ĐÃ LƯU")
print(f"{'='*62}")
for f in saved:
    print(f"  📄  {f}")
print(f"\n  Cache dataset → {CACHE_FILE}  (lần sau không cần tải lại)")
print(f"  Biểu đồ      → ./{OUTPUT_DIR}/")
print(f"{'='*62}\n")