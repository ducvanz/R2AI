"""
01_preprocess.py — Tiền xử lý dữ liệu và xây dựng index

Đầu vào : data/data.parquet
Đầu ra  :
  - data/provisions.parquet   (dữ liệu đã làm sạch + provision_text)
  - data/bm25_index.pkl       (BM25 index)
  - data/dense_index.npy      (Dense embeddings matrix)
  - data/provision_ids.json   (mapping index → row id)

Chạy: python src/01_preprocess.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATA_DIR, MODEL_DIR,
    build_provision_text, normalize_text
)

# ── Config ─────────────────────────────────────────────────────────────────
PARQUET_FILE  = DATA_DIR / "data_business.parquet"
OUT_PARQUET   = DATA_DIR / "provisions.parquet"
BM25_INDEX    = DATA_DIR / "bm25_index.pkl"
DENSE_INDEX   = DATA_DIR / "dense_index.npy"
PROVISION_IDS = DATA_DIR / "provision_ids.json"

DENSE_MODEL        = "intfloat/multilingual-e5-small"  # ~120MB, nhanh hơn base (~470MB)
DENSE_MODEL_LARGE  = "intfloat/multilingual-e5-base"   # dùng khi RAM >= 16GB
BATCH_SIZE         = 32   # nhỏ hơn để tránh OOM
CHUNK_SIZE         = 5000 # encode từng chunk, lưu checkpoint
CACHE_DIR          = str(MODEL_DIR)

# ── Step 1: Load & Clean ────────────────────────────────────────────────────

def load_and_clean(path: Path) -> pd.DataFrame:
    print(f"[1/4] Loading {path} ...")
    df = pd.read_parquet(path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Chuẩn hoá text
    for col in ["article_title", "content_text", "chapter_title",
                "source_note_text", "topic_title", "subject_title"]:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(normalize_text)

    # Lọc provision có content
    before = len(df)
    df = df[df["content_text"].str.len() > 10].reset_index(drop=True)
    print(f"  After filter (content > 10 chars): {len(df):,} (removed {before - len(df):,})")

    # Tạo provision_text cho indexing (full) và reranking (short)
    df["provision_text_full"]  = df.apply(
        lambda r: build_provision_text(r, mode="full"), axis=1
    )
    df["provision_text_short"] = df.apply(
        lambda r: build_provision_text(r, mode="short"), axis=1
    )

    # Unique index
    df["_idx"] = df.index.astype(int)

    return df


# ── Step 2: BM25 Index ──────────────────────────────────────────────────────

def build_bm25(df: pd.DataFrame) -> None:
    print("[2/4] Building BM25 index ...")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("  ⚠️  rank-bm25 not installed. Run: pip install rank-bm25")
        return

    # Tokenize: đơn giản split theo khoảng trắng (tiếng Việt tách từ đủ tốt)
    corpus = [text.lower().split() for text in df["provision_text_full"]]
    bm25 = BM25Okapi(corpus)

    with open(BM25_INDEX, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved → {BM25_INDEX}")


# ── Step 3: Dense Embeddings ────────────────────────────────────────────────

def build_dense(df: pd.DataFrame) -> None:
    """
    Encode tất cả provisions thành dense vectors.
    - Dùng chunked encoding để tránh OOM và có thể resume nếu bị ngắt
    - Checkpoint từng chunk ra disk, ghép lại ở cuối
    - Fallback sang model nhỏ hơn nếu model lớn không load được
    """
    try:
        print(0)
        from sentence_transformers import SentenceTransformer
        print(1)
    except ImportError:
        print("  sentence-transformers not installed. Bỏ qua dense index.")
        return

    # Thử load model, fallback nếu lỗi
    model = None
    print(2)
    for model_name in [DENSE_MODEL, DENSE_MODEL_LARGE, "paraphrase-multilingual-MiniLM-L12-v2"]:
        try:
            print(f"[3/4] Loading dense model: {model_name} ...")
            print("      (Lần đầu sẽ download model, có thể mất vài phút)")
            model = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
            print(f"  ✓ Model loaded: {model_name}")
            break
        except Exception as e:
            print(f"  ✗ Failed ({model_name}): {e}")

    if model is None:
        print("  ⚠️  Không load được bất kỳ model nào. Bỏ qua dense index.")
        print("       Hệ thống vẫn hoạt động với BM25 only.")
        return

    texts = ["passage: " + t for t in df["provision_text_full"].tolist()]
    n_total = len(texts)
    print(f"  Encoding {n_total:,} provisions (batch={BATCH_SIZE}, chunk={CHUNK_SIZE}) ...")

    # Chunked encoding với checkpoint
    chunk_dir = DATA_DIR / "_dense_chunks"
    chunk_dir.mkdir(exist_ok=True)

    all_embeddings = []
    n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_i in range(n_chunks):
        chunk_path = chunk_dir / f"chunk_{chunk_i:04d}.npy"

        # Resume: bỏ qua chunk đã xong
        if chunk_path.exists():
            print(f"  chunk {chunk_i+1}/{n_chunks}: resume từ cache ✓")
            all_embeddings.append(np.load(chunk_path))
            continue

        start = chunk_i * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, n_total)
        chunk_texts = texts[start:end]

        print(f"  chunk {chunk_i+1}/{n_chunks}: rows {start}–{end} ...", flush=True)
        try:
            emb = model.encode(
                chunk_texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            emb = emb.astype(np.float32)
            np.save(chunk_path, emb)
            all_embeddings.append(emb)
            print(f"    → {emb.shape}, saved checkpoint")
        except MemoryError:
            print(f"  ✗ OOM tại chunk {chunk_i}. Thử giảm BATCH_SIZE hoặc CHUNK_SIZE.")
            raise
        except Exception as e:
            print(f"  ✗ Lỗi tại chunk {chunk_i}: {e}")
            raise

    # Ghép tất cả chunks
    embeddings = np.vstack(all_embeddings)
    np.save(DENSE_INDEX, embeddings)
    print(f"  ✓ Dense index saved → {DENSE_INDEX}  shape={embeddings.shape}")

    # Dọn chunks tạm
    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)


# ── Step 4: Save metadata ───────────────────────────────────────────────────

def save_metadata(df: pd.DataFrame) -> None:
    print("[4/4] Saving metadata ...")

    # Lưu parquet đã xử lý
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved → {OUT_PARQUET}")

    # Lưu mapping index → provision info (JSON, dễ load)
    id_map = df[["_idx", "article_title", "subject_title", "topic_title",
                 "source_note_text", "content_text", "source_url",
                 "chapter_title"]].to_dict(orient="records")
    with open(PROVISION_IDS, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {PROVISION_IDS} ({len(id_map):,} provisions)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not PARQUET_FILE.exists():
        print(f"ERROR: {PARQUET_FILE} not found.")
        print("  → Đặt file data.parquet vào thư mục data/")
        sys.exit(1)

    df = load_and_clean(PARQUET_FILE)
    # build_bm25(df)
    build_dense(df)
    save_metadata(df)

    print("\n✅ Preprocessing complete!")
    print(f"   Provisions indexed: {len(df):,}")


if __name__ == "__main__":
    main()