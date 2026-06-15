"""
build_faiss.py — Convert dense_index.npy → FAISS index (chạy 1 lần duy nhất)
Chạy: python src/build_faiss.py
"""
import sys
from pathlib import Path
import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR

DENSE_INDEX = DATA_DIR / "dense_index.npy"
FAISS_INDEX = DATA_DIR / "dense_faiss.index"

print("Loading dense_index.npy ...")
embeddings = np.load(DENSE_INDEX).astype(np.float32)
N, dim = embeddings.shape
print(f"  ✓ Loaded: {N:,} vectors, dim={dim}")

# Chọn loại index
if N < 100_000:
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  ✓ IndexFlatIP built")
else:
    nlist = min(int(N ** 0.5), 4096)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    print(f"  ✓ IndexIVFFlat built (nlist={nlist})")

faiss.write_index(index, str(FAISS_INDEX))
print(f"  ✓ Saved → {FAISS_INDEX}")