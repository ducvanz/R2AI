"""
02_retriever.py — Stage 1: Hybrid Retrieval (BM25 + Dense)

Áp dụng Reciprocal Rank Fusion (RRF) để kết hợp BM25 và Dense,
trả về Top-K provision candidates cho Stage 2 (Reranking).

Chạy độc lập:
  python src/02_retriever.py --query "Điều kiện để DNNVV được hỗ trợ là gì?"
  python src/02_retriever.py --query "..." --top_k 50
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, MODEL_DIR, format_result, print_results

# ── Paths ───────────────────────────────────────────────────────────────────
BM25_INDEX    = DATA_DIR / "bm25_index.pkl"
DENSE_INDEX   = DATA_DIR / "dense_index.npy"
PROVISION_IDS = DATA_DIR / "provision_ids.json"
OUT_PARQUET   = DATA_DIR / "provisions.parquet"
FAISS_INDEX   = DATA_DIR / "dense_faiss.index"

DENSE_MODEL   = "intfloat/multilingual-e5-small"
CACHE_DIR     = str(MODEL_DIR)

# ── RRF ─────────────────────────────────────────────────────────────────────
RRF_K = 60   # standard constant


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    rankings: list of ranked lists of doc indices.
    Returns: sorted list of (doc_idx, rrf_score) descending.
    """
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, doc_idx in enumerate(ranked):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Retriever class ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid BM25 + Dense retriever với RRF fusion.
    Theo thiết kế của bài báo: first-stage retrieval để build candidate pool.
    """

    def __init__(self):
        self._bm25 = None
        self._dense_matrix = None
        self._dense_model = None
        self._faiss_index  = None
        self._provisions: list[dict] = []
        self._df: pd.DataFrame | None = None

    def load(self):
        """Load tất cả index và metadata."""
        print("[Retriever] Loading indexes ...")

        # Provisions metadata
        with open(PROVISION_IDS, encoding="utf-8") as f:
            self._provisions = json.load(f)

        # Parquet (for full row access)
        self._df = pd.read_parquet(OUT_PARQUET)

        # BM25
        if BM25_INDEX.exists():
            with open(BM25_INDEX, "rb") as f:
                self._bm25 = pickle.load(f)
            print(f"  ✓ BM25 loaded ({len(self._provisions):,} provisions)")
        else:
            print("  ⚠️  BM25 index not found. Run 01_preprocess.py first.")

        # Dense
        # if DENSE_INDEX.exists():
        #     self._dense_matrix = np.load(DENSE_INDEX).astype(np.float32)
        #     print(f"  ✓ Dense index loaded {self._dense_matrix.shape}")
        # else:
        #     print("  ⚠️  Dense index not found. Run 01_preprocess.py first.")

        import faiss

        if FAISS_INDEX.exists():
            self._faiss_index = faiss.read_index(str(FAISS_INDEX))

            # Với IVFFlat cần set nprobe (trade-off speed vs recall)
            # nprobe=1  → nhanh nhất, recall thấp hơn
            # nprobe=10 → cân bằng tốt (khuyến nghị)
            # nprobe=50 → gần exact search
            if hasattr(self._faiss_index, 'nprobe'):
                self._faiss_index.nprobe = 10

            print(f"  ✓ FAISS index loaded ({self._faiss_index.ntotal:,} vectors)")
        elif DENSE_INDEX.exists():
            # Fallback: vẫn load numpy nếu chưa build FAISS
            self._dense_matrix = np.load(DENSE_INDEX).astype(np.float32)
            print(f"  ⚠️  FAISS not found, fallback numpy {self._dense_matrix.shape}")
        else:
            print("  ⚠️  No dense index found. Run 01_preprocess.py first.")

        # Dense encoder
        try:
            from sentence_transformers import SentenceTransformer
            self._dense_model = SentenceTransformer(
                DENSE_MODEL, cache_folder=CACHE_DIR
            )
            print(f"  ✓ Dense model loaded: {DENSE_MODEL}")
        except Exception as e:
            print(f"  ⚠️  Could not load dense model: {e}")

        return self

    # ── BM25 retrieval ───────────────────────────────────────────────────────

    def _retrieve_bm25(self, query: str, top_k: int) -> list[int]:
        if self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return ranked.tolist()

    # ── Dense retrieval ──────────────────────────────────────────────────────

    # def _retrieve_dense(self, query: str, top_k: int) -> list[int]:
    #     if self._dense_matrix is None or self._dense_model is None:
    #         return []
    #     # multilingual-e5: prefix "query: " for questions
    #     q_emb = self._dense_model.encode(
    #         "query: " + query,
    #         normalize_embeddings=True,
    #         convert_to_numpy=True,
    #     ).astype(np.float32)

    #     # Cosine similarity (normalized) = dot product
    #     sims = self._dense_matrix @ q_emb  # shape: (N,)
    #     ranked = np.argsort(sims)[::-1][:top_k]
    #     return ranked.tolist()



    def _retrieve_dense(self, query: str, top_k: int) -> list[int]:
        if self._dense_model is None:
            return []

        # Encode query
        q_emb = self._dense_model.encode(
            "query: " + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        q_emb = q_emb.reshape(1, -1)   # FAISS cần shape (1, dim)

        if self._faiss_index is not None:
            # ✅ FAISS search — O(log N), nhanh hơn 10-50x
            _scores, indices = self._faiss_index.search(q_emb, top_k)
            # indices shape: (1, top_k) — lấy hàng đầu tiên
            return indices[0].tolist()
        elif self._dense_matrix is not None:
            # Fallback numpy
            sims = self._dense_matrix @ q_emb[0]
            ranked = np.argsort(sims)[::-1][:top_k]
            return ranked.tolist()
        else:
            return []

    # ── Hybrid ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
        mode: str = "hybrid",   # "hybrid" | "bm25" | "dense"
    ) -> list[dict]:
        """
        Trả về top_k provisions dưới dạng list[dict] với key 'score' và
        toàn bộ fields của provision.
        
        mode:
          - "hybrid" : RRF fusion BM25 + Dense (mặc định, tốt nhất)
          - "bm25"   : chỉ BM25
          - "dense"  : chỉ Dense
        """
        if mode == "bm25":
            bm25_ranked = self._retrieve_bm25(query, bm25_top_k)
            fused = [(idx, 1.0 / (i + 1)) for i, idx in enumerate(bm25_ranked)]
        elif mode == "dense":
            dense_ranked = self._retrieve_dense(query, dense_top_k)
            fused = [(idx, 1.0 / (i + 1)) for i, idx in enumerate(dense_ranked)]
        else:  # hybrid
            bm25_ranked  = self._retrieve_bm25(query, bm25_top_k)
            dense_ranked = self._retrieve_dense(query, dense_top_k)
            fused = rrf_fuse([bm25_ranked, dense_ranked])

        # Build result list
        results = []
        seen = set()
        for idx, score in fused[:top_k]:
            if idx in seen or idx >= len(self._provisions):
                continue
            seen.add(idx)
            row = self._provisions[idx]
            # Merge full DataFrame row for complete fields
            if self._df is not None and idx < len(self._df):
                full_row = self._df.iloc[idx].to_dict()
                row = {**row, **full_row}
            results.append({**row, "_retrieval_score": float(score)})

        return results

    def get_provision_text(self, idx: int) -> str:
        """Lấy provision_text của 1 provision theo index."""
        if self._df is not None and idx < len(self._df):
            return self._df.iloc[idx].get("provision_text_full", "")
        return ""


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Hybrid Retrieval")
    parser.add_argument("--query", required=True, help="Câu hỏi pháp luật")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Số provisions trả về (default: 10)")
    parser.add_argument("--mode", default="hybrid",
                        choices=["hybrid", "bm25", "dense"],
                        help="Chế độ retrieval")
    args = parser.parse_args()

    retriever = HybridRetriever().load()
    candidates = retriever.retrieve(args.query, top_k=args.top_k, mode=args.mode)

    results = [
        format_result(c, c["_retrieval_score"], i + 1)
        for i, c in enumerate(candidates)
    ]
    print_results(results, query=args.query)
    print(f"\nMode: {args.mode} | Returned: {len(results)} provisions")


if __name__ == "__main__":
    main()