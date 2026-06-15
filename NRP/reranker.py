"""
03_reranker.py — Stage 2: Neural Reranking (Cross-Encoder)

Áp dụng phương pháp từ bài báo:
  "Neural reranking for UK statutory retrieval: Provision-level evaluation
   and an open distilled model" (2026)

Cross-encoder nhận cặp (query, provision_text) và trả ra relevance score.
Đây là "pointwise reranking" — score từng cặp độc lập.

Hỗ trợ 2 mô hình:
  - "minilm"  : cross-encoder/ms-marco-MiniLM-L-6-v2  (nhanh, ~80MB)
  - "bge"     : BAAI/bge-reranker-v2-m3               (mạnh hơn, multilingual, ~1.1GB)

Chạy độc lập:
  python src/03_reranker.py --query "..." --model bge
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, MODEL_DIR, build_provision_text, format_result, print_results

CACHE_DIR = str(MODEL_DIR)

# ── Model registry ──────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # Cross-encoder trained on MS-MARCO — fast, ~80MB
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-base": "BAAI/bge-reranker-base",
    # BGE multilingual reranker — best for Vietnamese, ~1.1GB
    "bge": "BAAI/bge-reranker-v2-m3",
    # Larger BGE
    "bge-large": "BAAI/bge-reranker-large",
}


# ── CrossEncoder Reranker ───────────────────────────────────────────────────

class NeuralReranker:
    """
    Neural Cross-Encoder Reranker.
    
    Theo bài báo: cross-encoder consistently outperforms first-stage BM25/dense
    trên cả nDCG và MRR ở provision-level retrieval.
    
    Input : query (str) + list of candidates (list[dict])
    Output: sorted list[dict] với rerank_score
    """

    def __init__(self, model_name: str = "bge"):
        self.model_key = model_name
        self.model_name = MODEL_REGISTRY.get(model_name, model_name)
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Reranker] Device: {self._device}")

    def load(self):
        """Load cross-encoder model."""
        print(f"[Reranker] Loading {self.model_name} ...")
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self._device,
                # Tắt autocast warning trên CPU
                default_activation_function=None,
            )
            print(f"  ✓ Cross-encoder loaded: {self.model_name}")
        except Exception as e:
            print(f"  ✗ Failed to load cross-encoder: {e}")
            print("    Falling back to no reranking.")
        return self

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Rerank candidates với cross-encoder.
        
        Mỗi cặp (query, provision_text) được score độc lập.
        Provision_text được build từ title + content (mode="rerank").
        """
        if not candidates:
            return []

        if self._model is None:
            # Fallback: trả về theo thứ tự ban đầu
            print("  ⚠️  No reranker loaded, returning retrieval order.")
            return candidates[:top_k]

        # Build (query, text) pairs
        pairs = []
        for c in candidates:
            text = build_provision_text(c, mode="rerank")
            pairs.append((query, text))

        # Score tất cả pairs
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=len(pairs) > 20,
            convert_to_numpy=True,
        )

        # Gắn score và sort
        for i, c in enumerate(candidates):
            c["_rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["_rerank_score"], reverse=True)
        return reranked[:top_k]


# ── LLM Listwise Reranker (tuỳ chọn, dùng Gemma3 qua Ollama) ──────────────

class LLMListwiseReranker:
    """
    Listwise reranker dùng LLM (gemma3:4b qua Ollama).
    
    Theo bài báo: listwise reranking cho phép mô hình so sánh nhiều provisions
    cùng lúc, thường tốt hơn pointwise nhưng tốn kém hơn.
    
    Chỉ dùng khi cross-encoder không đủ tốt.
    """

    SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ của bạn là xếp hạng
các điều luật theo mức độ liên quan đến câu hỏi pháp lý được đưa ra.

Trả lời CHỈ bằng danh sách các số thứ tự theo thứ tự liên quan giảm dần,
cách nhau bằng dấu phẩy. Ví dụ: 3,1,5,2,4"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:4b"):
        self.base_url = base_url
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
        window_size: int = 10,
    ) -> list[dict]:
        """
        Listwise reranking với sliding window (nếu candidates > window_size).
        """
        import requests

        if not candidates:
            return []

        # Build candidate list text
        items = []
        for i, c in enumerate(candidates[:window_size]):
            title = c.get("article_title", "")
            content = c.get("content_text", "")[:300]
            items.append(f"[{i+1}] {title}: {content}")

        candidates_text = "\n".join(items)
        user_prompt = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{candidates_text}

Xếp hạng các điều luật theo mức độ liên quan (chỉ trả về danh sách số):"""

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip()

            # Parse ranking
            indices = []
            for part in answer.replace(" ", "").split(","):
                try:
                    idx = int(part) - 1  # 1-based → 0-based
                    if 0 <= idx < len(candidates[:window_size]):
                        indices.append(idx)
                except ValueError:
                    continue

            # Reorder
            seen = set(indices)
            remaining = [i for i in range(len(candidates[:window_size])) if i not in seen]
            final_order = indices + remaining

            reranked = [candidates[i] for i in final_order]
            # Add pseudo-scores
            for i, c in enumerate(reranked):
                c["_rerank_score"] = float(len(reranked) - i) / len(reranked)

            return reranked[:top_k]

        except Exception as e:
            print(f"  ⚠️  LLM reranking failed: {e}. Returning original order.")
            return candidates[:top_k]


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Neural Reranking")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="bge",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Reranker model")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Chạy full pipeline để test reranker
    from retreiver import HybridRetriever
    retriever = HybridRetriever().load()
    candidates = retriever.retrieve(args.query, top_k=50)

    reranker = NeuralReranker(model_name=args.model).load()
    reranked = reranker.rerank(args.query, candidates, top_k=args.top_k)

    results = [
        format_result(c, c.get("_rerank_score", 0.0), i + 1)
        for i, c in enumerate(reranked)
    ]
    print_results(results, query=args.query)


if __name__ == "__main__":
    main()