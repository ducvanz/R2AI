"""
04_pipeline.py — Pipeline đầy đủ: Query → Retrieval → Reranking → Results

Đây là entry point chính cho hệ thống truy vấn luật.

Chạy:
  python src/04_pipeline.py --query_file data/query.json
  python src/04_pipeline.py --query_file data/query.json --top_k 5 --reranker bge
  python src/04_pipeline.py --query "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào?"
  python src/04_pipeline.py --query "..." --top_k 5 --reranker bge
  python src/04_pipeline.py --query "..." --output results/my_query.json
  python src/04_pipeline.py --batch data/queries.txt   # batch mode
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, RESULTS_DIR, extract_law_code, format_result, print_results


# ── Pipeline Class ──────────────────────────────────────────────────────────

class LegalRAGPipeline:
    """
    2-Stage Pipeline:
      Stage 1: Hybrid Retrieval (BM25 + Dense, RRF fusion)
      Stage 2: Neural Cross-Encoder Reranking (provision-level)

    Thiết kế theo bài báo:
      - First-stage: BM25 hoặc Dense → Top-100 candidates
      - Second-stage: Cross-encoder reranks → Top-K final results
    """

    def __init__(
        self,
        reranker_model: str = "bge",
        retrieval_mode: str = "hybrid",
        retrieval_top_k: int = 100,
        rerank_top_k: int = 150,
    ):
        self.retrieval_mode   = retrieval_mode
        self.retrieval_top_k  = retrieval_top_k
        self.rerank_top_k     = rerank_top_k
        self.reranker_model   = reranker_model

        self._retriever = None
        self._reranker  = None

    def load(self):
        """Load tất cả components."""
        from retreiver import HybridRetriever
        from reranker  import NeuralReranker

        print("=" * 60)
        print("Loading Legal RAG Pipeline ...")
        print("=" * 60)

        self._retriever = HybridRetriever().load()
        self._reranker  = NeuralReranker(model_name=self.reranker_model).load()

        print("=" * 60)
        print("✅ Pipeline ready!\n")
        return self

    def query(
        self,
        question: str,
        top_k: int | None = None,
        return_raw: bool = False,
    ) -> list[dict]:
        """
        Truy vấn pháp luật với câu hỏi tự nhiên.

        Args:
            question  : Câu hỏi pháp luật (tiếng Việt)
            top_k     : Số điều luật trả về (override rerank_top_k)
            return_raw: Nếu True, trả về toàn bộ dict fields

        Returns:
            list[dict] với các fields:
              rank, score, law_code, article_title, content_text, ...
        """
        k = top_k or self.rerank_top_k
        t0 = time.time()

        # Stage 1: Retrieval
        candidates = self._retriever.retrieve(
            question,
            top_k=self.retrieval_top_k,
            mode=self.retrieval_mode,
        )
        t1 = time.time()

        # Stage 2: Reranking
        reranked = self._reranker.rerank(question, candidates, top_k=k)
        t2 = time.time()

        print(f"[Timing] Retrieval: {(t1-t0)*1000:.0f}ms | "
              f"Reranking: {(t2-t1)*1000:.0f}ms | "
              f"Total: {(t2-t0)*1000:.0f}ms")

        if return_raw:
            return reranked

        # Format output
        results = [
            format_result(c, c.get("_rerank_score", 0.0), i + 1)
            for i, c in enumerate(reranked)
        ]
        return results

    def query_batch(self, questions: list[str], top_k: int = 5) -> list[list[dict]]:
        """Batch query."""
        all_results = []
        for q in questions:
            results = self.query(q, top_k=top_k)
            all_results.append(results)
        return all_results


# ── Output formatters ────────────────────────────────────────────────────────

def format_as_law_refs(results: list[dict]) -> list[str]:
    """
    Format kết quả theo dạng tham chiếu luật:
    "04/2017/QH14|Luật Hỗ trợ DNNVV|Điều 4"
    """
    refs = []
    for r in results:
        law_code    = r.get("law_code") or "?"
        subject     = r.get("subject_title") or r.get("topic_title") or "?"
        article     = r.get("article_title") or "?"
        refs.append(f"{law_code}|{subject}|{article}")
    return refs


def save_results(results: list[dict], query: str, path: Path):
    """Lưu kết quả JSON."""
    output = {
        "query": query,
        "num_results": len(results),
        "law_references": format_as_law_refs(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved → {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Legal RAG Vietnam — Provision-Level Neural Reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/04_pipeline.py --query_file data/query.json
  python src/04_pipeline.py --query_file data/query.json --top_k 5 --reranker bge
  python src/04_pipeline.py --query "DNNVV cần điều kiện gì để được hỗ trợ?"
  python src/04_pipeline.py --query "..." --reranker minilm --top_k 3
  python src/04_pipeline.py --batch data/queries.txt --top_k 5
        """
    )
    parser.add_argument("--query",      type=str, help="Câu hỏi pháp luật (single query)")
    parser.add_argument("--query_file", type=str, help="File JSON chứa danh sách câu hỏi")
    parser.add_argument("--batch",      type=str, help="File txt, mỗi dòng 1 câu hỏi")
    parser.add_argument("--top_k",      type=int, default=7)
    parser.add_argument("--reranker",   type=str, default="minilmc",
                        choices=["bge", "bge-large", "bge-base", "minilm"],
                        help="Neural reranker model")
    parser.add_argument("--retrieval_mode", default="hybrid",
                        choices=["hybrid", "bm25", "dense"])
    parser.add_argument("--retrieval_top_k", type=int, default=30,
                        help="Số candidates cho Stage 1 (default: 80)")
    parser.add_argument("--output", type=str,
                        help="Path lưu kết quả JSON (tuỳ chọn, dùng cho single query)")
    parser.add_argument("--no_rerank", action="store_true",
                        help="Bỏ qua reranking, chỉ dùng retrieval")
    args = parser.parse_args()

    if not args.query and not args.query_file and not args.batch:
        parser.error("Cần một trong: --query, --query_file, hoặc --batch")

    # Load pipeline
    pipeline = LegalRAGPipeline(
        reranker_model  = args.reranker,
        retrieval_mode  = args.retrieval_mode,
        retrieval_top_k = args.retrieval_top_k,
        rerank_top_k    = args.top_k,
    ).load()

    # ── JSON file query mode ─────────────────────────────────────────────────
    if args.query_file:
        query_file = Path(args.query_file)
        if not query_file.exists():
            print(f"ERROR: {query_file} not found.")
            sys.exit(1)

        with open(query_file, encoding="utf-8") as f:
            query_items = json.load(f)

        print(f"📂 Loaded {len(query_items)} queries from {query_file}\n")

        out_path = Path(args.output) if args.output else RESULTS_DIR / "query_results225.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Đọc file cũ nếu đã tồn tại (để nối tiếp)
        all_results = []
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"📎 Appending to existing file ({len(all_results)} entries): {out_path}\n")

        for item in query_items:
            qid      = item.get("id", "unknown")
            question = item["question"]
            if qid <= 1339:
                continue
            if qid >1768:
                break

            results = pipeline.query(question, top_k=args.top_k)

            print_results(results, query=question)


            # Nối kết quả và lưu ngay sau mỗi câu hỏi
            all_results.append({
                "id": qid,
                "query": question,
                "num_results": len(results),
                "law_references": format_as_law_refs(results),
                "results": results,
            })
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved ({len(all_results)} total) → {out_path}")

    # ── Single query ─────────────────────────────────────────────────────────
    elif args.query:
        results = pipeline.query(args.query, top_k=args.top_k)

        print_results(results, query=args.query)

        print("\n📋 Law References:")
        for ref in format_as_law_refs(results):
            print(f"  → {ref}")

        if args.output:
            save_results(results, args.query, Path(args.output))
        else:
            ts = int(time.time())
            save_results(results, args.query, RESULTS_DIR / f"query_{ts}.json")

    # ── Batch mode (txt) ─────────────────────────────────────────────────────
    elif args.batch:
        batch_file = Path(args.batch)
        if not batch_file.exists():
            print(f"ERROR: {batch_file} not found.")
            sys.exit(1)

        questions = [
            line.strip() for line in batch_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"Batch: {len(questions)} queries")

        all_results = []
        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {q}")
            res = pipeline.query(q, top_k=args.top_k)
            print_results(res)
            all_results.append({"query": q, "results": res,
                                "law_refs": format_as_law_refs(res)})

        out_path = RESULTS_DIR / "batch_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Batch results saved → {out_path}")


if __name__ == "__main__":
    main()