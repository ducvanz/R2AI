from .retrieval_pipeline import RecallRetrieval
from .data_presentation import Corpus, RetrievalQuery
import pandas as pd
import numpy as np
import bm25s
import faiss


# -----------------------
# BM25
# -----------------------

class BM25(RecallRetrieval):

    def __init__(self, corpus:Corpus, top_k: int = None, norm: float = 0.75, name="BM25"):
        self.corpus = corpus
        self.name = name

        self.top_k = top_k
        self.norm = norm

        self.bm25 = None

    def fit(self):
        texts = [text.lower() for text in self.corpus.get_contents()]

        tokenized = bm25s.tokenize(texts)

        self.bm25 = bm25s.BM25(b=self.norm)

        self.bm25.index(tokenized)     

    def forward(self, query: RetrievalQuery):

        if self.bm25 is None:
            raise RuntimeError(
                "BM25Raw must be fitted first."
            )

        query_tokens = bm25s.tokenize(
            [query.content.lower()]
        )

        results, scores = self.bm25.retrieve(
            query_tokens,
            k=self.top_k or len(self.corpus.get_contents())
        )

        indices, scores = results[0], scores[0]

        self.corpus.scoring(pd.Series(scores, index=indices))

        df = self.corpus.get_scoreboard()
        return df.add_prefix(f"{self.name}_")


# -----------------------
# Dense Retrieval
# -----------------------

class Dense(RecallRetrieval):

    def __init__(self, 
                 corpus: Corpus,
                 model,
                 top_k: int = None,
                 index_type: str = "flat", # Hỗ trợ "flat" hoặc "hnsw"
                 name: str = "Dense",
                 batch_size: int = 32,     # Kích thước batch khi embedding
                 M: int = 32,              # Số lượng neighbor của mỗi node cho HNSW
                 ef_construction: int = 256,
                 ef_search: int = 64
        ):
        """
        Khởi tạo bộ truy xuất DenseRaw sử dụng kiến trúc Bi-Encoder và FAISS Indexing.
        Hỗ trợ chuẩn hóa L2 và tìm kiếm bằng Inner Product để tính Cosine Similarity.

        Args:
            corpus (Corpus): Đối tượng Corpus chứa tập tri thức (Knowledge Base).
            model: Mô hình Bi-Encoder (thường là instance của SentenceTransformer).

            top_k (int, optional): Số lượng documents gần nhất cần trả về. 
                Nếu để None, sẽ chấm điểm và trả về toàn bộ dữ liệu trong corpus.

            index_type (str, optional): Phương pháp lưu trữ và tìm kiếm của FAISS. Hỗ trợ 2 chế độ:
                - "flat": Quét vét cạn (Brute-force) bằng IndexFlatIP. Trả kết quả chính xác 100% 
                  nhưng chậm nếu corpus quá lớn.
                - "hnsw": Tìm kiếm xấp xỉ dạng đồ thị (Approximate Nearest Neighbor) bằng IndexHNSWFlat. 
                  Tốc độ query siêu nhanh cho dữ liệu lớn nhưng cần tốn RAM và thời gian fit đồ thị.
    
            name (str, optional): Tên định danh của Retriever. Dùng làm prefix cho các cột dataframe
                kết quả trả về. Mặc định là "Dense".

            batch_size (int, optional): Kích thước batch đưa vào model lúc embedding văn bản ở hàm `fit()`. 
                Tăng lên nếu GPU có nhiều VRAM để chạy nhanh hơn. Mặc định là 32.

            M (int, optional): [Chỉ dùng cho HNSW] Số lượng liên kết hàng xóm liền kề đối với mỗi node đồ thị. 
                M lớn hơn tốn RAM hơn nhưng tăng tỷ lệ chính xác. Mặc định là 32.

            ef_construction (int, optional): [Chỉ dùng cho HNSW] Kích thước danh sách ứng viên (candidate list) 
                trong quá trình xây dựng index. Số cao hơn thì đồ thị chất lượng hơn, nhưng tốn thời gian fit(). 
                Lưu ý: ef_construction >> M để quá trình dựng đồ thị ổn định. 
                ef_construction > ef_search để đảm bảo đồ thị đủ tốt trong tìm kiếm.
                Mặc định là 256.

            ef_search (int, optional): [Chỉ dùng cho HNSW] Số lượng node hàng xóm được quét qua khi thực hiện 
                tìm kiếm. Càng lớn thì query càng chính xác (Recall cao) nhưng tốc độ query sẽ giảm nhẹ. 
                Lưu ý: ef_search >= top_k để cho kết quả chính xác hơn. Mặc định là 64.
        """

        self.corpus = corpus
        self.name = name
        self.model = model
        self.top_k = top_k
        self.index_type = index_type.lower()
        self.batch_size = batch_size
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        self.index = None

    def fit(self):
        # Lưu ý: Abstract class RecallRetrieval.fit() không nhận tham số documents.
        # Dữ liệu lấy trực tiếp từ self.corpus
        texts = self.corpus.get_contents()
        
        # 1. Mã hóa toàn bộ text thành embeddings (Sử dụng model của SentenceTransformers hoặc tương tự)
        embeddings = self.model.encode(texts, 
                                       batch_size=self.batch_size, 
                                       show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings) # Normalize để  dot product -> cosine similarity
        
        d = embeddings.shape[1] # Độ phân giải (dimension) của vector
        
        # 2. Khởi tạo Faiss Index theo Mode
        if self.index_type == "flat":
            # Inner Product (Cosine Similarity) - yêu cầu vector đã chuẩn hóa (normalized)
            self.index = faiss.IndexFlatIP(d) 
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, self.M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")
        
        # 3. Add dữ liệu vào Index
        self.index.add(embeddings)

    def forward(self, query: RetrievalQuery):
        if self.index is None:
            raise RuntimeError("DenseRaw must be fitted first.")

        # 1. Mã hóa câu truy vấn
        # FAISS yêu cầu query phải có shape dạng 2D array: (1, vector_dimension)
        query_embedding = self.model.encode([query.content])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        # 2. Tìm kiếm trên FAISS
        k = self.top_k or len(self.corpus.get_contents())
        distances, indices = self.index.search(query_embedding, k)
        
        # Vì chỉ có 1 câu query nên ta lấy kết quả ở index [0]
        dist_scores, idxs = distances[0], indices[0]

        # Lọc bỏ các chỉ mục -1 (Trường hợp FAISS không tìm đủ top_k neighbor)
        valid_mask = idxs != -1
        dist_scores = dist_scores[valid_mask]
        idxs = idxs[valid_mask]

        # 3. Ghi nhận điểm số vào corpus và xuất kết quả
        self.corpus.scoring(pd.Series(dist_scores, index=idxs))
        
        df = self.corpus.get_scoreboard()
        return df.add_prefix(f"{self.name}_")
