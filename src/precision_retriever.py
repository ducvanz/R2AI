import pandas as pd
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from .retrieval_pipeline import PrecisionRetrieval
from .data_presentation import RetrievalQuery
from .preprocess_parquet import restore_article

class CrossEncoderReranker(PrecisionRetrieval):
    def __init__(self, model: CrossEncoder, batch_size:int = 32, max_length = 8192, name: str = "CrossEncoder"):
        """
        Khởi tạo Precision Retrieval sử dụng Cross-Encoder.

        Args:
            model (CrossEncoder): Instance của CrossEncoder (VD: BAAI/bge-reranker-v2-m3).
            top_k (int): Số lượng kết quả cuối cùng mong muốn.
            name (str): Tên định danh.
        """
        self.model = model
        self.model.max_length = max_length
        self.batch_size = batch_size
        self.name = name

    def forward(self, query: RetrievalQuery, document: pd.Series) -> pd.DataFrame:
        # 1. Chuẩn bị cặp (Query, Passage)
        # Giả sử document có cột 'content_text' chứa nội dung cần rerank
        pairs = [[query.content, restore_article(doc_text)] for doc_text in document]
        
        # 2. Dự đoán điểm số (Cross-Encoder không cần vector, nó trả về logits)
        with torch.inference_mode():
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # 3. Ghi điểm vào DataFrame
        return pd.DataFrame({
            f'{self.name}_score': scores
            }, index=document.index)
        