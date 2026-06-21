from .retrieval_pipeline import RecallRetrieval, RetrievalQuery
from .data_presentation import HierarchicalCorpus
from .preprocess_parquet import restore_article
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import faiss



BM25_LEN_NORMALIZE = 0.25


# -----------------------
# BM25
# -----------------------

class BM25Raw(RecallRetrieval):
    def __init__(self, text_column:str) :
        self.text_column = text_column

        self.bm25 = None
        self.doc_indices = None

    def fit(self, documents: pd.DataFrame):
        corpus_tokens = (
            documents[self.text_column]
            .fillna("").astype(str)
            .str.lower()
            .str.split()
            .tolist()
        )

        self.bm25 = BM25Okapi(
            corpus_tokens, b=BM25_LEN_NORMALIZE
        )

        self.doc_indices = (
            documents.index.to_numpy()
        )

    def forward(self, query:RetrievalQuery):

        if self.bm25 is None:
            raise RuntimeError(
                "BM25Layer must be fitted first."
            )

        query_tokens = (
            query.content.lower().split()
        )

        scores = self.bm25.get_scores(query_tokens)
        column_name = f"bm25_raw_{self.text_column}"

        result = pd.DataFrame({
            column_name: scores
        }, index=self.doc_indices)

        return result
  

class BM25Hier(RecallRetrieval):

    def __init__(self, title_include:bool=True, text_column:str="content_text"):
        self.map: HierarchicalCorpus = HierarchicalCorpus()
        self.text_column = text_column
        self.title_include = title_include

        self.bm25 = None
        self.doc_indices = None

    def fit(self, documents: pd.DataFrame):
        """
        flat_documents:
            output từ HierarchicalCorpus.fit()

        doc_indices:
            corpus_df.index
        """
        self.doc_indices = documents.index
        corpus = documents[self.text_column]

        flat_documents = self.map.fit(corpus, self.title_include)

        corpus_tokens = [
            str(text).lower().split()
            for text in flat_documents
        ]

        self.bm25 = BM25Okapi(
            corpus_tokens, b=BM25_LEN_NORMALIZE
        )

    def forward(self, query: RetrievalQuery):

        if self.bm25 is None:
            raise RuntimeError(
                "BM25Hier must be fitted first."
            )

        query_tokens = (query.content.lower().split())

        self.map.scoring(
            self.bm25.get_scores(query_tokens)
        )

        results: list = []
        paths: list = []

        for i, article in enumerate(self.map.get_root().child) :
            sc, pt = self.map.max_leaf_score(article)

            results.append(sc)
            paths.append(pt) 

        return pd.DataFrame({"bm25_hier" : results,
                             "bm25_Comment" : paths}, 
                            index=self.doc_indices)
    
# -----------------------
# Dense Retrieval
# -----------------------

class DenseRaw(RecallRetrieval):

    def __init__(self, model,
                top_k:int = None,
                index_type:str="flat",
                text_column: str = "content_text",
                batch_size:int = 32
        ):

        self.text_column = text_column
        self.top_k = top_k

        self.model = model
        self.index_type = index_type
        self.batch_size = batch_size

        self.index = None
        self.doc_indices = None

    def fit(self, documents: pd.DataFrame):
        texts = (
            documents[self.text_column]
            .fillna("")
            .apply(restore_article)
            .tolist()
        )

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]

        match self.index_type :
            case 'hnsw' :
                self.index = faiss.IndexHNSWFlat(dim, 32)       # Approximate nearest neighbor
            case 'flat':
                self.index = faiss.IndexFlatIP(dim)             # Fully search
            case _ :
                raise RuntimeError(
                    "Only support Index type as: flat, hnsw"
                )

        self.index.add(embeddings)

        self.doc_indices = (
            documents.index.to_numpy()
        )

    def forward(self, query: RetrievalQuery):

        if self.index is None:
            raise RuntimeError(
                "DenseRaw must be fitted first."
            )

        query_embedding = self.model.encode(
            [query.content],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_embedding = query_embedding.astype(np.float32)

        scores, indices = self.index.search(
            query_embedding,
            k = self.top_k or len(self.doc_indices)
        )

        scores = scores[0]
        score_indices = indices[0]

        column_name = f"dense_raw_{self.text_column}"

        return pd.DataFrame({
                column_name: scores
            },
            index=self.doc_indices[score_indices]
        )
    
class DenseHier(RecallRetrieval):

    def __init__(self, model,
                top_k:int = None,
                title_include:bool = True,
                index_type:str="flat",
                text_column: str = "content_text",
                batch_size:int = 32
        ):

        self.map: HierarchicalCorpus = HierarchicalCorpus()
        self.text_column = text_column
        self.top_k = top_k
        self.title_include = title_include

        self.model = model
        self.index_type = index_type
        self.batch_size = batch_size

        self.index = None
        self.doc_indices = None

    def fit(self, documents: pd.DataFrame):
        corpus = documents[self.text_column]

        flat_documents = self.map.fit(corpus, self.title_include)

        embeddings = self.model.encode(
            flat_documents,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]

        match self.index_type :
            case 'hnsw' :
                self.index = faiss.IndexHNSWFlat(dim, 32)       # Approximate nearest neighbor
            case 'flat':
                self.index = faiss.IndexFlatIP(dim)             # Fully search
            case _ :
                raise RuntimeError(
                    "Only support Index type as: flat, hnsw"
                )

        self.index.add(embeddings)

        self.doc_indices = (
            documents.index.to_numpy()
        )

    def forward(self, query: RetrievalQuery):

        if self.index is None:
            raise RuntimeError(
                "DenseRaw must be fitted first."
            )

        query_embedding = self.model.encode(
            [query.content],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_embedding = query_embedding.astype(np.float32)

        # Tính điểm giữa Query và Dataset
        scores, indices = self.index.search(
            query_embedding,
            k = self.top_k or len(self.map.flatten_corpus)
        )

        # Map lại score cho corpus (do Dense trả về top_k nên phần nào thiếu thì mặc định bằng 0)
        full_scores = np.zeros(
            len(self.map.flatten_corpus),
            dtype=np.float32
        )

        full_scores[indices] = scores

        self.map.scoring(full_scores)


        # Tính lại điểm đại diện cho các Điểm (Article) dựa trên Leaf tốt nhất.
        results: list = []
        paths: list = []

        for i, article in enumerate(self.map.get_root().child) :
            sc, pt = self.map.max_leaf_score(article)

            results.append(sc)
            paths.append(pt) 

        return pd.DataFrame({"dense_hier" : results,
                             "dense_Comment" : paths}, 
                            index=self.doc_indices)