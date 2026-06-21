from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd

@dataclass
class RetrievalQuery:
    content: str

class RecallRetrieval(ABC):
    """
    Thực hiện retrieve trên toàn bộ dataset để tạo Recall Candidates Pool.
    Yêu cầu phải fit(corpus) trước khi thực hiện bất kỳ forward nào.
    """

    @abstractmethod
    def fit(self, corpus: pd.DataFrame) -> None:
        """
        Khởi tạo corpus.
        """
        pass

    @abstractmethod
    def forward(self, query: RetrievalQuery) -> pd.DataFrame:
        """
        Input: query.
        Output: score đánh giá trên corpus khởi tạo ban đầu.
        Index của kết quả đánh dựa theo corpus.
        """
        pass

class PrecisionRetrieval(ABC) :
    @abstractmethod
    def forward(self, query: RetrievalQuery, document: pd.DataFrame) -> pd.DataFrame:
        pass

class RetrievalPipeline:

    def __init__(self, corpus: pd.DataFrame, 
                 retrievalLayers: list[RecallRetrieval] = None,
                 processorLayers: list[PrecisionRetrieval] = None):
        self.corpus: pd.DataFrame = corpus
        self.recallLayers: list[RecallRetrieval] = retrievalLayers
        self.precisionLayers: list[PrecisionRetrieval] = processorLayers

        self._load(corpus)

    def _load(self, corpus: pd.DataFrame) :
        print(f"[INFO] Đang fit lớp Recall Retrieval")
        for idx, retriever in enumerate(self.recallLayers) :
            print(f"\t [{idx+1}/{len(self.recallLayers)}] Đang fit {type(retriever)} với dữ liệu {retriever.text_column}")
            retriever.fit(corpus)

    def retrieve(self, query: str) -> pd.DataFrame:

        query = RetrievalQuery(
            content = query
        )

        scores:pd.DataFrame = pd.DataFrame()

        if self.recallLayers is not None:
            for layer in self.recallLayers:
                sc = layer.forward(query)
                scores = scores.join(sc, how='outer')

        return scores