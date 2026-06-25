from abc import ABC, abstractmethod
import pandas as pd
from .data_presentation import Corpus, RetrievalQuery


class RecallRetrieval(ABC):
    """
    Thực hiện retrieve trên toàn bộ dataset để tạo Recall Candidates Pool.
    Yêu cầu phải fit(corpus) trước khi thực hiện bất kỳ forward nào.
    """

    corpus: Corpus
    name: str

    @abstractmethod
    def fit(self) -> None:
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

    # @abstractmethod
    # def save(self, folder:str):
    #     pass

    # @abstractmethod
    # def load(self, folder:str):
    #     pass


class PrecisionRetrieval(ABC) :
    @abstractmethod
    def forward(self, query: RetrievalQuery, document: pd.DataFrame) -> pd.DataFrame:
        pass


class RetrievalPipeline:

    def __init__(self,
                 data: pd.Series,
                 retrievalLayers: list[RecallRetrieval] = None,
                 processorLayers: list[PrecisionRetrieval] = None
                 ):
        
        self.data: pd.Series = data
        self.recallLayers: list[RecallRetrieval] = retrievalLayers
        self.precisionLayers: list[PrecisionRetrieval] = processorLayers
        
        self._load()

    def _load(self) :
        print(f"[INFO] Đang build Corpuses")
        corpus_set:set[Corpus] = set([x.corpus for x in self.recallLayers])
        for idx, corpus in enumerate(corpus_set) :
            print(f"[INFO] [{idx+1}/{len(corpus_set)}] Đang build {type(corpus)}")
            corpus.fit(self.data)

        print(f"[INFO] Đang fit lớp Recall Retrievers")
        for idx, retriever in enumerate(self.recallLayers) :
            print(f"[INFO] [{idx+1}/{len(self.recallLayers)}] Đang fit {type(retriever)} với Knowledge Base {type(retriever.corpus)}")
            retriever.fit()

    def retrieve(self, query: str) -> pd.DataFrame:

        query = RetrievalQuery(
            content = query
        )

        scores:pd.DataFrame = pd.DataFrame()

        if self.recallLayers is not None:
            for layer in self.recallLayers:
                sc = layer.forward(query)
                scores = scores.join(sc, how='outer')


        # iLoc recovering
        scores.index = self.data.index[scores.index]

        return scores