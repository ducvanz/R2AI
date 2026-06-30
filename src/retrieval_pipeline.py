from abc import ABC, abstractmethod
import pandas as pd
import gc
import torch
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

    @abstractmethod
    def save(self, folder:str):
        pass

    @abstractmethod
    def load(self, folder:str):
        pass


class PrecisionRetrieval(ABC) :
    @abstractmethod
    def forward(self, query: RetrievalQuery, document: pd.DataFrame) -> pd.DataFrame:
        pass


class RetrievalPipeline:

    def __init__(self,
                 data: pd.Series,
                 data_prefix: pd.Series = None,
                 HyDE = None,
                 recallLayers: list[RecallRetrieval] = None,
                 fusion_target: dict[str, float] = None, fusion_norm: int = 60,
                 precisionLayers: list[PrecisionRetrieval] = None,
                 top_re_rank: int = 100,
                 save_folder:str = None
                 ):
        
        self.data: pd.Series = data
        self.data_prefix: pd.Series = data_prefix

        self.hyde = HyDE

        self.recallLayers: list[RecallRetrieval] = recallLayers

        self.fusion_target:dict[str, float] = fusion_target 
        self.k = fusion_norm
        if fusion_target is None :
            self.fusion_target = {f'{x.name}_score' : 1.0 for x in self.recallLayers}

        self.precisionLayers: list[PrecisionRetrieval] = precisionLayers
        self.top_rank = top_re_rank
        
        self._load(save_folder)

    def _load(self, save_folder:str = None) :
        print(f"[INFO] Đang build Corpuses")
        corpus_set:set[Corpus] = set([x.corpus for x in self.recallLayers])
        for idx, corpus in enumerate(corpus_set) :
            print(f"[INFO] [{idx+1}/{len(corpus_set)}] Đang build {type(corpus)}")
            corpus.fit(self.data, self.data_prefix)

        if save_folder is None :
            print(f"[INFO] Đang fit lớp Recall Retrievers")
            for idx, layer in enumerate(self.recallLayers) :

                print(f"[INFO] [{idx+1}/{len(self.recallLayers)}] Đang fit {type(layer)} với Knowledge Base {type(layer.corpus)}")
                layer.fit()

            gc.collect()
            torch.cuda.empty_cache()
        else :
            print(f"[INFO] Đang load saved models cho lớp Recall Retrievers")
            for idx, layer in enumerate(self.recallLayers) :
                print(f"[INFO] [{idx+1}/{len(self.recallLayers)}] Đang load {type(layer)}")
                layer.load(save_folder + f"/{layer.name}")

    def save(self, save_folder:str) :
        for layer in self.recallLayers :
            layer.save(save_folder + f"/{layer.name}")


    def retrieve(self, query: str) -> pd.DataFrame:

        if self.hyde is not None:
            query = self.hyde.enhance(query)

        query = RetrievalQuery(
            content = query
        )

        scores:pd.DataFrame = pd.DataFrame()

        if self.recallLayers is not None:
            for layer in self.recallLayers:

                sc = layer.forward(query)
                scores = scores.join(sc, how='outer')

            # Dọn rác VRAM các vector embedding trung gian sau Recall
            gc.collect()
            torch.cuda.empty_cache()

        self.fusion(scores)

        # iLoc recovering
        scores.index = self.data.index[scores.index]

        top_score_indices = scores.sort_values("RRF_score", ascending=False).head(self.top_rank).index

        candidiates = self.data.loc[top_score_indices]

        if self.precisionLayers is not None:
            for layer in self.precisionLayers:
                
                sc = layer.forward(query, candidiates)
                scores = scores.join(sc, how='outer')

            # Dọn rác VRAM sau khi inference xong toàn bộ
            gc.collect()
            torch.cuda.empty_cache()

        return scores
    
    def fusion(self, score_board:pd.DataFrame) -> pd.DataFrame:
        
        # Khởi tạo cột điểm Fusion mặc định là 0
        fusion_col = "RRF_score"
        score_board[fusion_col] = 0.0
        
        # Lặp qua các cột điểm đầu vào đã chỉ định
        for col, factor in self.fusion_target.items():
            if col not in score_board.columns:
                print(f"[WARNING] Cột '{col}' không tồn tại trong DataFrame. Bỏ qua.")
                continue
            
            # 1. Tính toán thứ hạng (Rank)
            # - ascending=False: Điểm số cao nhất (top 1) sẽ nhận rank 1.
            # - method='min': Nếu 2 document bằng điểm nhau, cả 2 nhận chung rank cao nhất.
            # - na_option='bottom': Đẩy các row bị NaN (do không được recall) xuống hạng bét.
            ranks = score_board[col].rank(ascending=False, method='min', na_option='bottom')
            
            # 2. Tính điểm RRF và cộng dồn
            # Chỉ cộng điểm cho những document thực sự có điểm ở retriever hiện tại (không bị NaN)
            valid_mask = score_board[col].notna()
            score_board.loc[valid_mask, fusion_col] += factor / (self.k + ranks[valid_mask])
        
        return score_board