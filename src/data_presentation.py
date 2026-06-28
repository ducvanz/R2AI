from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
import pandas as pd
import numpy as np

# -----------------------
# Single Entities
# -----------------------

@dataclass
class RetrievalQuery:
    content: str        # Câu hỏi

class NodeType(IntEnum):
    PSEUDO = 0
    ARTICLE = 1       # Điều
    CLAUSE = 2        # Khoản
    POINT = 3         # Điểm

@dataclass 
class Node:
    # text: str = field(default_factory=str)
    level: NodeType
    flat_index: int
    child: list[Node] = field(default_factory=list)

# -----------------------
# Knowledge Base + Scoring Organization
# -----------------------

class Corpus() :
    def __init__(self, prefix:pd.Series = None, data:pd.Series = None) :
        self.articles: list[Node] = None
        self.flat_texts: list[str] = None
        self.flat_scores: pd.Series = None

        if data is not None:
            self.fit(data, prefix)

    def fit(self, data:pd.Series, prefix:pd.Series = None):
        self.flat_texts = []
        self.articles = []

        from .preprocess_parquet import restore_article
        for i, obj in enumerate(data):
            text = restore_article(obj)
            if prefix is not None :
                text = " ".join(prefix.iloc[i], text)

            self.articles.append(Node(
                level = NodeType.ARTICLE,
                flat_index = len(self.flat_texts),
                child = None
            ))
            self.flat_texts.append(text)

    def get_contents(self) :
        if self.flat_texts is None:
            raise RuntimeError(
                "Bắt buộc phải `fit` trước để được hỗ trợ `get_contents`."
            )

        return self.flat_texts

    def get_structure(self) -> list[Node]:
        if self.articles is None:
            raise RuntimeError(
                "Bắt buộc phải `fit` trước để được hỗ trợ `get_structures`."
            )
        
        return self.articles

    def scoring(self, scores:pd.Series):
        """
        Gán score để tra cứu cụ thể sau. 
        Lưu ý: scores được đánh index đúng theo thứ tự ban đầu khi fit.
        """
        self.flat_scores = scores
        self.min_penalty = scores.min()

    def get_score(self, article: Node) -> float | None:
        if self.flat_scores is None:
            raise RuntimeError(
                "Bắt buộc phải `scoring` trước để được hỗ trợ `get_score`."
            )
        
        return self.flat_scores.get(article.flat_index)

    def get_scoreboard(self) -> pd.DataFrame:
        return pd.DataFrame({
            'score': [self.get_score(x) for x in self.articles]
            })

class HierarchicalCorpus(Corpus):
    """
    Mapping giữa:

    Tree Space
        (article, clause, point): Path
            ⇄
    Flat Retrieval Space
        flat_index: int

    Chỉ lưu mapping.
    Không lưu text sau khi fit().
    """

    def __init__(self, title_blending:bool=True, alpha:float=0.5, prefix:pd.Series = None, data:pd.Series = None) :
        self.articles: list[Node] = None
        self.flat_texts: list[str] = None
        self.flat_scores: pd.Series = None

        self.title_blending = title_blending
        self.alpha = alpha

        if data is not None:
            self.fit(data, prefix)

    def fit(self, corpus: pd.Series, prefix:pd.Series = None) -> list[str]:
        """
        Build mapping cho toàn bộ corpus.

        Parameters
        ----------
        corpus_df:
            DataSeries chứa toàn bộ điều luật (sau khi parse thành dict).

        Note
        ----------
            Corpus lưu theo iloc (incremental), không phải theo loc (custom index) của pd.Series.

        Returns
        -------
        flat_texts:
            Danh sách text dùng cho BM25/Dense.
        """ 

        # Reset
        self.flat_texts = []
        ROOT = Node(level=NodeType.PSEUDO,
                    flat_index=None)


        def _append_node(text:str, parent:Node, level:NodeType) -> Node:

            if text != "" :
                this = Node(level=level, flat_index=len(self.flat_texts))
                self.flat_texts.append(text)
                # self.leaves.append(this)
            else :
                this = Node(level=level, flat_index=None)

            parent.child.append(this)

            return this

        def _append_tree(obj:dict, parent:Node, level:NodeType, prefix:str = ""):
            has_children = (
                "content" in obj
                and isinstance(obj["content"], list)
            )

            if not has_children:
                text = "\n".join(filter( None, [prefix, obj['text']] ))
                _append_node(text, parent, level)

            else :
                text = "\n".join(filter( None, [prefix, obj['title']] )) 

                if self.title_blending :
                    this = _append_node("", parent, level)
                    children = obj.get('content', [])
                    for obj in children :
                        _append_tree(obj, this, level+1, text)
                else :
                    this = _append_node(text, parent, level)
                    children = obj.get('content', [])
                    for obj in children :
                        _append_tree(obj, this, level+1, "")
                
        for i, obj in enumerate(corpus):
            if not isinstance(obj, dict):
                print(i)
                print(type(obj))
                print(obj)
                raise RuntimeError()
            else :
                _append_tree(obj, ROOT, NodeType.ARTICLE,
                             prefix = prefix.iloc[i] if prefix is not None else "")

        # pd.DataFrame({'text' : self.flat_texts}).to_csv('results/flattens.csv', index=False)
        
        self.articles = ROOT.child
    
    def get_score(self, article:Node):
        if self.flat_scores is None:
            raise RuntimeError(
                "Bắt buộc phải `scoring` trước để được hỗ trợ `get_score`."
            )

        def _nested_score(parent:Node) -> tuple[float, float, list[int]] :
            if not parent.child:
                leaf_score = super(HierarchicalCorpus, self).get_score(parent)
                return leaf_score, leaf_score, []
            else :
                score_board = []
                penalty = 0

                for idx, child in enumerate(parent.child):
                    mean, max, path = _nested_score(child)
                    if mean is not None:
                        score_board.append([idx+1, mean, max, path])
                    else:
                        penalty += 1

                if not score_board:
                    return None, None, []

                child_score = [item[1] for item in score_board]
                if penalty > 0 :
                    child_score = [self.min_penalty / penalty] * penalty + child_score
                mean = np.mean(child_score)
                
                max_item = score_board[np.argmax([item[2] for item in score_board])]
                max = max_item[2]
                path = [max_item[0]] + max_item[3]

                return mean, max, path

        mean, max, path = _nested_score(article)
        
        if not mean:
            return None, []
        else :
            return mean * self.alpha + max * (1-self.alpha), path
        
    def get_scoreboard(self) -> pd.DataFrame:
        results = [self.get_score(x) for x in self.articles]

        return pd.DataFrame({
            'score':   [x[0] for x in results],
            'comment': [x[1] for x in results]
            })