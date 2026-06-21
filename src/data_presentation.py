from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
import pandas as pd


class NodeType(IntEnum):
    ROOT = 1          # Dataset
    ARTICLE = 2       # Điều
    CLAUSE = 3        # Khoản
    POINT = 4         # Điểm

@dataclass 
class Node:
    # text: str = field(default_factory=None)
    level: NodeType
    flat_index: int
    child: list[Node] = field(default_factory=list)


@dataclass
class HierarchicalCorpus:
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

    tree_root: Node = field(
        default_factory=lambda:
            Node(level=NodeType.ROOT, flat_index=None)
    )

    flatten_corpus: list[Node] = field(default_factory=list)

    flatten_scores: list[float] = field(default_factory=list)

    def fit(self, corpus: pd.Series, title_include:bool = True) -> list[str]:
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
        self.tree_root = Node(level=NodeType.ROOT, flat_index=None)
        self.flatten_corpus = []
        flat_texts = []

        def _append_node(text:str, parent:Node, level:NodeType) -> Node:

            if text != "" :
                this = Node(level=level, flat_index=len(flat_texts))
                flat_texts.append(text)
                self.flatten_corpus.append(this)
            else :
                this = Node(level=level, flat_index=None)

            parent.child.append(this)

            return this

        def _append_tree(obj:dict, parent:Node, level:NodeType):
            has_children = (
                "content" in obj
                and isinstance(obj["content"], list)
            )

            if not has_children:
                text = obj.get('text', "") 
                _append_node(text, parent, level)
            else :
                text = obj.get('title') if title_include else ""
                this = _append_node(text, parent, level)

                children = obj.get('content', [])
                for child in children :
                    _append_tree(child, this, level=level+1)
                
        for i, obj in enumerate(corpus):
            _append_tree(obj, self.tree_root, NodeType.ARTICLE)

        return flat_texts

    def get_root(self) -> Node:
        return self.tree_root
    
    def scoring(self, scores:list[float]):
        if len(scores) != len(self.flatten_corpus) :
            raise RuntimeError(
                "Score được nạp vào phải có kích thước tương đương Corpus được fit ban đầu." \
                "Hoặc bạn chưa fit corpus."
            )
        
        self.flatten_scores = scores

    def get_score(self, n : Node) -> float:
        if self.flatten_scores is None:
            raise RuntimeError(
                "Bắt buộc phải `scoring` trước để được hỗ trợ `get_score`."
            )
        i = n.flat_index
        if i is None:
            return None
        
        return self.flatten_scores[i]
    
    def max_leaf_score(self, parent:Node):
        if not parent.child :
            return self.get_score(parent), []
            
        score = 0
        path = []

        for idx, child in enumerate(parent.child) :
            s, p = self.max_leaf_score(child)
            if (s is not None) and (s > score):
                score = s
                # print(f"Path: {idx}|{p}")
                path = [idx+1]
                path.extend(p)

        return score, path