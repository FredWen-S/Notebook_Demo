# core/bm25_store.py
"""BM25 关键词检索模块"""

import re
import json
import pickle
from pathlib import Path
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

from models.document import Chunk, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    """
    对文本进行分词。

    处理策略：
    - 中文：用 jieba 分词
    - 英文：按空格分割，转小写
    - 过滤掉单字符和停用词
    """
    # jieba 分词（同时处理中英文）
    words = jieba.lcut(text)

    # 简单的停用词表（实际项目可以加载更完整的）
    stop_words = {
        "的", "了", "是", "在", "和", "有", "就", "不", "也", "都",
        "这", "那", "他", "她", "它", "我", "你", "们", "把", "被",
        "与", "为", "等", "及", "或", "但", "而", "从", "到", "对",
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "this", "that", "it", "as",
    }

    # 过滤：去掉停用词、单字符、纯标点
    tokens = []
    for w in words:
        w = w.strip().lower()
        if len(w) < 2:
            continue
        if w in stop_words:
            continue
        if re.match(r'^[\W]+$', w):  # 纯标点符号
            continue
        tokens.append(w)

    return tokens


class BM25Store:
    """
    BM25 关键词检索。

    功能：
    - 添加 chunk，建立 BM25 索引
    - 按关键词搜索
    - 保存 / 加载索引到磁盘
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        添加一批 chunk 到 BM25 索引。

        Args:
            chunks: Chunk 对象列表
        """
        if not chunks:
            return

        self.chunks.extend(chunks)

        # 对所有 chunk 文本分词
        new_tokenized = [_tokenize(c.content) for c in chunks]
        self.tokenized_corpus.extend(new_tokenized)

        # 重建 BM25 索引（rank-bm25 不支持增量添加，需要全量重建）
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 索引已更新，共 {len(self.chunks)} 个文档")

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """
        BM25 关键词搜索。

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            list[RetrievedChunk]: 检索结果（按 BM25 分数降序）
        """
        if self.bm25 is None or len(self.chunks) == 0:
            return []

        # 对查询分词
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # BM25 打分
        scores = self.bm25.get_scores(query_tokens)

        # 取 top-K
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:  # 忽略零分
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source="bm25",
                )
            )

        return results

    def save(self, dir_path: str | Path) -> None:
        """保存到磁盘"""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 保存 chunks
        chunks_data = [c.model_dump() for c in self.chunks]
        (dir_path / "bm25_chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 保存分词结果
        with open(dir_path / "tokenized_corpus.pkl", "wb") as f:
            pickle.dump(self.tokenized_corpus, f)

        print(f"BM25 索引已保存到: {dir_path}")

    def load(self, dir_path: str | Path) -> None:
        """从磁盘加载"""
        dir_path = Path(dir_path)

        # 加载 chunks
        chunks_data = json.loads(
            (dir_path / "bm25_chunks.json").read_text(encoding="utf-8")
        )
        self.chunks = [Chunk(**d) for d in chunks_data]

        # 加载分词结果
        with open(dir_path / "tokenized_corpus.pkl", "rb") as f:
            self.tokenized_corpus = pickle.load(f)

        # 重建 BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 索引已加载，共 {len(self.chunks)} 个文档")

    @property
    def size(self) -> int:
        return len(self.chunks)