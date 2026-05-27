# core/vector_store.py
"""向量数据库模块：基于 FAISS 的向量存储和检索"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

from models.document import Chunk, RetrievedChunk
from core.embedder import embed_texts, embed_query


class VectorStore:
    """
    基于 FAISS 的向量存储。

    功能：
    - 添加 chunk 向量
    - 按向量相似度搜索
    - 保存 / 加载索引到磁盘
    """

    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: 向量维度（必须和 Embedding 模型匹配）
        """
        self.dimension = dimension
        # 使用内积（Inner Product）索引
        # 因为我们的向量已经归一化，内积 = 余弦相似度
        self.index = faiss.IndexFlatIP(dimension)
        # 存储 chunk 信息，索引位置和 chunk 一一对应
        self.chunks: list[Chunk] = []

    def add_chunks(
            self,
            chunks: list[Chunk],
            model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """
        添加一批 chunk 到向量库。

        Args:
            chunks: Chunk 对象列表
            model_name: Embedding 模型名称
        """
        if not chunks:
            return

        # 提取文本
        texts = [c.content for c in chunks]

        # 生成 Embedding
        embeddings = embed_texts(texts, model_name=model_name)

        # 添加到 FAISS 索引
        self.index.add(embeddings.astype(np.float32))

        # 保存 chunk 信息
        self.chunks.extend(chunks)

        print(f"已添加 {len(chunks)} 个 chunk，总计 {self.index.ntotal} 个向量")

    def search(
            self,
            query: str,
            top_k: int = 10,
            model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> list[RetrievedChunk]:
        """
        向量相似度搜索。

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果
            model_name: Embedding 模型名称

        Returns:
            list[RetrievedChunk]: 检索结果
        """
        if self.index.ntotal == 0:
            return []

        # 查询文本 → 向量
        query_vec = embed_query(query, model_name=model_name)
        query_vec = query_vec.astype(np.float32).reshape(1, -1)

        # FAISS 搜索
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, actual_k)

        # 构造结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS 返回 -1 表示没有更多结果
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source="vector",
                )
            )

        return results

    def save(self, dir_path: str | Path) -> None:
        """
        保存索引到磁盘。

        Args:
            dir_path: 保存目录
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self.index, str(dir_path / "faiss.index"))

        # 保存 chunk 信息（序列化为 JSON）
        chunks_data = [c.model_dump() for c in self.chunks]
        (dir_path / "chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"向量库已保存到: {dir_path}")

    def load(self, dir_path: str | Path) -> None:
        """
        从磁盘加载索引。

        Args:
            dir_path: 保存目录
        """
        dir_path = Path(dir_path)

        index_path = dir_path / "faiss.index"
        chunks_path = dir_path / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"索引文件不完整: {dir_path}")

        # 加载 FAISS 索引
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d

        # 加载 chunk 信息
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**d) for d in chunks_data]

        print(f"向量库已加载，共 {self.index.ntotal} 个向量")

    @property
    def size(self) -> int:
        """当前存储的向量数量"""
        return self.index.ntotal