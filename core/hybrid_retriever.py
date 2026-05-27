# core/hybrid_retriever.py
"""混合检索模块：向量检索 + BM25 + RRF 融合"""

from typing import Optional
from collections import defaultdict

from models.document import Chunk, RetrievedChunk
from core.vector_store import VectorStore
from core.bm25_store import BM25Store


def reciprocal_rank_fusion(
        ranked_lists: list[list[RetrievedChunk]],
        k: int = 60,
) -> list[RetrievedChunk]:
    """
    RRF（Reciprocal Rank Fusion）融合多个排名列表。

    Args:
        ranked_lists: 多个排名列表，每个列表是 RetrievedChunk 列表
        k: RRF 常数（默认 60）

    Returns:
        list[RetrievedChunk]: 融合后的排名列表（按 RRF 分数降序）
    """
    # 用 chunk_id 作为文档标识，累加 RRF 分数
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, Chunk] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            chunk_map[cid] = result.chunk  # 保存 chunk 对象

    # 按 RRF 分数降序排列
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 构造结果
    results = []
    for cid, score in sorted_items:
        results.append(
            RetrievedChunk(
                chunk=chunk_map[cid],
                score=score,
                source="rrf",
            )
        )

    return results


class HybridRetriever:
    """
    混合检索器。

    同时使用向量检索和 BM25，用 RRF 融合结果。
    """

    def __init__(
            self,
            vector_store: VectorStore,
            bm25_store: BM25Store,
            vector_weight: float = 1.0,
            bm25_weight: float = 1.0,
            rrf_k: int = 60,
    ):
        """
        Args:
            vector_store: 向量数据库
            bm25_store: BM25 索引
            vector_weight: 向量检索权重（暂时保留，用于未来加权）
            bm25_weight: BM25 权重
            rrf_k: RRF 常数
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def search(
            self,
            query: str,
            top_k: int = 5,
            vector_top_k: int = 20,
            bm25_top_k: int = 20,
            model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> list[RetrievedChunk]:
        """
        混合检索。

        流程：
        1. 向量检索获取 top-vector_top_k
        2. BM25 检索获取 top-bm25_top_k
        3. RRF 融合
        4. 返回 top-top_k

        Args:
            query: 查询文本
            top_k: 最终返回的结果数
            vector_top_k: 向量检索的候选数量
            bm25_top_k: BM25 检索的候选数量
            model_name: Embedding 模型名称

        Returns:
            list[RetrievedChunk]: 融合后的检索结果
        """
        # 1. 向量检索
        vector_results = self.vector_store.search(
            query, top_k=vector_top_k, model_name=model_name
        )

        # 2. BM25 检索
        bm25_results = self.bm25_store.search(query, top_k=bm25_top_k)

        # 3. RRF 融合
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.rrf_k,
        )

        # 4. 取 top-K
        return fused[:top_k]

    def search_vector_only(
            self,
            query: str,
            top_k: int = 5,
            model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> list[RetrievedChunk]:
        """仅向量检索（对比用）"""
        return self.vector_store.search(query, top_k=top_k, model_name=model_name)

    def search_bm25_only(
            self,
            query: str,
            top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """仅 BM25 检索（对比用）"""
        return self.bm25_store.search(query, top_k=top_k)