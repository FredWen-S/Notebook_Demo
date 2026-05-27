# core/reranker.py
"""Reranker 重排序模块（可选）"""

from typing import Optional

from models.document import RetrievedChunk


class CrossEncoderReranker:
    """
    基于 Cross-Encoder 的重排序器。

    Cross-Encoder 会把 (query, document) 对作为输入，
    输出一个相关性分数。比 Embedding + 余弦相似度更准确，
    但也更慢（不能预计算）。
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"正在加载 Reranker 模型: {self.model_name} ...")
            self._model = CrossEncoder(self.model_name)
            print("Reranker 模型加载完成。")

    def rerank(
            self,
            query: str,
            results: list[RetrievedChunk],
            top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        对检索结果重排序。

        Args:
            query: 查询文本
            results: 初步检索结果
            top_k: 重排序后返回前 K 个

        Returns:
            list[RetrievedChunk]: 重排序后的结果
        """
        if not results:
            return []

        self._load_model()

        # 构建 (query, doc) 对
        pairs = [(query, r.chunk.content) for r in results]

        # Cross-Encoder 打分
        scores = self._model.predict(pairs)

        # 按分数排序
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # 更新分数并返回 top-K
        reranked = []
        for score, result in scored_results[:top_k]:
            reranked.append(
                RetrievedChunk(
                    chunk=result.chunk,
                    score=float(score),
                    source="reranker",
                )
            )

        return reranked