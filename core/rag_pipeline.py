# core/rag_pipeline.py
"""RAG Pipeline：完整的检索增强生成流水线"""

import re
from typing import Optional, Generator
from dataclasses import dataclass, field

from models.document import Chunk, RetrievedChunk
from core.hybrid_retriever import HybridRetriever
from core.reranker import CrossEncoderReranker
from core.llm_client import LLMClient


@dataclass
class Citation:
    """引用信息"""
    index: int  # 引用编号 [1], [2], ...
    chunk_id: str
    source_id: str
    filename: str
    content_preview: str  # chunk 内容的前 100 字


@dataclass
class RAGResponse:
    """RAG 回答结果"""
    answer: str  # LLM 的回答（包含 [1] [2] 等标记）
    citations: list[Citation]  # 引用列表
    retrieved_chunks: list[RetrievedChunk]  # 检索到的 chunk


# ============================================================
# Prompt 模板
# ============================================================
SYSTEM_PROMPT = """你是一个专业的知识助手。你的任务是根据提供的参考资料来回答用户的问题。

规则：
1. 只使用提供的参考资料来回答，不要编造信息。
2. 如果参考资料中没有相关信息，请明确说"根据提供的资料，我无法找到相关信息"。
3. 在回答中使用引用标记 [1] [2] 等来标注信息来源。
4. 引用标记对应参考资料的编号。
5. 回答要准确、有条理。
6. 如果有多个资料涉及同一话题，请综合它们的内容。"""

CONTEXT_TEMPLATE = """参考资料：

{context}

---

用户问题：{question}

请基于以上参考资料回答问题，并使用 [编号] 标注引用来源。"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """把检索到的 chunk 组装成上下文文本"""
    parts = []
    for i, rc in enumerate(chunks, start=1):
        filename = rc.chunk.metadata.get("filename", "未知文件")
        parts.append(f"[{i}] （来源：{filename}）\n{rc.chunk.content}")
    return "\n\n".join(parts)


def _parse_citations(
        answer: str,
        chunks: list[RetrievedChunk],
) -> list[Citation]:
    """从回答中解析引用标记"""
    # 找到所有 [数字] 格式的引用
    cited_indices = set()
    for match in re.finditer(r'\[(\d+)\]', answer):
        idx = int(match.group(1))
        cited_indices.add(idx)

    citations = []
    for idx in sorted(cited_indices):
        if 1 <= idx <= len(chunks):
            rc = chunks[idx - 1]
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=rc.chunk.chunk_id,
                    source_id=rc.chunk.source_id,
                    filename=rc.chunk.metadata.get("filename", "未知"),
                    content_preview=rc.chunk.content[:100],
                )
            )

    return citations


# ============================================================
# RAG Pipeline
# ============================================================
class RAGPipeline:
    """
    完整的 RAG 流水线。

    流程：检索 → (可选)重排序 → 组装 Prompt → 调用 LLM → 解析引用
    """

    def __init__(
            self,
            retriever: HybridRetriever,
            llm_client: LLMClient,
            reranker: Optional[CrossEncoderReranker] = None,
            top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.reranker = reranker
        self.top_k = top_k

    def query(
            self,
            question: str,
            temperature: float = 0.3,
    ) -> RAGResponse:
        """
        执行完整的 RAG 查询。

        Args:
            question: 用户的问题
            temperature: LLM 温度参数

        Returns:
            RAGResponse: 包含回答、引用、检索结果
        """
        # 1. 混合检索
        candidates = self.retriever.search(
            question,
            top_k=self.top_k * 4 if self.reranker else self.top_k,
        )

        # 2. 可选：重排序
        if self.reranker and candidates:
            candidates = self.reranker.rerank(
                question, candidates, top_k=self.top_k
            )
        else:
            candidates = candidates[:self.top_k]

        # 3. 组装 Prompt
        context = _build_context(candidates)
        user_message = CONTEXT_TEMPLATE.format(
            context=context,
            question=question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # 4. 调用 LLM
        answer = self.llm_client.chat(
            messages,
            temperature=temperature,
        )

        # 5. 解析引用
        citations = _parse_citations(answer, candidates)

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=candidates,
        )

    def query_stream(
            self,
            question: str,
            temperature: float = 0.3,
    ) -> Generator[str | RAGResponse, None, None]:
        """
        流式 RAG 查询。

        先 yield 文本片段（str），最后 yield 完整的 RAGResponse。
        """
        # 1-2. 检索 + 重排序
        candidates = self.retriever.search(
            question,
            top_k=self.top_k * 4 if self.reranker else self.top_k,
        )

        if self.reranker and candidates:
            candidates = self.reranker.rerank(
                question, candidates, top_k=self.top_k
            )
        else:
            candidates = candidates[:self.top_k]

        # 3. 组装 Prompt
        context = _build_context(candidates)
        user_message = CONTEXT_TEMPLATE.format(
            context=context,
            question=question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # 4. 流式调用 LLM
        full_answer = ""
        for chunk in self.llm_client.chat_stream(messages, temperature=temperature):
            full_answer += chunk
            yield chunk  # 流式输出文本片段

        # 5. 解析引用
        citations = _parse_citations(full_answer, candidates)

        # 最后 yield 完整结果
        yield RAGResponse(
            answer=full_answer,
            citations=citations,
            retrieved_chunks=candidates,
        )