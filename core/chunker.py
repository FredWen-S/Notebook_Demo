# core/chunker.py
"""文本切分模块：递归字符切分器"""

import re
from typing import Optional
from models.document import ParsedDocument, Chunk


class RecursiveChunker:
    """
    递归字符切分器。

    按照优先级递归地切分文本：
    1. 双换行（段落）
    2. 单换行
    3. 句号/问号/叹号
    4. 逗号/分号
    5. 空格
    6. 任意位置
    """

    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            separators: Optional[list[str]] = None,
    ):
        """
        Args:
            chunk_size: 每个 chunk 的目标大小（字符数）
            chunk_overlap: 相邻 chunk 的重叠大小
            separators: 切分符列表，按优先级排列
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # 段落边界（最优先）
            "\n",  # 换行
            "。",  # 中文句号
            "！",  # 中文叹号
            "？",  # 中文问号
            ". ",  # 英文句号
            "! ",  # 英文叹号
            "? ",  # 英文问号
            "；",  # 中文分号
            "，",  # 中文逗号
            ", ",  # 英文逗号
            " ",  # 空格
            "",  # 任意位置（最后兜底）
        ]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """
        递归切分文本。

        核心逻辑：
        1. 找到当前可用的最高优先级分隔符
        2. 用它切分文本
        3. 合并小片段，使每个片段不超过 chunk_size
        4. 如果某个片段还是太大，用下一级分隔符继续切
        """
        # 最终结果
        final_chunks: list[str] = []

        # 找到当前文本中存在的第一个分隔符
        separator = separators[-1]  # 默认用最后一个（空字符串）
        remaining_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_separators = []
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break

        # 用分隔符切分
        if separator:
            splits = text.split(separator)
        else:
            # 空字符串分隔 = 按单个字符切
            splits = list(text)

        # 合并小片段
        current_chunk = ""
        for split in splits:
            piece = split if not separator else split
            piece_with_sep = piece + separator if separator else piece

            if len(current_chunk) + len(piece_with_sep) <= self.chunk_size:
                current_chunk += piece_with_sep
            else:
                # 当前 chunk 满了
                if current_chunk:
                    final_chunks.append(current_chunk.rstrip(separator))

                # 如果这个片段本身就超过 chunk_size，递归切分
                if len(piece_with_sep) > self.chunk_size and remaining_separators:
                    sub_chunks = self._split_text(piece, remaining_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece_with_sep

        # 别忘了最后一个 chunk
        if current_chunk:
            final_chunks.append(current_chunk.rstrip(separator))

        return final_chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """给相邻 chunk 添加重叠"""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            # 从上一个 chunk 的末尾取 overlap 长度的文本
            overlap_text = prev[-self.chunk_overlap:]
            # 如果 overlap 不是从句子开头开始，尝试找到最近的句子开头
            # 简单处理：找到 overlap 中第一个换行或句号后的位置
            for sep in ["\n", "。", ". "]:
                idx = overlap_text.find(sep)
                if idx != -1:
                    overlap_text = overlap_text[idx + len(sep):]
                    break

            result.append(overlap_text + curr)

        return result

    def chunk_text(self, text: str) -> list[str]:
        """
        切分文本为 chunk 列表。

        Args:
            text: 要切分的原始文本

        Returns:
            list[str]: chunk 文本列表
        """
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = self._split_text(text, self.separators)

        # 过滤空 chunk
        chunks = [c.strip() for c in chunks if c.strip()]

        # 添加重叠
        chunks = self._add_overlap(chunks)

        return chunks

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        切分一个已解析的文档，返回 Chunk 对象列表。

        Args:
            doc: 解析后的文档

        Returns:
            list[Chunk]: Chunk 对象列表
        """
        texts = self.chunk_text(doc.content)

        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                source_id=doc.source_id,
                content=text,
                index=i,
                metadata={
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                },
            )
            chunks.append(chunk)

        return chunks