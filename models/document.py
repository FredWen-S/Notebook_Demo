
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class ParsedDocument(BaseModel):
    """解析后的文档"""
    source_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    filename: str
    file_type: str          # "pdf" / "md" / "txt"
    content: str            # 提取的纯文本
    page_count: int = 0     # PDF 的页数
    char_count: int = 0     # 字符数
    created_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context):
        """自动计算字符数"""
        if self.char_count == 0:
            self.char_count = len(self.content)


class Chunk(BaseModel):
    """切分后的文本块"""
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str          # 所属文档的 ID
    content: str            # chunk 的文本内容
    index: int              # 在文档中的序号（第几个 chunk）
    metadata: dict = Field(default_factory=dict)
    # metadata 可能包含：page_number, section_title 等


class RetrievedChunk(BaseModel):
    """检索到的 chunk（附带分数）"""
    chunk: Chunk
    score: float            # 检索分数
    source: str             # 来自哪个检索系统："vector" / "bm25" / "rrf"

