# models/notebook.py
"""Notebook 数据模型"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class NotebookMeta(BaseModel):
    """Notebook 元数据"""
    id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    sources: list[dict] = Field(default_factory=list)
    chunk_count: int = 0


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str  # "user" / "assistant"
    content: str
    citations: list[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)