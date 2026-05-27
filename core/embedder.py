# core/embedder.py
"""Embedding 模块：把文本变成向量"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional

# 模型缓存，避免重复加载
_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> SentenceTransformer:
    """
    获取 Embedding 模型（有缓存）。

    首次调用会下载模型（约 100-500MB），之后会从本地缓存加载。

    Args:
        model_name: 模型名称

    Returns:
        SentenceTransformer 模型实例
    """
    if model_name not in _model_cache:
        print(f"正在加载 Embedding 模型: {model_name} ...")
        _model_cache[model_name] = SentenceTransformer(model_name)
        print(f"模型加载完成。向量维度: {_model_cache[model_name].get_sentence_embedding_dimension()}")
    return _model_cache[model_name]


def embed_texts(
        texts: list[str],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
        show_progress: bool = True,
) -> np.ndarray:
    """
    把一组文本转成向量。

    Args:
        texts: 文本列表
        model_name: Embedding 模型名称
        batch_size: 批处理大小
        show_progress: 是否显示进度条

    Returns:
        np.ndarray: 形状为 (len(texts), dim) 的向量数组
    """
    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 归一化，方便用内积算相似度
    )
    return embeddings


def embed_query(
        query: str,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> np.ndarray:
    """
    把一个查询文本转成向量。

    Args:
        query: 查询文本
        model_name: Embedding 模型名称

    Returns:
        np.ndarray: 一维向量
    """
    model = get_model(model_name)
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding