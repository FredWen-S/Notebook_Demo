# config.py
"""全局配置"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """应用配置"""

    # ---- 路径 ----
    data_dir: Path = Path("data")
    notebooks_dir: Path = Path("data/notebooks")
    uploads_dir: Path = Path("data/uploads")
    db_path: Path = Path("data/notebook_demo.db")

    # ---- Embedding ----
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384

    # ---- Chunking ----
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ---- 检索 ----
    vector_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60

    # ---- LLM ----
    llm_provider: str = "ollama"  # "ollama" / "openai" / "claude" / "gemini"
    llm_model: str = "qwen2.5:7b"  # 模型名称
    llm_api_key: str = ""  # API Key（Ollama 不需要）
    llm_base_url: str = ""  # API 地址（Ollama 不需要）
    llm_temperature: float = 0.3  # 低温度 = 更确定的回答
    llm_max_tokens: int = 2048

    # ---- Reranker（可选）----
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    def __post_init__(self):
        """确保目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = AppConfig()