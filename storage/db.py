# storage/db.py
"""简单的 JSON 文件存储（基础版，进阶可替换为 SQLite）"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from models.document import ParsedDocument, Chunk


class NotebookStore:
    """
    Notebook 存储管理。

    每个 Notebook 是一个文件夹，里面包含：
    - metadata.json：Notebook 元信息
    - sources/：上传的原始文件
    - index/：向量索引和 BM25 索引
    """

    def __init__(self, base_dir: str | Path = "data/notebooks"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_notebook(self, name: str) -> dict:
        """创建一个新 Notebook"""
        notebook_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        nb_dir = self.base_dir / notebook_id
        nb_dir.mkdir(parents=True, exist_ok=True)
        (nb_dir / "sources").mkdir(exist_ok=True)
        (nb_dir / "index").mkdir(exist_ok=True)

        metadata = {
            "id": notebook_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "sources": [],
            "chunk_count": 0,
        }

        (nb_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return metadata

    def list_notebooks(self) -> list[dict]:
        """列出所有 Notebook"""
        notebooks = []
        for nb_dir in sorted(self.base_dir.iterdir()):
            meta_path = nb_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                notebooks.append(meta)
        return notebooks

    def get_notebook(self, notebook_id: str) -> Optional[dict]:
        """获取 Notebook 元数据"""
        meta_path = self.base_dir / notebook_id / "metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return None

    def update_notebook(self, notebook_id: str, metadata: dict) -> None:
        """更新 Notebook 元数据"""
        meta_path = self.base_dir / notebook_id / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_notebook_dir(self, notebook_id: str) -> Path:
        """获取 Notebook 目录路径"""
        return self.base_dir / notebook_id

    def get_index_dir(self, notebook_id: str) -> Path:
        """获取索引目录路径"""
        return self.base_dir / notebook_id / "index"

    def get_sources_dir(self, notebook_id: str) -> Path:
        """获取上传文件目录路径"""
        return self.base_dir / notebook_id / "sources"