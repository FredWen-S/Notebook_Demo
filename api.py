# api.py
"""FastAPI 后端 API"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
from pathlib import Path

from config import config
from storage.db import NotebookStore
from core.parser import parse_file
from core.chunker import RecursiveChunker
from core.vector_store import VectorStore
from core.bm25_store import BM25Store
from core.hybrid_retriever import HybridRetriever
from core.llm_client import create_llm_client
from core.rag_pipeline import RAGPipeline
from core.generator import ContentGenerator

app = FastAPI(title="Notebook Demo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局资源
nb_store = NotebookStore(config.notebooks_dir)
# 索引缓存
_index_cache: dict[str, dict] = {}


def get_indexes(notebook_id: str) -> dict:
    """获取或创建索引"""
    if notebook_id not in _index_cache:
        index_dir = nb_store.get_index_dir(notebook_id)
        vs = VectorStore(dimension=config.embedding_dimension)
        bs = BM25Store()

        if (index_dir / "faiss.index").exists():
            vs.load(index_dir)
            bs.load(index_dir)

        _index_cache[notebook_id] = {"vector_store": vs, "bm25_store": bs}

    return _index_cache[notebook_id]


# ---- 请求/响应模型 ----
class CreateNotebookRequest(BaseModel):
    name: str


class QueryRequest(BaseModel):
    question: str
    temperature: float = 0.3


class GenerateRequest(BaseModel):
    output_type: str = "faq"  # faq / study_guide / briefing_doc / audio_script


# ---- API 端点 ----
@app.get("/api/notebooks")
def list_notebooks():
    return nb_store.list_notebooks()


@app.post("/api/notebooks")
def create_notebook(req: CreateNotebookRequest):
    return nb_store.create_notebook(req.name)


@app.get("/api/notebooks/{notebook_id}")
def get_notebook(notebook_id: str):
    meta = nb_store.get_notebook(notebook_id)
    if not meta:
        raise HTTPException(404, "Notebook not found")
    return meta


@app.post("/api/notebooks/{notebook_id}/upload")
async def upload_document(notebook_id: str, file: UploadFile = File(...)):
    """上传并处理文档"""
    meta = nb_store.get_notebook(notebook_id)
    if not meta:
        raise HTTPException(404, "Notebook not found")

    # 保存文件
    source_dir = nb_store.get_sources_dir(notebook_id)
    file_path = source_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 解析和切分
    doc = parse_file(file_path)
    chunker = RecursiveChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = chunker.chunk_document(doc)

    # 建索引
    indexes = get_indexes(notebook_id)
    indexes["vector_store"].add_chunks(chunks, model_name=config.embedding_model)
    indexes["bm25_store"].add_chunks(chunks)

    # 保存
    index_dir = nb_store.get_index_dir(notebook_id)
    indexes["vector_store"].save(index_dir)
    indexes["bm25_store"].save(index_dir)

    # 更新元数据
    meta["sources"].append({
        "source_id": doc.source_id,
        "filename": doc.filename,
        "char_count": doc.char_count,
        "chunk_count": len(chunks),
    })
    meta["chunk_count"] = indexes["vector_store"].size
    nb_store.update_notebook(notebook_id, meta)

    return {"message": "success", "chunks": len(chunks)}


@app.post("/api/notebooks/{notebook_id}/query")
def query(notebook_id: str, req: QueryRequest):
    """RAG 问答"""
    indexes = get_indexes(notebook_id)
    if indexes["vector_store"].size == 0:
        raise HTTPException(400, "No documents in this notebook")

    retriever = HybridRetriever(indexes["vector_store"], indexes["bm25_store"])
    llm_client = create_llm_client(
        provider=config.llm_provider,
        model=config.llm_model,
        api_key=config.llm_api_key or None,
    )
    pipeline = RAGPipeline(retriever, llm_client, top_k=config.final_top_k)

    response = pipeline.query(req.question, temperature=req.temperature)

    return {
        "answer": response.answer,
        "citations": [
            {
                "index": c.index,
                "filename": c.filename,
                "content_preview": c.content_preview,
            }
            for c in response.citations
        ],
    }


@app.post("/api/notebooks/{notebook_id}/generate")
def generate_content(notebook_id: str, req: GenerateRequest):
    """生成 FAQ / Study Guide 等"""
    indexes = get_indexes(notebook_id)
    if indexes["vector_store"].size == 0:
        raise HTTPException(400, "No documents in this notebook")

    llm_client = create_llm_client(
        provider=config.llm_provider,
        model=config.llm_model,
        api_key=config.llm_api_key or None,
    )
    generator = ContentGenerator(llm_client)
    result = generator.generate(
        indexes["vector_store"].chunks,
        output_type=req.output_type,
    )

    return {"content": result}