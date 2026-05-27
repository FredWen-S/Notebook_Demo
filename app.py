# app.py
"""Notebook Demo —— Streamlit 前端"""

import streamlit as st
import shutil
from pathlib import Path
from datetime import datetime

from config import config
from storage.db import NotebookStore
from core.parser import parse_file, SUPPORTED_TYPES
from core.chunker import RecursiveChunker
from core.vector_store import VectorStore
from core.bm25_store import BM25Store
from core.hybrid_retriever import HybridRetriever
from core.llm_client import create_llm_client
from core.rag_pipeline import RAGPipeline, RAGResponse
from core.generator import ContentGenerator
from models.document import Chunk

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="Notebook Demo",
    page_icon="📓",
    layout="wide",
)


# ============================================================
# 初始化
# ============================================================
@st.cache_resource
def get_notebook_store():
    return NotebookStore(config.notebooks_dir)


store = get_notebook_store()


def get_or_create_index(notebook_id: str):
    """获取或创建索引（缓存在 session_state 中）"""
    cache_key = f"index_{notebook_id}"
    if cache_key not in st.session_state:
        index_dir = store.get_index_dir(notebook_id)

        vector_store = VectorStore(dimension=config.embedding_dimension)
        bm25_store = BM25Store()

        # 尝试从磁盘加载
        if (index_dir / "faiss.index").exists():
            try:
                vector_store.load(index_dir)
                bm25_store.load(index_dir)
            except Exception as e:
                st.warning(f"加载索引失败: {e}，将重新建立。")

        st.session_state[cache_key] = {
            "vector_store": vector_store,
            "bm25_store": bm25_store,
        }

    return st.session_state[cache_key]


def get_pipeline(notebook_id: str) -> RAGPipeline:
    """创建 RAG Pipeline"""
    indexes = get_or_create_index(notebook_id)

    retriever = HybridRetriever(
        indexes["vector_store"],
        indexes["bm25_store"],
        rrf_k=config.rrf_k,
    )

    llm_client = create_llm_client(
        provider=config.llm_provider,
        model=config.llm_model,
        api_key=config.llm_api_key or None,
        base_url=config.llm_base_url or None,
    )

    reranker = None
    if config.use_reranker:
        from core.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(config.reranker_model)

    return RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        reranker=reranker,
        top_k=config.final_top_k,
    )


# ============================================================
# 侧边栏：Notebook 管理
# ============================================================
with st.sidebar:
    st.title("📓 Notebook Demo")
    st.caption("类 NotebookLM 的本地 RAG 知识库问答系统")

    st.divider()

    # 创建新 Notebook
    with st.expander("➕ 创建新 Notebook", expanded=False):
        new_name = st.text_input("Notebook 名称", key="new_nb_name")
        if st.button("创建", key="create_nb"):
            if new_name.strip():
                meta = store.create_notebook(new_name.strip())
                st.success(f"已创建: {new_name}")
                st.rerun()
            else:
                st.warning("请输入名称")

    st.divider()

    # Notebook 列表
    st.subheader("我的 Notebooks")
    notebooks = store.list_notebooks()

    if not notebooks:
        st.info("还没有 Notebook，请先创建一个。")
        st.stop()

    # 选择 Notebook
    nb_options = {nb["id"]: nb["name"] for nb in notebooks}
    selected_nb_id = st.selectbox(
        "选择 Notebook",
        options=list(nb_options.keys()),
        format_func=lambda x: nb_options[x],
        key="selected_nb",
    )

    if selected_nb_id:
        nb_meta = store.get_notebook(selected_nb_id)
        st.caption(f"创建于: {nb_meta['created_at'][:10]}")
        st.caption(f"文档数: {len(nb_meta['sources'])}")
        st.caption(f"Chunk 数: {nb_meta['chunk_count']}")

    st.divider()

    # LLM 设置
    with st.expander("⚙️ LLM 设置", expanded=False):
        provider = st.selectbox(
            "Provider",
            ["ollama", "openai", "claude", "gemini"],
            key="llm_provider",
        )
        config.llm_provider = provider

        if provider == "ollama":
            config.llm_model = st.text_input("模型", value="qwen2.5:7b", key="llm_model")
        else:
            config.llm_model = st.text_input("模型", value="gpt-4o-mini", key="llm_model")
            config.llm_api_key = st.text_input("API Key", type="password", key="api_key")

# ============================================================
# 主区域
# ============================================================
if not selected_nb_id:
    st.info("请在左侧选择一个 Notebook。")
    st.stop()

nb_meta = store.get_notebook(selected_nb_id)
if nb_meta is None:
    st.error(f"Notebook {selected_nb_id} 不存在。")
    st.stop()

# 两列布局：左边是源文件管理，右边是对话
col_sources, col_chat = st.columns([1, 2])

# ---- 左列：文档管理 ----
with col_sources:
    st.subheader("📄 文档源")

    # 文件上传
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        key=f"upload_{selected_nb_id}",
    )

    if uploaded_files:
        if st.button("📥 处理文档", key="process_docs"):
            with st.spinner("正在处理文档..."):
                chunker = RecursiveChunker(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
                indexes = get_or_create_index(selected_nb_id)
                all_new_chunks = []

                for uploaded_file in uploaded_files:
                    # 保存文件
                    source_dir = store.get_sources_dir(selected_nb_id)
                    file_path = source_dir / uploaded_file.name
                    file_path.write_bytes(uploaded_file.read())

                    try:
                        # 解析
                        doc = parse_file(file_path)
                        st.write(f"✅ {uploaded_file.name} ({doc.char_count} 字符)")

                        # 切分
                        chunks = chunker.chunk_document(doc)
                        all_new_chunks.extend(chunks)

                        # 更新元数据
                        nb_meta["sources"].append({
                            "source_id": doc.source_id,
                            "filename": doc.filename,
                            "char_count": doc.char_count,
                            "chunk_count": len(chunks),
                            "added_at": datetime.now().isoformat(),
                        })

                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {e}")

                if all_new_chunks:
                    # 添加到向量库
                    st.write("正在建立向量索引...")
                    indexes["vector_store"].add_chunks(
                        all_new_chunks,
                        model_name=config.embedding_model,
                    )

                    # 添加到 BM25
                    st.write("正在建立 BM25 索引...")
                    indexes["bm25_store"].add_chunks(all_new_chunks)

                    # 保存索引
                    index_dir = store.get_index_dir(selected_nb_id)
                    indexes["vector_store"].save(index_dir)
                    indexes["bm25_store"].save(index_dir)

                    # 更新元数据
                    nb_meta["chunk_count"] = indexes["vector_store"].size
                    store.update_notebook(selected_nb_id, nb_meta)

                    st.success(f"处理完成！新增 {len(all_new_chunks)} 个 chunk。")

    # 显示已有文档
    if nb_meta["sources"]:
        st.divider()
        for src in nb_meta["sources"]:
            st.write(f"📎 {src['filename']} ({src['chunk_count']} chunks)")

    # 生成功能
    st.divider()
    st.subheader("📝 生成内容")

    gen_type = st.selectbox(
        "选择类型",
        ["faq", "study_guide", "briefing_doc", "audio_script"],
        format_func=lambda x: {
            "faq": "📋 FAQ 常见问题",
            "study_guide": "📖 学习指南",
            "briefing_doc": "📊 简报文档",
            "audio_script": "🎙️ 播客脚本",
        }[x],
        key="gen_type",
    )

    if st.button("生成", key="generate"):
        indexes = get_or_create_index(selected_nb_id)
        if indexes["vector_store"].size == 0:
            st.warning("请先上传文档。")
        else:
            with st.spinner("正在生成..."):
                try:
                    llm_client = create_llm_client(
                        provider=config.llm_provider,
                        model=config.llm_model,
                        api_key=config.llm_api_key or None,
                    )
                    generator = ContentGenerator(llm_client)
                    # 用所有 chunk 来生成
                    all_chunks = indexes["vector_store"].chunks
                    result = generator.generate(all_chunks, output_type=gen_type)
                    st.session_state[f"generated_{selected_nb_id}"] = result
                except Exception as e:
                    st.error(f"生成失败: {e}")

    # 显示生成结果
    gen_key = f"generated_{selected_nb_id}"
    if gen_key in st.session_state:
        st.text_area("生成结果", st.session_state[gen_key], height=300)

# ---- 右列：对话 ----
with col_chat:
    st.subheader("💬 基于资料的问答")

    # 初始化聊天记录
    chat_key = f"chat_{selected_nb_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # 显示聊天记录
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📎 引用来源"):
                    for c in msg["citations"]:
                        st.caption(f"[{c['index']}] {c['filename']}: {c['preview'][:80]}...")

    # 用户输入
    if question := st.chat_input("输入你的问题...", key=f"input_{selected_nb_id}"):
        # 检查是否有文档
        indexes = get_or_create_index(selected_nb_id)
        if indexes["vector_store"].size == 0:
            st.warning("请先上传文档再提问。")
        else:
            # 添加用户消息
            st.session_state[chat_key].append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("user"):
                st.write(question)

            # 生成回答
            with st.chat_message("assistant"):
                with st.spinner("正在检索和生成回答..."):
                    try:
                        pipeline = get_pipeline(selected_nb_id)
                        response = pipeline.query(question)

                        st.write(response.answer)

                        # 显示引用
                        citations_data = []
                        if response.citations:
                            with st.expander("📎 引用来源"):
                                for c in response.citations:
                                    st.caption(
                                        f"[{c.index}] {c.filename}: {c.content_preview[:80]}..."
                                    )
                                    citations_data.append({
                                        "index": c.index,
                                        "filename": c.filename,
                                        "preview": c.content_preview,
                                    })

                        # 保存到聊天记录
                        st.session_state[chat_key].append({
                            "role": "assistant",
                            "content": response.answer,
                            "citations": citations_data,
                        })

                    except Exception as e:
                        error_msg = f"生成回答失败: {e}"
                        st.error(error_msg)
                        st.session_state[chat_key].append({
                            "role": "assistant",
                            "content": error_msg,
                        })