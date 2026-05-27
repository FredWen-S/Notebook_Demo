# tests/test_rag.py
"""端到端测试 RAG Pipeline"""

from models.document import Chunk
from core.vector_store import VectorStore
from core.bm25_store import BM25Store
from core.hybrid_retriever import HybridRetriever
from core.llm_client import create_llm_client
from core.rag_pipeline import RAGPipeline


def test_rag_pipeline():
    # 1. 准备测试数据
    chunks = [
        Chunk(
            source_id="doc1", index=0,
            content="梯度下降是一种优化算法，用于最小化损失函数。它通过计算损失函数对参数的梯度，然后沿梯度的反方向更新参数。学习率控制每次更新的步长大小。",
            metadata={"filename": "ml_basics.pdf"},
        ),
        Chunk(
            source_id="doc1", index=1,
            content="过拟合是指模型在训练数据上表现很好，但在新数据上表现差。常见的解决方法包括：增加训练数据、使用正则化、Dropout、早停法等。",
            metadata={"filename": "ml_basics.pdf"},
        ),
        Chunk(
            source_id="doc2", index=0,
            content="Transformer 架构由 Vaswani 等人在 2017 年提出，核心是自注意力机制。它摒弃了 RNN 的循环结构，可以并行处理序列，大幅提升了训练效率。",
            metadata={"filename": "transformer.md"},
        ),
        Chunk(
            source_id="doc2", index=1,
            content="BERT 使用 Transformer 的编码器部分，通过掩码语言模型和下一句预测来预训练。GPT 使用 Transformer 的解码器部分，采用自回归方式训练。",
            metadata={"filename": "transformer.md"},
        ),
    ]

    # 2. 建立索引
    vector_store = VectorStore(dimension=384)
    vector_store.add_chunks(chunks)

    bm25_store = BM25Store()
    bm25_store.add_chunks(chunks)

    # 3. 创建组件
    retriever = HybridRetriever(vector_store, bm25_store)
    llm_client = create_llm_client("ollama", model="qwen2.5:7b")
    pipeline = RAGPipeline(retriever, llm_client, top_k=3)

    # 4. 提问
    question = "什么是梯度下降？学习率有什么作用？"
    print(f"问题: {question}\n")

    response = pipeline.query(question)

    print(f"回答:\n{response.answer}\n")
    print(f"引用:")
    for c in response.citations:
        print(f"  [{c.index}] {c.filename}: {c.content_preview[:50]}...")

    print("\n✓ RAG Pipeline 测试通过！")


if __name__ == "__main__":
    test_rag_pipeline()