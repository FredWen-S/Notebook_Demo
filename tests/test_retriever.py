from models.document import Chunk
from core.vector_store import VectorStore
from core.bm25_store import BM25Store
from core.hybrid_retriever import HybridRetriever


def test_hybrid_retriever():
    # 准备 chunks
    chunks = [
        Chunk(source_id="doc1", content="机器学习是人工智能的一个重要分支", index=0),
        Chunk(source_id="doc1", content="深度学习使用多层神经网络来处理数据", index=1),
        Chunk(source_id="doc1", content="ResNet-50 是一种经典的卷积神经网络架构，在 ImageNet 上达到了很高的准确率",
              index=2),
        Chunk(source_id="doc1", content="BERT 是谷歌提出的预训练语言模型", index=3),
        Chunk(source_id="doc1", content="GPT 系列模型使用 Transformer 解码器架构", index=4),
        Chunk(source_id="doc1", content="注意力机制允许模型关注输入的不同部分", index=5),
        Chunk(source_id="doc1", content="随机森林是一种集成学习方法，结合多棵决策树", index=6),
        Chunk(source_id="doc1", content="支持向量机通过找到最优超平面来分类", index=7),
    ]

    # 建立索引
    vector_store = VectorStore(dimension=384)
    vector_store.add_chunks(chunks)

    bm25_store = BM25Store()
    bm25_store.add_chunks(chunks)

    # 混合检索器
    retriever = HybridRetriever(vector_store, bm25_store)

    # 测试 1: 语义搜索
    print("=" * 60)
    query1 = "如何让模型理解文本"
    print(f"查询 1（语义）: {query1}")
    results = retriever.search(query1, top_k=3)
    for i, r in enumerate(results):
        print(f"  {i + 1}. [RRF={r.score:.5f}] {r.chunk.content[:50]}")

    # 测试 2: 精确关键词搜索
    print()
    query2 = "ResNet-50 ImageNet"
    print(f"查询 2（关键词）: {query2}")
    results = retriever.search(query2, top_k=3)
    for i, r in enumerate(results):
        print(f"  {i + 1}. [RRF={r.score:.5f}] {r.chunk.content[:50]}")

    # 对比三种检索方式
    print("\n" + "=" * 60)
    query3 = "Transformer 架构"
    print(f"\n对比查询: {query3}")

    vec_results = retriever.search_vector_only(query3, top_k=3)
    bm25_results = retriever.search_bm25_only(query3, top_k=3)
    hybrid_results = retriever.search(query3, top_k=3)

    print("\n  向量检索 Top-3:")
    for i, r in enumerate(vec_results):
        print(f"    {i + 1}. {r.chunk.content[:50]}")

    print("\n  BM25 Top-3:")
    for i, r in enumerate(bm25_results):
        print(f"    {i + 1}. {r.chunk.content[:50]}")

    print("\n  混合检索 (RRF) Top-3:")
    for i, r in enumerate(hybrid_results):
        print(f"    {i + 1}. {r.chunk.content[:50]}")

    print("\n✓ 混合检索测试通过！")


if __name__ == "__main__":
    test_hybrid_retriever()