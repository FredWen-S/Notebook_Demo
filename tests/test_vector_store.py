# tests/test_vector_store.py
"""测试向量存储和检索"""

from models.document import Chunk
from core.vector_store import VectorStore


def test_vector_store():
    # 准备测试 chunks
    chunks = [
        Chunk(source_id="doc1", content="机器学习是人工智能的一个重要分支", index=0),
        Chunk(source_id="doc1", content="深度学习使用多层神经网络来处理数据", index=1),
        Chunk(source_id="doc1", content="自然语言处理让计算机能够理解人类语言", index=2),
        Chunk(source_id="doc1", content="计算机视觉让机器能够看懂图片和视频", index=3),
        Chunk(source_id="doc1", content="强化学习通过奖励机制让智能体学会决策", index=4),
    ]

    # 创建向量库
    store = VectorStore(dimension=384)
    store.add_chunks(chunks)

    print(f"\n向量库大小: {store.size}")

    # 测试搜索
    query = "什么是神经网络"
    results = store.search(query, top_k=3)

    print(f"\n查询: {query}")
    print(f"Top-3 结果:")
    for i, r in enumerate(results):
        print(f"  {i + 1}. [分数={r.score:.4f}] {r.chunk.content}")

    # 验证
    assert len(results) > 0
    # "深度学习使用多层神经网络" 应该排名靠前
    assert "神经网络" in results[0].chunk.content or "深度学习" in results[0].chunk.content

    # 测试保存和加载
    store.save("data/test_vector_store")

    store2 = VectorStore()
    store2.load("data/test_vector_store")
    results2 = store2.search(query, top_k=3)

    assert len(results2) == len(results)
    print("\n✓ 向量存储测试通过！")


if __name__ == "__main__":
    test_vector_store()