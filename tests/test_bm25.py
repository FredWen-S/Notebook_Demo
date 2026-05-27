from models.document import Chunk
from core.bm25_store import BM25Store


def test_bm25():
    chunks = [
        Chunk(source_id="doc1", content="机器学习是人工智能的一个重要分支领域", index=0),
        Chunk(source_id="doc1", content="深度学习使用多层神经网络来处理复杂数据", index=1),
        Chunk(source_id="doc1", content="ResNet-50 是一种经典的卷积神经网络架构", index=2),
        Chunk(source_id="doc1", content="BERT 模型在自然语言处理任务中表现优异", index=3),
        Chunk(source_id="doc1", content="随机森林是一种集成学习方法", index=4),
    ]

    store = BM25Store()
    store.add_chunks(chunks)

    # 测试精确关键词搜索
    query = "ResNet-50"
    results = store.search(query, top_k=3)

    print(f"查询: {query}")
    print(f"结果:")
    for i, r in enumerate(results):
        print(f"  {i + 1}. [BM25={r.score:.4f}] {r.chunk.content}")

    # ResNet-50 应该排名第一
    assert len(results) > 0
    assert "ResNet" in results[0].chunk.content

    print("\n✓ BM25 测试通过！")


if __name__ == "__main__":
    test_bm25()