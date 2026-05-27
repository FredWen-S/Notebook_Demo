# tests/test_chunker.py
"""测试文本切分"""

from core.chunker import RecursiveChunker
from models.document import ParsedDocument


def test_chunker():
    # 创建一个假文档
    doc = ParsedDocument(
        filename="test.txt",
        file_type="txt",
        content=(
            "机器学习是人工智能的一个子领域，它致力于开发能够从数据中学习的算法。"
            "通过机器学习，计算机可以在没有被明确编程的情况下完成任务。\n\n"
            "监督学习是机器学习的一种主要方法。在监督学习中，模型从带有标签的训练数据中学习。"
            "常见的监督学习算法包括线性回归、决策树和支持向量机。\n\n"
            "无监督学习是另一种重要方法。它不需要标签数据，而是从数据中发现隐藏的模式。"
            "聚类和降维是常见的无监督学习任务。\n\n"
            "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。"
            "卷积神经网络擅长图像识别，循环神经网络擅长序列数据处理。"
            "Transformer 架构近年来在自然语言处理领域取得了巨大突破。"
        )
    )

    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    print(f"原文长度: {len(doc.content)} 字符")
    print(f"切分成 {len(chunks)} 个 chunk:")
    print("-" * 50)

    for chunk in chunks:
        print(f"\n[Chunk {chunk.index}] ({len(chunk.content)} 字符)")
        print(f"  内容: {chunk.content[:80]}...")
        print(f"  元数据: {chunk.metadata}")

    # 验证
    assert len(chunks) > 1, "应该切成多个 chunk"
    for chunk in chunks:
        assert len(chunk.content) > 0, "chunk 不应为空"
        assert chunk.source_id == doc.source_id

    print("\n✓ Chunker 测试通过！")


if __name__ == "__main__":
    test_chunker()