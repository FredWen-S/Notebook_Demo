from pathlib import Path
from core.parser import parse_file


def test_parse_txt():
    """测试 TXT 解析"""
    # 创建一个临时 TXT 文件
    test_file = Path("data/test_sample.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(
        "这是第一段。机器学习是人工智能的一个子领域。\n\n"
        "这是第二段。深度学习是机器学习的一种方法。\n\n"
        "这是第三段。神经网络是深度学习的基础。",
        encoding="utf-8"
    )

    doc = parse_file(test_file)

    print(f"文件名: {doc.filename}")
    print(f"类型: {doc.file_type}")
    print(f"字符数: {doc.char_count}")
    print(f"内容预览: {doc.content[:100]}...")
    print(f"source_id: {doc.source_id}")

    assert doc.file_type == "txt"
    assert doc.char_count > 0
    assert "机器学习" in doc.content
    print("\n✓ TXT 解析测试通过！")


if __name__ == "__main__":
    test_parse_txt()