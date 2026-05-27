# core/parser.py
"""文档解析模块：支持 PDF / Markdown / TXT"""

import re
from pathlib import Path
from typing import Optional

# PDF 解析
import fitz  # PyMuPDF

# Markdown 解析
import markdown
from html.parser import HTMLParser

from models.document import ParsedDocument


# ============================================================
# HTML 标签清理器（用于 Markdown 解析后去掉 HTML 标签）
# ============================================================
class _HTMLStripper(HTMLParser):
    """把 HTML 标签去掉，只保留纯文本"""

    def __init__(self):
        super().__init__()
        self.result = []
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        # 段落和标题结束后加换行
        if tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br"):
            self.result.append("\n")
        self.current_tag = None

    def handle_data(self, data):
        self.result.append(data)

    def get_text(self) -> str:
        return "".join(self.result)


def _strip_html(html_text: str) -> str:
    """去掉 HTML 标签"""
    stripper = _HTMLStripper()
    stripper.feed(html_text)
    return stripper.get_text()


# ============================================================
# 文本清理
# ============================================================
def _clean_text(text: str) -> str:
    """清理提取出的文本"""
    # 合并连续的空行为一个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去掉每行首尾空白
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # 去掉首尾空白
    text = text.strip()
    return text


# ============================================================
# PDF 解析
# ============================================================
def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """
    解析 PDF 文件。

    Args:
        file_path: PDF 文件路径

    Returns:
        ParsedDocument: 解析后的文档对象
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    doc = fitz.open(str(file_path))

    pages_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # 提取纯文本
        if text.strip():
            pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    full_text = _clean_text(full_text)

    result = ParsedDocument(
        filename=file_path.name,
        file_type="pdf",
        content=full_text,
        page_count=len(doc),
    )

    doc.close()
    return result


# ============================================================
# Markdown 解析
# ============================================================
def parse_markdown(file_path: str | Path) -> ParsedDocument:
    """
    解析 Markdown 文件。

    先把 Markdown 转成 HTML，再去掉 HTML 标签，得到纯文本。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    raw_md = file_path.read_text(encoding="utf-8")

    # Markdown → HTML → 纯文本
    html = markdown.markdown(raw_md, extensions=["tables", "fenced_code"])
    text = _strip_html(html)
    text = _clean_text(text)

    return ParsedDocument(
        filename=file_path.name,
        file_type="md",
        content=text,
    )


# ============================================================
# TXT 解析
# ============================================================
def parse_txt(file_path: str | Path) -> ParsedDocument:
    """解析纯文本文件"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    text = _clean_text(text)

    return ParsedDocument(
        filename=file_path.name,
        file_type="txt",
        content=text,
    )


# ============================================================
# 统一入口
# ============================================================
SUPPORTED_TYPES = {".pdf", ".md", ".markdown", ".txt"}


def parse_file(file_path: str | Path) -> ParsedDocument:
    """
    自动识别文件类型并解析。

    Args:
        file_path: 文件路径

    Returns:
        ParsedDocument

    Raises:
        ValueError: 不支持的文件类型
        FileNotFoundError: 文件不存在
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in (".md", ".markdown"):
        return parse_markdown(file_path)
    elif suffix == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(
            f"不支持的文件类型: {suffix}。支持的类型: {SUPPORTED_TYPES}"
        )