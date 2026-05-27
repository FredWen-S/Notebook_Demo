# 从 0 开始手搓一个本地 Notebook Demo

## 类 NotebookLM 的本地 RAG 知识库问答系统 —— 完整实践手册

> **适用读者**：Python / CS 基础较弱，但想学习 AI Agent、RAG、NotebookLM 类产品实现的学生
>
> **最终成果**：一个完全本地运行的"资料问答系统"，支持上传文档、智能检索、基于资料回答问题、引用来源

---

# 目录

- [第 0 章 · 你将构建什么](#第-0-章--你将构建什么)
- [第 1 章 · 环境搭建](#第-1-章--环境搭建)
- [第 2 章 · 核心概念速通](#第-2-章--核心概念速通)
- [第 3 章 · 文档解析 Document Parsing](#第-3-章--文档解析-document-parsing)
- [第 4 章 · 文本切分 Chunking](#第-4-章--文本切分-chunking)
- [第 5 章 · Embedding 与向量数据库](#第-5-章--embedding-与向量数据库)
- [第 6 章 · BM25 关键词检索](#第-6-章--bm25-关键词检索)
- [第 7 章 · 混合检索与 RRF 融合](#第-7-章--混合检索与-rrf-融合)
- [第 8 章 · 对接 LLM —— Ollama 与 API](#第-8-章--对接-llm--ollama-与-api)
- [第 9 章 · RAG Pipeline 完整串联](#第-9-章--rag-pipeline-完整串联)
- [第 10 章 · Streamlit 前端](#第-10-章--streamlit-前端)
- [第 11 章 · 进阶功能](#第-11-章--进阶功能)
- [第 12 章 · 进阶版：FastAPI 后端重构](#第-12-章--进阶版fastapi-后端重构)
- [第 13 章 · 部署、评估与简历](#第-13-章--部署评估与简历)
- [附录 A · 用 LangChain / LlamaIndex 重构](#附录-a--用-langchain--llamaindex-重构)
- [附录 B · 常见报错速查表](#附录-b--常见报错速查表)
- [附录 C · 简历 Bullet Points](#附录-c--简历-bullet-points)

---

# 第 0 章 · 你将构建什么

## 0.1 项目演示

想象一下：你有一堆 PDF 论文、Markdown 笔记、TXT 文件。你想问一个问题，系统能从这些资料里找到答案，并告诉你"答案来自哪个文件的哪一段"。这就是我们要做的 **Notebook Demo**。

最终你可以这样使用它：

```bash
# 启动服务
streamlit run app.py

# 然后在浏览器里：
# 1. 创建一个 Notebook（比如"机器学习笔记"）
# 2. 上传几篇 PDF / Markdown / TXT
# 3. 系统自动解析、切分、建索引
# 4. 在对话框里问问题，比如"什么是梯度下降？"
# 5. 系统返回基于你上传资料的回答，并标注引用来源
# 6. 还可以一键生成 FAQ、Study Guide、Briefing Doc
```

## 0.2 完整项目结构

以下是我们最终要实现的项目文件结构，**每个文件的作用**都标注了：

```
notebook_demo/
│
├── app.py                      # Streamlit 前端入口
├── config.py                   # 全局配置（模型、路径、参数）
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明
│
├── core/                       # 核心逻辑（后端 pipeline）
│   ├── __init__.py
│   ├── parser.py               # 文档解析：PDF / Markdown / TXT → 纯文本
│   ├── chunker.py              # 文本切分：把长文本切成小段
│   ├── embedder.py             # Embedding：把文本变成向量
│   ├── vector_store.py         # 向量数据库：存储和检索向量
│   ├── bm25_store.py           # BM25 检索：关键词搜索
│   ├── hybrid_retriever.py     # 混合检索 + RRF 融合
│   ├── reranker.py             # （可选）重排序器
│   ├── llm_client.py           # LLM 调用客户端
│   ├── rag_pipeline.py         # RAG 完整 pipeline
│   └── generator.py            # 生成 FAQ / Study Guide / Briefing Doc
│
├── models/                     # 数据模型（Pydantic）
│   ├── __init__.py
│   ├── document.py             # 文档相关数据结构
│   └── notebook.py             # Notebook 相关数据结构
│
├── storage/                    # 持久化存储
│   ├── __init__.py
│   └── db.py                   # SQLite 数据库操作
│
├── data/                       # 运行时数据（自动创建）
│   ├── notebooks/              # 每个 notebook 的数据
│   ├── uploads/                # 上传的原始文件
│   └── notebook_demo.db        # SQLite 数据库文件
│
├── tests/                      # 测试
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_rag.py
│
└── docs/                       # 补充文档
    └── architecture.md         # 架构说明
```

## 0.3 技术栈一览

| 层级 | 技术 | 作用 |
|------|------|------|
| 前端 | Streamlit | 网页界面 |
| 文档解析 | PyMuPDF (fitz) | 解析 PDF |
| 文档解析 | markdown, pathlib | 解析 Markdown / TXT |
| 文本切分 | 手写 RecursiveChunker | 把长文本切成小段 |
| Embedding | sentence-transformers | 把文本变成向量 |
| 向量存储 | FAISS | 向量相似度搜索 |
| 关键词检索 | rank-bm25 | BM25 关键词搜索 |
| 混合检索 | 手写 RRF | 融合向量 + 关键词结果 |
| 重排序 | cross-encoder（可选）| 精排 |
| LLM | Ollama（本地）/ OpenAI / Claude / Gemini（API）| 生成回答 |
| 数据模型 | Pydantic | 数据结构定义 |
| 存储 | SQLite | 元数据持久化 |
| 配置 | Python dataclass | 配置管理 |

---

# 第 1 章 · 环境搭建

## 1.1 目标

搭建好开发环境，确保所有工具都能正常工作。

## 1.2 安装 Python

### Windows

1. 访问 https://www.python.org/downloads/ ，下载 Python 3.11 或 3.12
2. **安装时勾选 "Add Python to PATH"**（非常重要！）
3. 打开 PowerShell 或 CMD，验证：

```bash
python --version
# 应该看到 Python 3.11.x 或 3.12.x
```

> **Windows 常见问题**：如果输入 `python` 打开了 Microsoft Store，去"设置 → 应用 → 应用执行别名"，关掉 "python.exe" 和 "python3.exe" 的应用安装程序。

### macOS

```bash
# 推荐用 Homebrew
brew install python@3.11

python3 --version
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

## 1.3 创建项目

```bash
# 创建项目文件夹
mkdir notebook_demo
cd notebook_demo

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# 激活后，命令行前面会出现 (.venv)，说明你在虚拟环境里了
```

> **为什么要虚拟环境？** 虚拟环境就像一个"隔离房间"，项目的依赖库都装在这个房间里，不会和你电脑上其他 Python 项目冲突。

## 1.4 安装依赖

创建 `requirements.txt`：

```txt
# ---- 前端 ----
streamlit>=1.30.0

# ---- 文档解析 ----
PyMuPDF>=1.23.0          # PDF 解析，import 时用 fitz
markdown>=3.5             # Markdown 转 HTML

# ---- NLP / Embedding ----
sentence-transformers>=2.2.0   # 生成 embedding 向量
torch>=2.0.0                    # PyTorch，sentence-transformers 的依赖

# ---- 向量数据库 ----
faiss-cpu>=1.7.4          # Facebook 的向量检索库（CPU 版本）

# ---- 关键词检索 ----
rank-bm25>=0.2.2          # BM25 算法

# ---- 中文分词（如果需要处理中文文档）----
jieba>=0.42.1

# ---- 数据模型 ----
pydantic>=2.0

# ---- LLM 客户端 ----
httpx>=0.25.0             # HTTP 客户端，用于调用 Ollama API
openai>=1.10.0            # OpenAI SDK（也可调用兼容 API）

# ---- 可选：Reranker ----
# 取消下面的注释来安装 cross-encoder reranker
# sentence-transformers 已包含，无需额外安装

# ---- 开发工具 ----
pytest>=7.0
```

安装：

```bash
pip install -r requirements.txt
```

> **安装时间**：首次安装可能需要 10-20 分钟，因为 PyTorch 和 sentence-transformers 比较大。
>
> **Windows 注意**：如果 `faiss-cpu` 安装失败，试试：
> ```bash
> pip install faiss-cpu --no-cache-dir
> ```
> 如果还是不行，可以用 `conda install -c conda-forge faiss-cpu`。

## 1.5 安装 Ollama（本地 LLM）

Ollama 让你在自己电脑上运行大语言模型，不需要联网、不花钱。

### 安装

- **Windows / macOS**：访问 https://ollama.com 下载安装
- **Linux**：`curl -fsSL https://ollama.ai/install.sh | sh`

### 下载模型

```bash
# 推荐：qwen2.5（中英文都好，7B 大小适中）
ollama pull qwen2.5:7b

# 备选：llama3.1（英文强）
ollama pull llama3.1:8b

# 小模型（内存不够时用）
ollama pull qwen2.5:3b
```

### 验证

```bash
ollama run qwen2.5:7b "你好，请用一句话介绍自己"
```

如果能看到模型回复，说明 Ollama 安装成功。

> **硬件要求**：
> - 7B 模型：至少 8GB 内存
> - 3B 模型：至少 4GB 内存
> - 如果电脑配置低，可以先用 API 方案（见第 8 章）

## 1.6 验收标准

运行以下 Python 脚本，全部通过就说明环境 OK：

```python
# test_env.py
"""环境检查脚本 —— 运行方式：python test_env.py"""

def check_import(name, import_name=None):
    """尝试导入一个包"""
    try:
        __import__(import_name or name)
        print(f"  ✓ {name}")
        return True
    except ImportError:
        print(f"  ✗ {name} —— 请运行: pip install {name}")
        return False

def check_ollama():
    """检查 Ollama 是否可用"""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = r.json().get("models", [])
        print(f"  ✓ Ollama 运行中，已安装 {len(models)} 个模型")
        for m in models:
            print(f"    - {m['name']}")
        return True
    except Exception:
        print("  ✗ Ollama 未运行 —— 请启动 Ollama")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Notebook Demo 环境检查")
    print("=" * 50)

    print("\n[1] Python 包:")
    all_ok = True
    all_ok &= check_import("streamlit")
    all_ok &= check_import("PyMuPDF", "fitz")
    all_ok &= check_import("markdown")
    all_ok &= check_import("sentence-transformers", "sentence_transformers")
    all_ok &= check_import("faiss-cpu", "faiss")
    all_ok &= check_import("rank-bm25", "rank_bm25")
    all_ok &= check_import("pydantic")
    all_ok &= check_import("httpx")
    all_ok &= check_import("openai")
    all_ok &= check_import("jieba")

    print("\n[2] Ollama:")
    all_ok &= check_ollama()

    print("\n" + "=" * 50)
    if all_ok:
        print("全部通过！可以开始开发了。")
    else:
        print("有未通过的项，请按提示修复。")
    print("=" * 50)
```

## 1.7 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError` | 没在虚拟环境里安装 | 确认 `(.venv)` 出现在命令行前 |
| `torch` 安装很慢 | 包很大（~2GB）| 耐心等待，或用清华源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch` |
| PowerShell 禁止运行脚本 | 执行策略限制 | 运行 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Ollama 连接失败 | Ollama 没启动 | Windows 任务栏找 Ollama 图标；或命令行运行 `ollama serve` |

---

# 第 2 章 · 核心概念速通

在写代码之前，我们先搞懂几个核心概念。这一章不写代码，只讲原理。如果你已经懂了，可以直接跳到第 3 章。

## 2.1 什么是 RAG？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

普通的大语言模型（LLM）就像一个"读过很多书但不能翻书的学生"——它只能靠记忆回答问题，可能会编造答案（幻觉）。

RAG 的思路是：**先从你的资料里检索相关内容，再把这些内容交给 LLM，让它基于资料来回答**。就像"开卷考试"。

```
用户问题  →  检索相关资料  →  把资料 + 问题一起交给 LLM  →  基于资料的回答
```

**在我们的 Notebook Demo 里**：用户上传文档，系统建立索引；用户问问题时，系统先从文档中检索相关段落，再交给 LLM 生成回答。

## 2.2 什么是 Embedding？

**Embedding = 把文本变成一组数字（向量）**

人类理解文本靠"语义"，计算机理解不了。Embedding 就是把文本翻译成计算机能理解的数字。

```
"机器学习是人工智能的一个分支" → [0.12, -0.45, 0.78, 0.33, ...]
"Machine Learning is a branch of AI" → [0.11, -0.44, 0.77, 0.34, ...]
                                         ↑ 语义相近，向量也相近！
```

两段话意思相近，它们的向量就会"靠得近"。这样我们就能用数学方法来找"语义相似"的文本。

**在我们的 Notebook Demo 里**：每个文档切成的小段（chunk）都会被转成向量，用户的问题也转成向量，然后找"最近的"chunk。

## 2.3 什么是向量数据库？

**向量数据库 = 专门存储向量、支持"找最近的向量"的数据库**

普通数据库（如 MySQL）擅长精确查找，比如"找 id=123 的用户"。向量数据库擅长"近似查找"，比如"找和这个向量最相似的 10 个向量"。

我们用 **FAISS**（Facebook AI Similarity Search），它是一个高效的向量检索库。

**在我们的 Notebook Demo 里**：所有 chunk 的向量存在 FAISS 里，查询时找出最相似的 top-K 个 chunk。

## 2.4 什么是 BM25？

**BM25 = 一种经典的关键词检索算法**

你在搜索引擎里输入关键词，搜索引擎怎么知道哪些网页最相关？BM25 就是做这个的。它根据"关键词出现的频率"和"文档长度"来打分。

```
查询："梯度下降 学习率"
文档A（包含"梯度下降"3次，"学习率"2次）→ 高分
文档B（包含"梯度下降"1次）→ 中分
文档C（不包含这些词）→ 0分
```

**为什么向量检索之外还需要 BM25？** 因为向量检索擅长找"语义相似"的内容，但有时用户搜索的是精确的术语、人名、编号，这时关键词检索更准。两者互补。

## 2.5 什么是混合检索（Hybrid Search）和 RRF？

**混合检索 = 同时用向量检索 + 关键词检索，然后合并结果**

问题来了：两个系统各自返回一个排名列表，怎么合并？

**RRF = Reciprocal Rank Fusion（倒数排名融合）**，是一种简单有效的融合方法：

```
RRF_score(doc) = Σ 1 / (k + rank_i)
```

其中 `rank_i` 是 doc 在第 i 个系统中的排名，`k` 是一个常数（通常取 60）。

简单说：一个文档在两个系统中排名都靠前，它的 RRF 分数就高；只在一个系统中排名靠前，分数也不差，但不如两边都靠前。

**在我们的 Notebook Demo 里**：向量检索返回 top-20，BM25 返回 top-20，用 RRF 融合后取 top-5 交给 LLM。

## 2.6 什么是 Reranker？

**Reranker = 重排序器，对初步检索结果做精细排序**

初步检索（向量 + BM25）速度快但不够精确。Reranker 用一个更强的模型对候选结果逐一打分，给出更准确的排名。

类比：初步检索像"海选"，Reranker 像"评委精选"。

**在我们的 Notebook Demo 里**：Reranker 是可选的，加上它效果更好，但会稍慢。

## 2.7 什么是 Chunk？

**Chunk = 把长文档切成的一小段一小段**

为什么不把整篇文档一起查？因为：
1. LLM 有输入长度限制
2. 整篇文档里只有一小段是相关的，全塞进去会干扰回答
3. 向量检索对短文本效果更好

切分策略很重要：太大了不精确，太小了丢失上下文。我们会实现一个"递归切分器"。

## 2.8 什么是 Citation（引用）？

**Citation = 告诉用户"这个回答来自哪个文件的哪一段"**

这是 NotebookLM 的核心特性之一。用户需要知道答案的来源，以便验证。

**在我们的 Notebook Demo 里**：每个回答都会附带引用标记，比如 `[1]`，点击可以看到原文。

## 2.9 整体流程图

```
┌──────────────────────────────────────────────────────────────┐
│                      用户上传文档                              │
│                   (PDF / MD / TXT)                            │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               文档解析 (parser.py)                             │
│            提取纯文本 + 元数据（标题、页码等）                   │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               文本切分 (chunker.py)                            │
│         把长文本切成 300-500 字的小段（chunk）                   │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
              ┌─────────┴─────────┐
              ▼                   ▼
┌──────────────────┐  ┌──────────────────────┐
│   生成 Embedding  │  │   建立 BM25 索引      │
│  (embedder.py)   │  │  (bm25_store.py)     │
│   存入 FAISS     │  │   存入内存            │
│ (vector_store.py)│  │                      │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    用户提问                                    │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│              混合检索 (hybrid_retriever.py)                    │
│    向量检索 top-K  +  BM25 top-K  →  RRF 融合  →  top-N      │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│              （可选）Reranker 重排序                            │
│                  (reranker.py)                                │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                RAG Pipeline (rag_pipeline.py)                 │
│     组装 Prompt：系统指令 + 检索到的 chunks + 用户问题          │
│                 → 调用 LLM → 生成回答                         │
└───────────────────────┬──────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────┐
│              返回带 Citation 的回答给用户                       │
└──────────────────────────────────────────────────────────────┘
```

---

# 第 3 章 · 文档解析 Document Parsing

## 3.1 目标

实现 `core/parser.py`，能把 PDF、Markdown、TXT 文件解析成统一的纯文本格式。

## 3.2 原理

不同格式的文件，内部结构完全不同：
- **PDF**：二进制格式，内部是页面对象、字体、图片等，需要专门的库来提取文本
- **Markdown**：纯文本 + 标记语法（`#`、`**`、`-` 等），需要去掉标记
- **TXT**：最简单，直接读取即可

我们的解析器要把它们统一变成这样的结构：

```python
{
    "source_id": "abc123",           # 文件唯一标识
    "filename": "paper.pdf",         # 文件名
    "file_type": "pdf",              # 文件类型
    "content": "这是提取出的全文...",   # 纯文本内容
    "metadata": {                    # 元数据
        "pages": 10,
        "title": "论文标题"
    }
}
```

## 3.3 数据模型

先定义数据结构。创建 `models/document.py`：

```python
# models/document.py
"""文档相关的数据模型"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class ParsedDocument(BaseModel):
    """解析后的文档"""
    source_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    filename: str
    file_type: str          # "pdf" / "md" / "txt"
    content: str            # 提取的纯文本
    page_count: int = 0     # PDF 的页数
    char_count: int = 0     # 字符数
    created_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context):
        """自动计算字符数"""
        if self.char_count == 0:
            self.char_count = len(self.content)


class Chunk(BaseModel):
    """切分后的文本块"""
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str          # 所属文档的 ID
    content: str            # chunk 的文本内容
    index: int              # 在文档中的序号（第几个 chunk）
    metadata: dict = Field(default_factory=dict)
    # metadata 可能包含：page_number, section_title 等


class RetrievedChunk(BaseModel):
    """检索到的 chunk（附带分数）"""
    chunk: Chunk
    score: float            # 检索分数
    source: str             # 来自哪个检索系统："vector" / "bm25" / "rrf"
```

创建 `models/__init__.py`：

```python
# models/__init__.py
from .document import ParsedDocument, Chunk, RetrievedChunk
```

## 3.4 解析器代码

创建 `core/parser.py`：

```python
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
```

创建 `core/__init__.py`：

```python
# core/__init__.py
```

## 3.5 运行方式

创建一个测试文件来验证：

```python
# tests/test_parser.py
"""测试文档解析"""

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
```

运行：

```bash
python -m tests.test_parser
```

## 3.6 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'fitz'` | PyMuPDF 没装好 | `pip install PyMuPDF` |
| `UnicodeDecodeError` | 文件编码不是 UTF-8 | 打开文件时加 `errors="ignore"` 或尝试 `encoding="gbk"` |
| PDF 提取出乱码 | 扫描件 PDF（图片不是文字）| 需要 OCR，超出基础版范围 |

## 3.7 验收标准

- [ ] 能解析 TXT 文件，提取纯文本
- [ ] 能解析 PDF 文件，提取纯文本
- [ ] 能解析 Markdown 文件，去掉标记，得到纯文本
- [ ] `parse_file()` 能自动识别文件类型
- [ ] 不支持的文件类型会抛出清晰的错误信息

---

# 第 4 章 · 文本切分 Chunking

## 4.1 目标

实现 `core/chunker.py`，把长文本智能地切成 300-500 字的小段。

## 4.2 原理

### 为什么要切分？

假设你有一篇 10 页的论文。用户问"什么是梯度下降？"，答案可能只在第 3 页的一个段落里。如果把整篇论文塞给 LLM：
- 可能超出 LLM 的输入长度限制
- LLM 会被大量无关内容干扰
- 向量检索对短文本效果更好

### 切分策略

最简单的方法是"每 500 字切一刀"，但这样可能把一句话从中间切断。更好的方法是**递归切分**：

1. 先按"双换行"（段落边界）切
2. 如果某段还是太长，按"单换行"切
3. 还是太长，按"句号"切
4. 还是太长，按"空格"切
5. 最后实在不行，按字符切

这就是"递归字符切分器"（Recursive Character Text Splitter）的核心思路。

### 重叠（Overlap）

切分时，相邻的 chunk 之间应该有一些重叠：

```
原文:  AAAA BBBB CCCC DDDD EEEE FFFF
Chunk1: AAAA BBBB CCCC
Chunk2:           CCCC DDDD EEEE     ← CCCC 重叠了
Chunk3:                     EEEE FFFF
```

为什么？因为如果一个概念恰好在两个 chunk 的交界处，重叠可以保证它至少完整出现在一个 chunk 里。

## 4.3 代码

```python
# core/chunker.py
"""文本切分模块：递归字符切分器"""

import re
from typing import Optional
from models.document import ParsedDocument, Chunk


class RecursiveChunker:
    """
    递归字符切分器。
    
    按照优先级递归地切分文本：
    1. 双换行（段落）
    2. 单换行
    3. 句号/问号/叹号
    4. 逗号/分号
    5. 空格
    6. 任意位置
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ):
        """
        Args:
            chunk_size: 每个 chunk 的目标大小（字符数）
            chunk_overlap: 相邻 chunk 的重叠大小
            separators: 切分符列表，按优先级排列
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",     # 段落边界（最优先）
            "\n",       # 换行
            "。",       # 中文句号
            "！",       # 中文叹号
            "？",       # 中文问号
            ". ",       # 英文句号
            "! ",       # 英文叹号
            "? ",       # 英文问号
            "；",       # 中文分号
            "，",       # 中文逗号
            ", ",       # 英文逗号
            " ",        # 空格
            "",         # 任意位置（最后兜底）
        ]
    
    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """
        递归切分文本。
        
        核心逻辑：
        1. 找到当前可用的最高优先级分隔符
        2. 用它切分文本
        3. 合并小片段，使每个片段不超过 chunk_size
        4. 如果某个片段还是太大，用下一级分隔符继续切
        """
        # 最终结果
        final_chunks: list[str] = []
        
        # 找到当前文本中存在的第一个分隔符
        separator = separators[-1]  # 默认用最后一个（空字符串）
        remaining_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_separators = []
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break
        
        # 用分隔符切分
        if separator:
            splits = text.split(separator)
        else:
            # 空字符串分隔 = 按单个字符切
            splits = list(text)
        
        # 合并小片段
        current_chunk = ""
        for split in splits:
            piece = split if not separator else split
            piece_with_sep = piece + separator if separator else piece
            
            if len(current_chunk) + len(piece_with_sep) <= self.chunk_size:
                current_chunk += piece_with_sep
            else:
                # 当前 chunk 满了
                if current_chunk:
                    final_chunks.append(current_chunk.rstrip(separator))
                
                # 如果这个片段本身就超过 chunk_size，递归切分
                if len(piece_with_sep) > self.chunk_size and remaining_separators:
                    sub_chunks = self._split_text(piece, remaining_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece_with_sep
        
        # 别忘了最后一个 chunk
        if current_chunk:
            final_chunks.append(current_chunk.rstrip(separator))
        
        return final_chunks
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """给相邻 chunk 添加重叠"""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            # 从上一个 chunk 的末尾取 overlap 长度的文本
            overlap_text = prev[-self.chunk_overlap:]
            # 如果 overlap 不是从句子开头开始，尝试找到最近的句子开头
            # 简单处理：找到 overlap 中第一个换行或句号后的位置
            for sep in ["\n", "。", ". "]:
                idx = overlap_text.find(sep)
                if idx != -1:
                    overlap_text = overlap_text[idx + len(sep):]
                    break
            
            result.append(overlap_text + curr)
        
        return result
    
    def chunk_text(self, text: str) -> list[str]:
        """
        切分文本为 chunk 列表。
        
        Args:
            text: 要切分的原始文本
            
        Returns:
            list[str]: chunk 文本列表
        """
        if not text or not text.strip():
            return []
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = self._split_text(text, self.separators)
        
        # 过滤空 chunk
        chunks = [c.strip() for c in chunks if c.strip()]
        
        # 添加重叠
        chunks = self._add_overlap(chunks)
        
        return chunks

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        切分一个已解析的文档，返回 Chunk 对象列表。
        
        Args:
            doc: 解析后的文档
            
        Returns:
            list[Chunk]: Chunk 对象列表
        """
        texts = self.chunk_text(doc.content)
        
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                source_id=doc.source_id,
                content=text,
                index=i,
                metadata={
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                },
            )
            chunks.append(chunk)
        
        return chunks
```

## 4.4 运行方式

```python
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
```

运行：

```bash
python -m tests.test_chunker
```

## 4.5 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| chunk 太小，丢失上下文 | chunk_size 设太小 | 建议 300-500 |
| chunk 太大，检索不精确 | chunk_size 设太大 | 建议不超过 1000 |
| 某些 chunk 只有一两个字 | 切分逻辑有边界情况 | 过滤 `len(chunk) < 20` 的 chunk |

## 4.6 验收标准

- [ ] 能把长文本切成指定大小的 chunk
- [ ] 切分时优先在段落/句子边界处切
- [ ] 相邻 chunk 有适当重叠
- [ ] 每个 Chunk 对象包含正确的 source_id 和 metadata

---

# 第 5 章 · Embedding 与向量数据库

## 5.1 目标

实现 `core/embedder.py` 和 `core/vector_store.py`，把 chunk 变成向量，存进 FAISS，支持向量检索。

## 5.2 原理

### Embedding 模型

我们用 **sentence-transformers** 库加载预训练的 Embedding 模型。模型输入一段文本，输出一个固定长度的向量（比如 384 维或 768 维）。

推荐模型：
- `all-MiniLM-L6-v2`：英文，384 维，速度快
- `paraphrase-multilingual-MiniLM-L12-v2`：多语言（含中文），384 维
- `BAAI/bge-small-zh-v1.5`：中文专用，512 维

### FAISS

FAISS（Facebook AI Similarity Search）是一个高效的向量检索库。我们用它的最简单模式 `IndexFlatIP`（内积）或 `IndexFlatL2`（欧氏距离）来存储和搜索向量。

## 5.3 代码：Embedding

```python
# core/embedder.py
"""Embedding 模块：把文本变成向量"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional

# 模型缓存，避免重复加载
_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> SentenceTransformer:
    """
    获取 Embedding 模型（有缓存）。
    
    首次调用会下载模型（约 100-500MB），之后会从本地缓存加载。
    
    Args:
        model_name: 模型名称
        
    Returns:
        SentenceTransformer 模型实例
    """
    if model_name not in _model_cache:
        print(f"正在加载 Embedding 模型: {model_name} ...")
        _model_cache[model_name] = SentenceTransformer(model_name)
        print(f"模型加载完成。向量维度: {_model_cache[model_name].get_sentence_embedding_dimension()}")
    return _model_cache[model_name]


def embed_texts(
    texts: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    把一组文本转成向量。
    
    Args:
        texts: 文本列表
        model_name: Embedding 模型名称
        batch_size: 批处理大小
        show_progress: 是否显示进度条
        
    Returns:
        np.ndarray: 形状为 (len(texts), dim) 的向量数组
    """
    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 归一化，方便用内积算相似度
    )
    return embeddings


def embed_query(
    query: str,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> np.ndarray:
    """
    把一个查询文本转成向量。
    
    Args:
        query: 查询文本
        model_name: Embedding 模型名称
        
    Returns:
        np.ndarray: 一维向量
    """
    model = get_model(model_name)
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding
```

## 5.4 代码：向量数据库

```python
# core/vector_store.py
"""向量数据库模块：基于 FAISS 的向量存储和检索"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

from models.document import Chunk, RetrievedChunk
from core.embedder import embed_texts, embed_query


class VectorStore:
    """
    基于 FAISS 的向量存储。
    
    功能：
    - 添加 chunk 向量
    - 按向量相似度搜索
    - 保存 / 加载索引到磁盘
    """
    
    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: 向量维度（必须和 Embedding 模型匹配）
        """
        self.dimension = dimension
        # 使用内积（Inner Product）索引
        # 因为我们的向量已经归一化，内积 = 余弦相似度
        self.index = faiss.IndexFlatIP(dimension)
        # 存储 chunk 信息，索引位置和 chunk 一一对应
        self.chunks: list[Chunk] = []
    
    def add_chunks(
        self,
        chunks: list[Chunk],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """
        添加一批 chunk 到向量库。
        
        Args:
            chunks: Chunk 对象列表
            model_name: Embedding 模型名称
        """
        if not chunks:
            return
        
        # 提取文本
        texts = [c.content for c in chunks]
        
        # 生成 Embedding
        embeddings = embed_texts(texts, model_name=model_name)
        
        # 添加到 FAISS 索引
        self.index.add(embeddings.astype(np.float32))
        
        # 保存 chunk 信息
        self.chunks.extend(chunks)
        
        print(f"已添加 {len(chunks)} 个 chunk，总计 {self.index.ntotal} 个向量")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> list[RetrievedChunk]:
        """
        向量相似度搜索。
        
        Args:
            query: 查询文本
            top_k: 返回前 K 个结果
            model_name: Embedding 模型名称
            
        Returns:
            list[RetrievedChunk]: 检索结果
        """
        if self.index.ntotal == 0:
            return []
        
        # 查询文本 → 向量
        query_vec = embed_query(query, model_name=model_name)
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        
        # FAISS 搜索
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, actual_k)
        
        # 构造结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS 返回 -1 表示没有更多结果
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source="vector",
                )
            )
        
        return results
    
    def save(self, dir_path: str | Path) -> None:
        """
        保存索引到磁盘。
        
        Args:
            dir_path: 保存目录
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 FAISS 索引
        faiss.write_index(self.index, str(dir_path / "faiss.index"))
        
        # 保存 chunk 信息（序列化为 JSON）
        chunks_data = [c.model_dump() for c in self.chunks]
        (dir_path / "chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"向量库已保存到: {dir_path}")
    
    def load(self, dir_path: str | Path) -> None:
        """
        从磁盘加载索引。
        
        Args:
            dir_path: 保存目录
        """
        dir_path = Path(dir_path)
        
        index_path = dir_path / "faiss.index"
        chunks_path = dir_path / "chunks.json"
        
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"索引文件不完整: {dir_path}")
        
        # 加载 FAISS 索引
        self.index = faiss.read_index(str(index_path))
        self.dimension = self.index.d
        
        # 加载 chunk 信息
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**d) for d in chunks_data]
        
        print(f"向量库已加载，共 {self.index.ntotal} 个向量")
    
    @property
    def size(self) -> int:
        """当前存储的向量数量"""
        return self.index.ntotal
```

## 5.5 运行方式

```python
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
        print(f"  {i+1}. [分数={r.score:.4f}] {r.chunk.content}")
    
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
```

运行：

```bash
python -m tests.test_vector_store
```

## 5.6 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| 第一次运行很慢 | 在下载 Embedding 模型 | 耐心等待，模型会缓存到 `~/.cache/` |
| 维度不匹配错误 | VectorStore 的 dimension 和模型输出维度不一致 | 检查模型的维度，`all-MiniLM-L6-v2` 是 384 维，`bge-small-zh` 是 512 维 |
| 内存不足 | chunk 太多或模型太大 | 减少 batch_size，或使用更小的模型 |

## 5.7 验收标准

- [ ] 能把文本转成向量
- [ ] 能把 chunk 向量存入 FAISS
- [ ] 能根据查询文本搜索最相似的 chunk
- [ ] 能把索引保存到磁盘并重新加载
- [ ] 检索结果语义上合理

---

# 第 6 章 · BM25 关键词检索

## 6.1 目标

实现 `core/bm25_store.py`，支持基于关键词的 BM25 检索。

## 6.2 原理

向量检索擅长"语义匹配"（意思相近就能找到），但对**精确关键词**不太敏感。比如用户搜"ResNet-50"，向量检索可能返回所有关于"图像识别"的内容，但 BM25 能精确找到提到"ResNet-50"的段落。

BM25 的计算公式核心思想：
- 一个词在文档中出现越多次 → 分数越高
- 但如果这个词在所有文档中都很常见（比如"的"、"是"）→ 权重降低
- 文档越短，词频的权重越高（避免长文档因为词多而占便宜）

我们用 `rank-bm25` 库来实现。

### 中文分词

BM25 需要把文本拆成"词"。英文自然用空格分隔，但中文没有空格。所以中文需要**分词**，我们用 `jieba`。

```
"机器学习是人工智能的分支" → ["机器学习", "是", "人工智能", "的", "分支"]
```

## 6.3 代码

```python
# core/bm25_store.py
"""BM25 关键词检索模块"""

import re
import json
import pickle
from pathlib import Path
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

from models.document import Chunk, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    """
    对文本进行分词。
    
    处理策略：
    - 中文：用 jieba 分词
    - 英文：按空格分割，转小写
    - 过滤掉单字符和停用词
    """
    # jieba 分词（同时处理中英文）
    words = jieba.lcut(text)
    
    # 简单的停用词表（实际项目可以加载更完整的）
    stop_words = {
        "的", "了", "是", "在", "和", "有", "就", "不", "也", "都", 
        "这", "那", "他", "她", "它", "我", "你", "们", "把", "被",
        "与", "为", "等", "及", "或", "但", "而", "从", "到", "对",
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "this", "that", "it", "as",
    }
    
    # 过滤：去掉停用词、单字符、纯标点
    tokens = []
    for w in words:
        w = w.strip().lower()
        if len(w) < 2:
            continue
        if w in stop_words:
            continue
        if re.match(r'^[\W]+$', w):  # 纯标点符号
            continue
        tokens.append(w)
    
    return tokens


class BM25Store:
    """
    BM25 关键词检索。
    
    功能：
    - 添加 chunk，建立 BM25 索引
    - 按关键词搜索
    - 保存 / 加载索引到磁盘
    """
    
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None
    
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        添加一批 chunk 到 BM25 索引。
        
        Args:
            chunks: Chunk 对象列表
        """
        if not chunks:
            return
        
        self.chunks.extend(chunks)
        
        # 对所有 chunk 文本分词
        new_tokenized = [_tokenize(c.content) for c in chunks]
        self.tokenized_corpus.extend(new_tokenized)
        
        # 重建 BM25 索引（rank-bm25 不支持增量添加，需要全量重建）
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"BM25 索引已更新，共 {len(self.chunks)} 个文档")
    
    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """
        BM25 关键词搜索。
        
        Args:
            query: 查询文本
            top_k: 返回前 K 个结果
            
        Returns:
            list[RetrievedChunk]: 检索结果（按 BM25 分数降序）
        """
        if self.bm25 is None or len(self.chunks) == 0:
            return []
        
        # 对查询分词
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        
        # BM25 打分
        scores = self.bm25.get_scores(query_tokens)
        
        # 取 top-K
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:  # 忽略零分
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source="bm25",
                )
            )
        
        return results
    
    def save(self, dir_path: str | Path) -> None:
        """保存到磁盘"""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 chunks
        chunks_data = [c.model_dump() for c in self.chunks]
        (dir_path / "bm25_chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        # 保存分词结果
        with open(dir_path / "tokenized_corpus.pkl", "wb") as f:
            pickle.dump(self.tokenized_corpus, f)
        
        print(f"BM25 索引已保存到: {dir_path}")
    
    def load(self, dir_path: str | Path) -> None:
        """从磁盘加载"""
        dir_path = Path(dir_path)
        
        # 加载 chunks
        chunks_data = json.loads(
            (dir_path / "bm25_chunks.json").read_text(encoding="utf-8")
        )
        self.chunks = [Chunk(**d) for d in chunks_data]
        
        # 加载分词结果
        with open(dir_path / "tokenized_corpus.pkl", "rb") as f:
            self.tokenized_corpus = pickle.load(f)
        
        # 重建 BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"BM25 索引已加载，共 {len(self.chunks)} 个文档")
    
    @property
    def size(self) -> int:
        return len(self.chunks)
```

## 6.4 运行方式

```python
# tests/test_bm25.py
"""测试 BM25 检索"""

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
        print(f"  {i+1}. [BM25={r.score:.4f}] {r.chunk.content}")
    
    # ResNet-50 应该排名第一
    assert len(results) > 0
    assert "ResNet" in results[0].chunk.content
    
    print("\n✓ BM25 测试通过！")


if __name__ == "__main__":
    test_bm25()
```

运行：

```bash
python -m tests.test_bm25
```

## 6.5 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| jieba 第一次加载慢 | 在初始化词典 | 正常现象，后续调用会快 |
| 中文搜索效果差 | 分词不准 | 检查分词结果，可以添加自定义词典 `jieba.add_word("ResNet")` |
| 全是零分 | 查询词都不在文档中 | 检查分词后的 token，确认有匹配 |

## 6.6 验收标准

- [ ] 能建立 BM25 索引
- [ ] 精确关键词搜索能找到正确结果
- [ ] 结果按 BM25 分数降序排列
- [ ] 能保存和加载索引

---

# 第 7 章 · 混合检索与 RRF 融合

## 7.1 目标

实现 `core/hybrid_retriever.py`，把向量检索和 BM25 检索的结果融合，得到更好的排序。

## 7.2 原理

**为什么要融合？**

| 场景 | 向量检索 | BM25 | 胜者 |
|------|---------|------|------|
| 搜"深度学习的基本原理" | 能找到语义相关的段落 | 能找到包含这些词的段落 | 两者互补 |
| 搜"ResNet-50 准确率" | 可能返回所有关于CNN的内容 | 精确找到提到"ResNet-50"的内容 | BM25 |
| 搜"如何让模型更聪明" | 理解"聪明"≈"性能好" | 找不到"聪明"这个词 | 向量检索 |

**RRF 公式详解**：

```
RRF_score(doc) = Σ  1 / (k + rank_i)
                i=1..n

k = 60（常数，可调）
rank_i = doc 在第 i 个排名列表中的位置（从 1 开始）
         如果 doc 不在某个列表中，rank_i = +∞（贡献为 0）
```

举例：
- 文档 A：在向量检索中排第 1，BM25 中排第 3
  - RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
- 文档 B：在向量检索中排第 2，BM25 中不在列表
  - RRF = 1/(60+2) + 0 = 0.01613
- 文档 A 的 RRF 分数更高 ✓

## 7.3 代码

```python
# core/hybrid_retriever.py
"""混合检索模块：向量检索 + BM25 + RRF 融合"""

from typing import Optional
from collections import defaultdict

from models.document import Chunk, RetrievedChunk
from core.vector_store import VectorStore
from core.bm25_store import BM25Store


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    RRF（Reciprocal Rank Fusion）融合多个排名列表。
    
    Args:
        ranked_lists: 多个排名列表，每个列表是 RetrievedChunk 列表
        k: RRF 常数（默认 60）
        
    Returns:
        list[RetrievedChunk]: 融合后的排名列表（按 RRF 分数降序）
    """
    # 用 chunk_id 作为文档标识，累加 RRF 分数
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, Chunk] = {}
    
    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            chunk_map[cid] = result.chunk  # 保存 chunk 对象
    
    # 按 RRF 分数降序排列
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 构造结果
    results = []
    for cid, score in sorted_items:
        results.append(
            RetrievedChunk(
                chunk=chunk_map[cid],
                score=score,
                source="rrf",
            )
        )
    
    return results


class HybridRetriever:
    """
    混合检索器。
    
    同时使用向量检索和 BM25，用 RRF 融合结果。
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        """
        Args:
            vector_store: 向量数据库
            bm25_store: BM25 索引
            vector_weight: 向量检索权重（暂时保留，用于未来加权）
            bm25_weight: BM25 权重
            rrf_k: RRF 常数
        """
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> list[RetrievedChunk]:
        """
        混合检索。
        
        流程：
        1. 向量检索获取 top-vector_top_k
        2. BM25 检索获取 top-bm25_top_k
        3. RRF 融合
        4. 返回 top-top_k
        
        Args:
            query: 查询文本
            top_k: 最终返回的结果数
            vector_top_k: 向量检索的候选数量
            bm25_top_k: BM25 检索的候选数量
            model_name: Embedding 模型名称
            
        Returns:
            list[RetrievedChunk]: 融合后的检索结果
        """
        # 1. 向量检索
        vector_results = self.vector_store.search(
            query, top_k=vector_top_k, model_name=model_name
        )
        
        # 2. BM25 检索
        bm25_results = self.bm25_store.search(query, top_k=bm25_top_k)
        
        # 3. RRF 融合
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.rrf_k,
        )
        
        # 4. 取 top-K
        return fused[:top_k]
    
    def search_vector_only(
        self,
        query: str,
        top_k: int = 5,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> list[RetrievedChunk]:
        """仅向量检索（对比用）"""
        return self.vector_store.search(query, top_k=top_k, model_name=model_name)
    
    def search_bm25_only(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """仅 BM25 检索（对比用）"""
        return self.bm25_store.search(query, top_k=top_k)
```

## 7.4 运行方式

```python
# tests/test_retriever.py
"""测试混合检索"""

from models.document import Chunk
from core.vector_store import VectorStore
from core.bm25_store import BM25Store
from core.hybrid_retriever import HybridRetriever


def test_hybrid_retriever():
    # 准备 chunks
    chunks = [
        Chunk(source_id="doc1", content="机器学习是人工智能的一个重要分支", index=0),
        Chunk(source_id="doc1", content="深度学习使用多层神经网络来处理数据", index=1),
        Chunk(source_id="doc1", content="ResNet-50 是一种经典的卷积神经网络架构，在 ImageNet 上达到了很高的准确率", index=2),
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
        print(f"  {i+1}. [RRF={r.score:.5f}] {r.chunk.content[:50]}")
    
    # 测试 2: 精确关键词搜索
    print()
    query2 = "ResNet-50 ImageNet"
    print(f"查询 2（关键词）: {query2}")
    results = retriever.search(query2, top_k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. [RRF={r.score:.5f}] {r.chunk.content[:50]}")
    
    # 对比三种检索方式
    print("\n" + "=" * 60)
    query3 = "Transformer 架构"
    print(f"\n对比查询: {query3}")
    
    vec_results = retriever.search_vector_only(query3, top_k=3)
    bm25_results = retriever.search_bm25_only(query3, top_k=3)
    hybrid_results = retriever.search(query3, top_k=3)
    
    print("\n  向量检索 Top-3:")
    for i, r in enumerate(vec_results):
        print(f"    {i+1}. {r.chunk.content[:50]}")
    
    print("\n  BM25 Top-3:")
    for i, r in enumerate(bm25_results):
        print(f"    {i+1}. {r.chunk.content[:50]}")
    
    print("\n  混合检索 (RRF) Top-3:")
    for i, r in enumerate(hybrid_results):
        print(f"    {i+1}. {r.chunk.content[:50]}")
    
    print("\n✓ 混合检索测试通过！")


if __name__ == "__main__":
    test_hybrid_retriever()
```

运行：

```bash
python -m tests.test_retriever
```

## 7.5 验收标准

- [ ] 向量检索和 BM25 检索各自返回结果
- [ ] RRF 融合后的结果合理（综合考虑语义和关键词）
- [ ] 精确关键词查询能通过 BM25 得到增强
- [ ] 语义查询能通过向量检索得到增强

---

# 第 8 章 · 对接 LLM —— Ollama 与 API

## 8.1 目标

实现 `core/llm_client.py`，统一封装多种 LLM 调用方式（本地 Ollama + 云端 API）。

## 8.2 原理

LLM 负责 RAG 的"G"（Generation）——根据检索到的资料生成回答。我们提供两种方案：

1. **Ollama（本地）**：免费、隐私，但需要一定硬件配置
2. **OpenAI / Claude / Gemini API（云端）**：效果好，但需要 API Key 和付费

我们的代码会抽象出统一接口，后端可以随时切换。

## 8.3 代码

```python
# core/llm_client.py
"""LLM 客户端：统一封装 Ollama 和各种 API"""

import json
from typing import Optional, Generator
from abc import ABC, abstractmethod

import httpx


class LLMClient(ABC):
    """LLM 客户端基类"""
    
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        发送对话请求，返回完整回复。
        
        Args:
            messages: 对话消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数（0-1，越高越随机）
            max_tokens: 最大生成 token 数
            
        Returns:
            str: 模型的回复文本
        """
        pass
    
    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        流式对话，逐步返回文本片段。
        
        Yields:
            str: 文本片段
        """
        pass


# ============================================================
# Ollama 客户端
# ============================================================
class OllamaClient(LLMClient):
    """
    Ollama 本地 LLM 客户端。
    
    Ollama 运行在 http://localhost:11434，我们通过 HTTP API 调用。
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)  # LLM 生成可能比较慢
    
    def _check_available(self) -> bool:
        """检查 Ollama 是否可用"""
        try:
            r = self.client.get(f"{self.base_url}/api/tags")
            return r.status_code == 200
        except Exception:
            return False
    
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        r = self.client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]
    
    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]


# ============================================================
# OpenAI 兼容客户端（支持 OpenAI / Claude / Gemini 等）
# ============================================================
class OpenAICompatibleClient(LLMClient):
    """
    OpenAI API 兼容客户端。
    
    很多 LLM 提供商都兼容 OpenAI 的 API 格式，所以这个客户端可以用于：
    - OpenAI (gpt-4o, gpt-4o-mini)
    - Anthropic Claude (通过兼容 endpoint)
    - Google Gemini (通过兼容 endpoint)
    - 其他兼容 OpenAI API 的服务
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(
            timeout=120.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
    
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        r = self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    
    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue


# ============================================================
# 工厂函数：根据配置创建客户端
# ============================================================
def create_llm_client(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """
    根据配置创建 LLM 客户端。
    
    Args:
        provider: "ollama" / "openai" / "claude" / "gemini"
        model: 模型名称
        api_key: API Key（云端方案需要）
        base_url: API 地址
        
    Returns:
        LLMClient: LLM 客户端实例
        
    Examples:
        # 本地 Ollama
        client = create_llm_client("ollama", model="qwen2.5:7b")
        
        # OpenAI
        client = create_llm_client("openai", api_key="sk-xxx", model="gpt-4o-mini")
        
        # Claude (通过 OpenAI 兼容 API)
        client = create_llm_client(
            "claude",
            api_key="sk-ant-xxx",
            model="claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com/v1"
        )
    """
    if provider == "ollama":
        return OllamaClient(
            model=model or "qwen2.5:7b",
            base_url=base_url or "http://localhost:11434",
        )
    elif provider in ("openai", "claude", "gemini"):
        if not api_key:
            raise ValueError(f"{provider} 需要提供 api_key")
        
        defaults = {
            "openai": ("gpt-4o-mini", "https://api.openai.com/v1"),
            "claude": ("claude-sonnet-4-20250514", "https://api.anthropic.com/v1"),
            "gemini": ("gemini-1.5-flash", "https://generativelanguage.googleapis.com/v1beta/openai"),
        }
        default_model, default_url = defaults[provider]
        
        return OpenAICompatibleClient(
            api_key=api_key,
            model=model or default_model,
            base_url=base_url or default_url,
        )
    else:
        raise ValueError(f"不支持的 provider: {provider}")
```

## 8.4 运行方式

```python
# 测试 Ollama（确保 Ollama 已启动）
from core.llm_client import create_llm_client

client = create_llm_client("ollama", model="qwen2.5:7b")
reply = client.chat([{"role": "user", "content": "你好，用一句话介绍什么是机器学习"}])
print(reply)
```

## 8.5 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `Connection refused` | Ollama 没启动 | 启动 Ollama 应用或运行 `ollama serve` |
| `model not found` | 模型没下载 | `ollama pull qwen2.5:7b` |
| 超时 | 模型太大，生成太慢 | 增加 timeout 或换小模型 |
| API 401 | API Key 错误 | 检查 Key 是否正确 |

## 8.6 验收标准

- [ ] Ollama 客户端能正常调用本地模型
- [ ] 流式输出正常工作
- [ ] API 客户端能正常调用（如果有 Key）
- [ ] `create_llm_client` 工厂函数能正确创建不同客户端

---

# 第 9 章 · RAG Pipeline 完整串联

## 9.1 目标

实现 `core/rag_pipeline.py`，把检索、Prompt 组装、LLM 调用、Citation 生成串成完整流水线。

## 9.2 原理

RAG Pipeline 的完整流程：

```
用户问题
   ↓
混合检索 → 得到 top-K 个相关 chunk
   ↓
组装 Prompt：
  - 系统指令（你是一个基于资料回答问题的助手...）
  - 检索到的 chunk（每个 chunk 标上编号 [1] [2] ...）
  - 用户的问题
   ↓
调用 LLM → 生成回答（要求引用 [1] [2] 等标记）
   ↓
解析 Citation → 把 [1] [2] 映射回具体的文件和段落
   ↓
返回 {answer, citations}
```

## 9.3 配置文件

先创建全局配置：

```python
# config.py
"""全局配置"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """应用配置"""
    
    # ---- 路径 ----
    data_dir: Path = Path("data")
    notebooks_dir: Path = Path("data/notebooks")
    uploads_dir: Path = Path("data/uploads")
    db_path: Path = Path("data/notebook_demo.db")
    
    # ---- Embedding ----
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    
    # ---- Chunking ----
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # ---- 检索 ----
    vector_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    
    # ---- LLM ----
    llm_provider: str = "ollama"          # "ollama" / "openai" / "claude" / "gemini"
    llm_model: str = "qwen2.5:7b"        # 模型名称
    llm_api_key: str = ""                 # API Key（Ollama 不需要）
    llm_base_url: str = ""                # API 地址（Ollama 不需要）
    llm_temperature: float = 0.3          # 低温度 = 更确定的回答
    llm_max_tokens: int = 2048
    
    # ---- Reranker（可选）----
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5
    
    def __post_init__(self):
        """确保目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = AppConfig()
```

## 9.4 Reranker（可选）

```python
# core/reranker.py
"""Reranker 重排序模块（可选）"""

from typing import Optional

from models.document import RetrievedChunk


class CrossEncoderReranker:
    """
    基于 Cross-Encoder 的重排序器。
    
    Cross-Encoder 会把 (query, document) 对作为输入，
    输出一个相关性分数。比 Embedding + 余弦相似度更准确，
    但也更慢（不能预计算）。
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"正在加载 Reranker 模型: {self.model_name} ...")
            self._model = CrossEncoder(self.model_name)
            print("Reranker 模型加载完成。")
    
    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        对检索结果重排序。
        
        Args:
            query: 查询文本
            results: 初步检索结果
            top_k: 重排序后返回前 K 个
            
        Returns:
            list[RetrievedChunk]: 重排序后的结果
        """
        if not results:
            return []
        
        self._load_model()
        
        # 构建 (query, doc) 对
        pairs = [(query, r.chunk.content) for r in results]
        
        # Cross-Encoder 打分
        scores = self._model.predict(pairs)
        
        # 按分数排序
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 更新分数并返回 top-K
        reranked = []
        for score, result in scored_results[:top_k]:
            reranked.append(
                RetrievedChunk(
                    chunk=result.chunk,
                    score=float(score),
                    source="reranker",
                )
            )
        
        return reranked
```

## 9.5 RAG Pipeline 代码

```python
# core/rag_pipeline.py
"""RAG Pipeline：完整的检索增强生成流水线"""

import re
from typing import Optional, Generator
from dataclasses import dataclass, field

from models.document import Chunk, RetrievedChunk
from core.hybrid_retriever import HybridRetriever
from core.reranker import CrossEncoderReranker
from core.llm_client import LLMClient


@dataclass
class Citation:
    """引用信息"""
    index: int            # 引用编号 [1], [2], ...
    chunk_id: str
    source_id: str
    filename: str
    content_preview: str  # chunk 内容的前 100 字
    

@dataclass
class RAGResponse:
    """RAG 回答结果"""
    answer: str                           # LLM 的回答（包含 [1] [2] 等标记）
    citations: list[Citation]             # 引用列表
    retrieved_chunks: list[RetrievedChunk]  # 检索到的 chunk


# ============================================================
# Prompt 模板
# ============================================================
SYSTEM_PROMPT = """你是一个专业的知识助手。你的任务是根据提供的参考资料来回答用户的问题。

规则：
1. 只使用提供的参考资料来回答，不要编造信息。
2. 如果参考资料中没有相关信息，请明确说"根据提供的资料，我无法找到相关信息"。
3. 在回答中使用引用标记 [1] [2] 等来标注信息来源。
4. 引用标记对应参考资料的编号。
5. 回答要准确、有条理。
6. 如果有多个资料涉及同一话题，请综合它们的内容。"""

CONTEXT_TEMPLATE = """参考资料：

{context}

---

用户问题：{question}

请基于以上参考资料回答问题，并使用 [编号] 标注引用来源。"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """把检索到的 chunk 组装成上下文文本"""
    parts = []
    for i, rc in enumerate(chunks, start=1):
        filename = rc.chunk.metadata.get("filename", "未知文件")
        parts.append(f"[{i}] （来源：{filename}）\n{rc.chunk.content}")
    return "\n\n".join(parts)


def _parse_citations(
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """从回答中解析引用标记"""
    # 找到所有 [数字] 格式的引用
    cited_indices = set()
    for match in re.finditer(r'\[(\d+)\]', answer):
        idx = int(match.group(1))
        cited_indices.add(idx)
    
    citations = []
    for idx in sorted(cited_indices):
        if 1 <= idx <= len(chunks):
            rc = chunks[idx - 1]
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=rc.chunk.chunk_id,
                    source_id=rc.chunk.source_id,
                    filename=rc.chunk.metadata.get("filename", "未知"),
                    content_preview=rc.chunk.content[:100],
                )
            )
    
    return citations


# ============================================================
# RAG Pipeline
# ============================================================
class RAGPipeline:
    """
    完整的 RAG 流水线。
    
    流程：检索 → (可选)重排序 → 组装 Prompt → 调用 LLM → 解析引用
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.reranker = reranker
        self.top_k = top_k
    
    def query(
        self,
        question: str,
        temperature: float = 0.3,
    ) -> RAGResponse:
        """
        执行完整的 RAG 查询。
        
        Args:
            question: 用户的问题
            temperature: LLM 温度参数
            
        Returns:
            RAGResponse: 包含回答、引用、检索结果
        """
        # 1. 混合检索
        candidates = self.retriever.search(
            question,
            top_k=self.top_k * 4 if self.reranker else self.top_k,
        )
        
        # 2. 可选：重排序
        if self.reranker and candidates:
            candidates = self.reranker.rerank(
                question, candidates, top_k=self.top_k
            )
        else:
            candidates = candidates[:self.top_k]
        
        # 3. 组装 Prompt
        context = _build_context(candidates)
        user_message = CONTEXT_TEMPLATE.format(
            context=context,
            question=question,
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        # 4. 调用 LLM
        answer = self.llm_client.chat(
            messages,
            temperature=temperature,
        )
        
        # 5. 解析引用
        citations = _parse_citations(answer, candidates)
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=candidates,
        )
    
    def query_stream(
        self,
        question: str,
        temperature: float = 0.3,
    ) -> Generator[str | RAGResponse, None, None]:
        """
        流式 RAG 查询。
        
        先 yield 文本片段（str），最后 yield 完整的 RAGResponse。
        """
        # 1-2. 检索 + 重排序
        candidates = self.retriever.search(
            question,
            top_k=self.top_k * 4 if self.reranker else self.top_k,
        )
        
        if self.reranker and candidates:
            candidates = self.reranker.rerank(
                question, candidates, top_k=self.top_k
            )
        else:
            candidates = candidates[:self.top_k]
        
        # 3. 组装 Prompt
        context = _build_context(candidates)
        user_message = CONTEXT_TEMPLATE.format(
            context=context,
            question=question,
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        # 4. 流式调用 LLM
        full_answer = ""
        for chunk in self.llm_client.chat_stream(messages, temperature=temperature):
            full_answer += chunk
            yield chunk  # 流式输出文本片段
        
        # 5. 解析引用
        citations = _parse_citations(full_answer, candidates)
        
        # 最后 yield 完整结果
        yield RAGResponse(
            answer=full_answer,
            citations=citations,
            retrieved_chunks=candidates,
        )
```

## 9.6 内容生成器（FAQ / Study Guide / Briefing Doc）

```python
# core/generator.py
"""内容生成器：FAQ / Study Guide / Briefing Doc"""

from core.llm_client import LLMClient
from models.document import Chunk


FAQ_PROMPT = """你是一个知识助手。根据以下资料，生成一份 FAQ（常见问题解答）。

要求：
1. 生成 5-10 个最重要的问题和回答
2. 问题要具体、有价值
3. 回答要准确、基于资料
4. 格式：每个问题用 "Q: " 开头，回答用 "A: " 开头
5. 问答之间空一行

资料：
{content}

请生成 FAQ："""


STUDY_GUIDE_PROMPT = """你是一个知识助手。根据以下资料，生成一份学习指南（Study Guide）。

要求：
1. 总结核心概念和关键知识点
2. 按逻辑顺序组织
3. 包含关键术语及其定义
4. 适合复习和快速回顾
5. 使用清晰的层次结构

资料：
{content}

请生成学习指南："""


BRIEFING_DOC_PROMPT = """你是一个知识助手。根据以下资料，生成一份简报文档（Briefing Document）。

要求：
1. 开头写一段摘要（Executive Summary）
2. 然后分主题详细展开
3. 语气专业、客观
4. 包含关键发现和重要数据
5. 结尾给出结论或建议

资料：
{content}

请生成简报文档："""


AUDIO_SCRIPT_PROMPT = """你是一个播客脚本作者。根据以下资料，写一份两人对话的播客脚本。

要求：
1. 两位主持人：A（引导者）和 B（解释者）
2. 对话自然、口语化
3. A 负责提问和过渡，B 负责解释
4. 把核心内容用对话形式呈现
5. 开头有简短介绍，结尾有总结
6. 时长约 5-8 分钟的内容

资料：
{content}

请生成播客脚本："""


class ContentGenerator:
    """内容生成器"""
    
    TEMPLATES = {
        "faq": FAQ_PROMPT,
        "study_guide": STUDY_GUIDE_PROMPT,
        "briefing_doc": BRIEFING_DOC_PROMPT,
        "audio_script": AUDIO_SCRIPT_PROMPT,
    }
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def generate(
        self,
        chunks: list[Chunk],
        output_type: str = "faq",
        temperature: float = 0.5,
    ) -> str:
        """
        根据 chunk 内容生成指定类型的文档。
        
        Args:
            chunks: 相关的 chunk 列表
            output_type: "faq" / "study_guide" / "briefing_doc" / "audio_script"
            temperature: 温度参数
            
        Returns:
            str: 生成的文档内容
        """
        if output_type not in self.TEMPLATES:
            raise ValueError(f"不支持的类型: {output_type}。支持: {list(self.TEMPLATES.keys())}")
        
        # 把 chunk 内容合并
        content = "\n\n---\n\n".join(c.content for c in chunks)
        
        # 如果内容太长，截断（避免超出 LLM 输入限制）
        max_content_len = 8000  # 字符数
        if len(content) > max_content_len:
            content = content[:max_content_len] + "\n\n[... 内容已截断 ...]"
        
        prompt = self.TEMPLATES[output_type].format(content=content)
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        return self.llm_client.chat(messages, temperature=temperature)
```

## 9.7 运行方式

```python
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
```

运行：

```bash
python -m tests.test_rag
```

## 9.8 验收标准

- [ ] 完整的 RAG 流水线能正常运行
- [ ] 回答基于检索到的资料
- [ ] 回答包含 [1] [2] 等引用标记
- [ ] Citation 能正确解析并映射到原始 chunk
- [ ] 内容生成器能生成 FAQ / Study Guide / Briefing Doc / Audio Script

---

# 第 10 章 · Streamlit 前端

## 10.1 目标

实现 `app.py`，用 Streamlit 做一个交互式的 Web 前端。

## 10.2 原理

Streamlit 是一个 Python 库，可以用纯 Python 写出好看的 Web 界面，不需要学 HTML/CSS/JavaScript。

```python
import streamlit as st
st.title("Hello")
st.write("World")
# 运行：streamlit run app.py
# 浏览器打开 localhost:8501 就能看到页面
```

## 10.3 存储层

先实现一个简单的存储层，管理 notebook 和文档的元数据：

```python
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
```

创建 `storage/__init__.py`：

```python
# storage/__init__.py
```

## 10.4 Streamlit 前端代码

```python
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
    st.stop()

nb_meta = store.get_notebook(selected_nb_id)

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
```

## 10.5 运行方式

```bash
# 确保在项目根目录、虚拟环境已激活
streamlit run app.py

# 浏览器会自动打开 http://localhost:8501
```

## 10.6 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `streamlit: command not found` | 没在虚拟环境中 | 激活虚拟环境 |
| 页面白屏 | 代码有语法错误 | 看终端的报错信息 |
| 上传文件后没反应 | 忘了点"处理文档"按钮 | 上传后要点击按钮 |
| Embedding 加载慢 | 首次加载模型 | 第一次会慢，后续很快 |

## 10.7 验收标准

- [ ] 能创建 Notebook
- [ ] 能上传和处理 PDF / Markdown / TXT 文件
- [ ] 处理后能看到文件列表和 chunk 数量
- [ ] 能在对话框提问，得到基于资料的回答
- [ ] 回答带有 Citation 引用
- [ ] 能生成 FAQ / Study Guide / Briefing Doc / Audio Script
- [ ] 页面布局合理，交互流畅

---

# 第 11 章 · 进阶功能

## 11.1 Notebook 数据模型

```python
# models/notebook.py
"""Notebook 数据模型"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class NotebookMeta(BaseModel):
    """Notebook 元数据"""
    id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    sources: list[dict] = Field(default_factory=list)
    chunk_count: int = 0
    
    
class ChatMessage(BaseModel):
    """聊天消息"""
    role: str          # "user" / "assistant"
    content: str
    citations: list[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
```

## 11.2 使用 Cross-Encoder Reranker

在 `config.py` 中启用：

```python
config.use_reranker = True
config.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

首次运行会下载 Reranker 模型。Reranker 会让检索结果更准确，但每次查询会稍微慢一些（需要逐一打分）。

## 11.3 自定义分词词典

如果你的文档有很多专业术语，jieba 可能分不准。可以添加自定义词典：

```python
# 在 bm25_store.py 的 _tokenize 函数开头添加
import jieba

# 添加专业术语
jieba.add_word("Transformer")
jieba.add_word("ResNet")
jieba.add_word("BERT")
jieba.add_word("GPT")
jieba.add_word("梯度下降")
jieba.add_word("反向传播")

# 或者从文件加载
# jieba.load_userdict("custom_dict.txt")
# 文件格式：每行一个词，可选指定词频和词性
# Transformer 5 n
# ResNet 5 n
```

## 11.4 性能优化建议

1. **Embedding 缓存**：重启应用时不要重新计算 Embedding，从磁盘加载
2. **模型预加载**：应用启动时预加载 Embedding 模型和 LLM
3. **异步处理**：大文件处理可以放到后台
4. **增量索引**：新增文档时只处理新文档，不重建整个索引（BM25 需要重建，FAISS 支持增量添加）

---

# 第 12 章 · 进阶版：FastAPI 后端重构

## 12.1 目标

把后端逻辑从 Streamlit 分离出来，用 FastAPI 实现 REST API。

## 12.2 为什么要分离？

Streamlit 适合快速原型，但：
- 不适合多用户并发
- 前后端耦合，不好维护
- 不方便给其他客户端（移动端、其他 Web 框架）调用

## 12.3 额外依赖

```bash
pip install fastapi uvicorn python-multipart
```

## 12.4 API 代码

```python
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
```

## 12.5 运行 FastAPI

```bash
uvicorn api:app --reload --port 8000

# API 文档：http://localhost:8000/docs
```

## 12.6 API 测试

```bash
# 创建 Notebook
curl -X POST http://localhost:8000/api/notebooks \
  -H "Content-Type: application/json" \
  -d '{"name": "测试笔记本"}'

# 上传文件
curl -X POST http://localhost:8000/api/notebooks/{notebook_id}/upload \
  -F "file=@your_document.pdf"

# 提问
curl -X POST http://localhost:8000/api/notebooks/{notebook_id}/query \
  -H "Content-Type: application/json" \
  -d '{"question": "文档的主要内容是什么？"}'
```

---

# 第 13 章 · 部署、评估与简历

## 13.1 简单评估

```python
# eval/simple_eval.py
"""简单的 RAG 评估脚本"""

from core.vector_store import VectorStore
from core.bm25_store import BM25Store
from core.hybrid_retriever import HybridRetriever
from core.llm_client import create_llm_client
from core.rag_pipeline import RAGPipeline
from models.document import Chunk


def evaluate_retrieval(retriever, test_cases):
    """
    评估检索质量。
    
    test_cases 格式：
    [
        {"question": "...", "expected_source_ids": ["doc1"]},
        ...
    ]
    """
    total = len(test_cases)
    hits = 0
    
    for case in test_cases:
        results = retriever.search(case["question"], top_k=5)
        retrieved_ids = {r.chunk.source_id for r in results}
        expected_ids = set(case["expected_source_ids"])
        
        if retrieved_ids & expected_ids:  # 有交集就算命中
            hits += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {status} Q: {case['question'][:40]}...")
    
    recall = hits / total if total > 0 else 0
    print(f"\n  Recall@5: {recall:.2%} ({hits}/{total})")
    return recall
```

## 13.2 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建数据目录
RUN mkdir -p data/notebooks data/uploads

# 暴露端口
EXPOSE 8501

# 启动
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  notebook-demo:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - LLM_PROVIDER=ollama
      - OLLAMA_HOST=http://host.docker.internal:11434
```

## 13.3 项目写进简历

请见附录 C。

---

# 附录 A · 用 LangChain / LlamaIndex 重构

本手册主线全部手写，目的是帮你理解每一个组件。但在实际工作中，很多人会用框架来简化开发。下面展示如何用 LangChain 和 LlamaIndex 替代我们手写的组件。

## A.1 LangChain 版本

```python
# langchain_version.py
"""用 LangChain 重写核心组件（对比学习用）"""

# pip install langchain langchain-community langchain-ollama chromadb

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1. 文档加载
loader = PyMuPDFLoader("your_doc.pdf")
docs = loader.load()

# 2. 文本切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)

# 3. Embedding + 向量库
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. LLM
llm = Ollama(model="qwen2.5:7b")

# 5. RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)

# 使用
result = qa_chain({"query": "什么是梯度下降？"})
print(result["result"])
```

**对比分析**：

| 方面 | 手写 | LangChain |
|------|------|-----------|
| 理解深度 | 深入理解每个组件 | 黑盒调用 |
| 灵活性 | 完全可控 | 受框架约束 |
| 代码量 | 较多 | 很少 |
| 调试 | 容易定位问题 | 需要了解框架内部 |
| 适合场景 | 学习、定制需求 | 快速开发、标准需求 |

## A.2 LlamaIndex 版本

```python
# llamaindex_version.py
"""用 LlamaIndex 重写（对比学习用）"""

# pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 加载文档
documents = SimpleDirectoryReader("./your_docs/").load_data()

# 2. 配置模型
llm = Ollama(model="qwen2.5:7b")
embed_model = HuggingFaceEmbedding(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. 建索引 + 查询
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)
query_engine = index.as_query_engine(llm=llm)

# 使用
response = query_engine.query("什么是梯度下降？")
print(response)
```

---

# 附录 B · 常见报错速查表

| 错误信息 | 可能原因 | 解决方案 |
|----------|---------|---------|
| `ModuleNotFoundError: No module named 'xxx'` | 包没装 | `pip install xxx` |
| `No module named 'fitz'` | PyMuPDF 没装 | `pip install PyMuPDF` |
| `Connection refused (localhost:11434)` | Ollama 没启动 | 启动 Ollama 应用 |
| `model 'xxx' not found` | 模型没下载 | `ollama pull xxx` |
| `CUDA out of memory` | GPU 内存不足 | 用 CPU 版本或换小模型 |
| `UnicodeDecodeError` | 文件编码问题 | 指定 `encoding="utf-8"` 或 `"gbk"` |
| `ValueError: shapes not aligned` | 向量维度不匹配 | 检查模型维度和 VectorStore 维度一致 |
| `OSError: [Errno 28] No space left` | 磁盘空间不足 | 清理磁盘或换目录 |
| Streamlit `DuplicateWidgetID` | widget 的 key 重复了 | 给每个 widget 唯一的 key |
| `torch` 装不上 | Python 版本或系统不兼容 | 用 Python 3.11，参考 PyTorch 官网安装指南 |
| PDF 提取出空白 | 扫描件 PDF | 需要 OCR（如 `pytesseract`） |
| BM25 全是零分 | 查询词不在文档中 | 换一个查询词，或检查分词 |
| Reranker 加载慢 | 首次下载模型 | 耐心等待 |
| `httpx.TimeoutException` | LLM 响应太慢 | 增加 timeout 值 |

---

# 附录 C · 简历 Bullet Points

以下是这个项目写进英文简历的建议表述：

## 项目标题

**Notebook Demo — Local RAG-powered Knowledge Base Q&A System**

## Bullet Points（根据你的实现深度选择）

### 基础版

- Built a local RAG (Retrieval-Augmented Generation) knowledge base system supporting PDF, Markdown, and TXT document ingestion with automated parsing, chunking, and indexing
- Implemented hybrid search combining FAISS-based semantic vector retrieval with BM25 keyword retrieval, fused via Reciprocal Rank Fusion (RRF) for improved retrieval accuracy
- Designed a modular Python pipeline covering document parsing (PyMuPDF), recursive text chunking with configurable overlap, sentence-transformer embeddings, and multi-strategy retrieval
- Integrated Ollama for local LLM inference with support for OpenAI/Claude/Gemini API fallback, enabling citation-grounded answers with source attribution
- Developed a Streamlit-based interactive frontend supporting multi-notebook management, document upload, conversational Q&A with citations, and automated FAQ/Study Guide generation

### 进阶版（如果实现了 FastAPI + Reranker + Docker）

- Architected a full-stack RAG application with FastAPI REST backend and Streamlit frontend, containerized with Docker for reproducible deployment
- Enhanced retrieval precision by integrating a Cross-Encoder reranker (ms-marco-MiniLM) as a second-stage ranker on top of hybrid retrieval results
- Implemented persistent storage with SQLite for metadata and FAISS/BM25 index serialization, supporting incremental document updates across sessions
- Built an evaluation framework to measure retrieval recall and answer quality, achieving XX% Recall@5 on a custom test set

## 技术关键词

确保简历中出现这些关键词（ATS 友好）：

RAG, Retrieval-Augmented Generation, Vector Database, FAISS, BM25, Hybrid Search, RRF, Reciprocal Rank Fusion, Embedding, Sentence Transformers, Cross-Encoder, Reranker, LLM, Ollama, OpenAI API, Chunking, Document Parsing, PyMuPDF, Streamlit, FastAPI, Pydantic, SQLite, Docker, Python

---

# 总结

恭喜你完成了整个 Notebook Demo 项目！回顾一下你学到了什么：

1. **文档解析**：把 PDF/Markdown/TXT 变成纯文本
2. **文本切分**：智能地把长文本切成小段（Chunk）
3. **Embedding**：把文本变成向量
4. **向量数据库**：用 FAISS 存储和检索向量
5. **BM25**：关键词检索
6. **混合检索 + RRF**：融合语义检索和关键词检索
7. **Reranker**：精细重排序
8. **RAG Pipeline**：检索 + Prompt 组装 + LLM 生成 + Citation
9. **Streamlit 前端**：交互式 Web 界面
10. **FastAPI 后端**：REST API 服务
11. **内容生成**：FAQ / Study Guide / Briefing Doc / Audio Script

这不是一个"玩具项目"，而是一个涵盖了 RAG 系统核心环节的完整实现。把它写好、调好、放进简历，它能充分展示你对 RAG 系统的理解和工程能力。

继续前进，祝你学习愉快！
