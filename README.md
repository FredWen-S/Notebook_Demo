# Notebook Demo

A local RAG-powered knowledge base Q&A system inspired by Google NotebookLM. Upload your documents (PDF, Markdown, TXT), ask questions, and get grounded answers with source citations — all running locally on your machine.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Document Ingestion** — Upload PDF, Markdown, and TXT files with automatic parsing and chunking
- **Hybrid Search** — Combines FAISS vector retrieval (semantic) + BM25 keyword retrieval, fused via Reciprocal Rank Fusion (RRF)
- **Local LLM** — Runs entirely offline with Ollama (qwen2.5, llama3.1, etc.), with optional OpenAI / Claude / Gemini API support
- **Citation-grounded Answers** — Every answer includes `[1] [2]` references mapped back to source documents
- **Content Generation** — One-click generation of FAQ, Study Guide, Briefing Doc, and Podcast Script from your documents
- **Multi-Notebook** — Organize documents into separate notebooks, each with its own index
- **Optional Reranker** — Cross-encoder reranking for higher retrieval precision

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  Hybrid Retriever               │
│  FAISS (semantic) + BM25 (keyword) │
│  → RRF Fusion → Top-K          │
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│  (Optional) Cross-Encoder Reranker │
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│  RAG Pipeline                   │
│  Prompt Assembly + LLM Call     │
│  → Answer with Citations        │
└─────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Document Parsing | PyMuPDF, markdown |
| Chunking | Custom recursive character splitter |
| Embedding | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector Store | FAISS |
| Keyword Search | rank-bm25 |
| Fusion | Hand-written RRF |
| Reranker | cross-encoder (optional) |
| LLM | Ollama (local) / OpenAI / Claude / Gemini (API) |
| Data Models | Pydantic |
| Storage | JSON + FAISS index files |
| API (advanced) | FastAPI |

## Project Structure

```
notebook_demo/
├── app.py                  # Streamlit frontend
├── api.py                  # FastAPI backend (advanced)
├── config.py               # Global configuration
├── requirements.txt        # Python dependencies
│
├── core/                   # Core pipeline
│   ├── parser.py           # PDF / Markdown / TXT → plain text
│   ├── chunker.py          # Recursive text chunking with overlap
│   ├── embedder.py         # Text → vector embeddings
│   ├── vector_store.py     # FAISS vector storage & retrieval
│   ├── bm25_store.py       # BM25 keyword retrieval
│   ├── hybrid_retriever.py # Hybrid search + RRF fusion
│   ├── reranker.py         # Cross-encoder reranker (optional)
│   ├── llm_client.py       # Unified LLM client (Ollama / API)
│   ├── rag_pipeline.py     # Full RAG pipeline with citations
│   └── generator.py        # FAQ / Study Guide / Briefing Doc generation
│
├── models/                 # Pydantic data models
│   ├── document.py         # ParsedDocument, Chunk, RetrievedChunk
│   └── notebook.py         # NotebookMeta, ChatMessage
│
├── storage/                # Persistence
│   └── db.py               # Notebook metadata (JSON-based)
│
├── tests/                  # Tests
│   ├── test_parser.py
│   ├── test_chunker.py
│   ├── test_vector_store.py
│   ├── test_bm25.py
│   ├── test_retriever.py
│   └── test_rag.py
│
├── eval/                   # Evaluation
│   └── simple_eval.py
│
└── data/                   # Runtime data (auto-created)
    ├── notebooks/          # Per-notebook indexes & sources
    └── uploads/
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) (for local LLM)

### 1. Clone & set up environment

```bash
git clone <your-repo-url>
cd notebook_demo

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull a local model

```bash
# Recommended: good at both Chinese and English
ollama pull qwen2.5:7b

# Low-memory alternative (4GB RAM)
ollama pull qwen2.5:3b
```

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, create a notebook, upload documents, and start asking questions.

### (Optional) Run as API server

```bash
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload --port 8000
```

API docs available at http://localhost:8000/docs.

## Usage

1. **Create a Notebook** — Click "Create New Notebook" in the sidebar
2. **Upload Documents** — Drag & drop PDF / MD / TXT files, then click "Process Documents"
3. **Ask Questions** — Type your question in the chat input; answers will include `[1] [2]` citation markers
4. **Generate Content** — Select FAQ / Study Guide / Briefing Doc / Podcast Script and click "Generate"

## Configuration

Edit `config.py` or use the sidebar settings in the Streamlit UI:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_provider` | `"ollama"` | `"ollama"` / `"openai"` / `"claude"` / `"gemini"` |
| `llm_model` | `"qwen2.5:7b"` | Model name |
| `chunk_size` | `500` | Target chunk size (characters) |
| `chunk_overlap` | `50` | Overlap between adjacent chunks |
| `final_top_k` | `5` | Number of chunks sent to LLM |
| `use_reranker` | `False` | Enable cross-encoder reranking |
| `embedding_model` | `"paraphrase-multilingual-MiniLM-L12-v2"` | Sentence-transformers model |

## Using Cloud LLM APIs

If your machine can't run local models, switch to a cloud API in the sidebar or in `config.py`:

```python
config.llm_provider = "openai"     # or "claude" / "gemini"
config.llm_model = "gpt-4o-mini"
config.llm_api_key = "sk-..."
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `streamlit: command not found` | Activate your virtualenv first |
| LLM times out | Switch to a smaller model (`qwen2.5:3b`) or use a cloud API |
| `Connection refused` on port 11434 | Start Ollama (`ollama serve` or launch the desktop app) |
| First run is slow | Embedding model downloads on first use (~100-500 MB), subsequent runs use cache |
| PDF extracts blank text | The PDF is likely a scanned image; OCR is not included in the base version |

## License

MIT
