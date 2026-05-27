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