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