# 测试 Ollama（确保 Ollama 已启动）
from core.llm_client import create_llm_client

client = create_llm_client("ollama", model="qwen2.5:7b")
reply = client.chat([{"role": "user", "content": "你好，用一句话介绍什么是机器学习"}])
print(reply)