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