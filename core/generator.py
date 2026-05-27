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