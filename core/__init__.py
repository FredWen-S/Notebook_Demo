# core/__init__.py
"""
Notebook Demo 核心引擎包
在这里统一暴露核心对外接口，保护内部实现细节。
"""

# 从 parser 模块导入核心方法和常量
from .parser import parse_file, SUPPORTED_TYPES

# 使用 __all__ 限制别人从 core 模块只能导入这些东西
__all__ = [
    "parse_file",
    "SUPPORTED_TYPES",
]