"""
Stash AI - LLM-powered insights for StashApp

This package provides AI capabilities for analyzing and interacting with
your Stash media library using large language models.
"""

from .config import LLMConfig
from .tools import get_all_tools, get_tools_schema

__version__ = "0.1.0"

__all__ = [
    "LLMConfig",
    "get_all_tools",
    "get_tools_schema",
]
