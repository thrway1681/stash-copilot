"""LLM abstraction layer for multiple providers."""

from .base import BaseLLMProvider, CompletionResult, Message, ToolCall
from .provider import get_provider, register_provider

# Import providers to trigger registration via @register_provider decorator
from .providers import (
    anthropic,
    google,
    ollama,
    openai,
    openrouter,
)

__all__ = [
    "BaseLLMProvider",
    "CompletionResult",
    "Message",
    "ToolCall",
    "get_provider",
    "register_provider",
]
