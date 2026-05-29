"""Provider factory and registry for LLM providers."""

from collections.abc import Callable

from ..config import LLMConfig
from .base import BaseLLMProvider

# Registry of available providers
_PROVIDERS: dict[str, type[BaseLLMProvider]] = {}


def register_provider(name: str) -> Callable[[type[BaseLLMProvider]], type[BaseLLMProvider]]:
    """
    Decorator to register an LLM provider.

    Usage:
        @register_provider("ollama")
        class OllamaProvider(BaseLLMProvider):
            ...

    Args:
        name: The provider identifier (e.g., "ollama", "openai")

    Returns:
        Decorator function
    """

    def decorator(cls: type[BaseLLMProvider]) -> type[BaseLLMProvider]:
        _PROVIDERS[name.lower()] = cls
        return cls

    return decorator


def get_provider(config: LLMConfig) -> BaseLLMProvider:
    """
    Get an LLM provider instance based on configuration.

    Args:
        config: LLM configuration specifying provider and settings

    Returns:
        Configured provider instance

    Raises:
        ValueError: If the specified provider is not registered
    """
    provider_name = config.provider.lower()

    if provider_name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys()) or "none"
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. Available providers: {available}"
        )

    provider_class = _PROVIDERS[provider_name]
    return provider_class(config)
