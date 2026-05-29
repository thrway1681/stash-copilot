"""Provider factory and registry for embedding providers."""

from collections.abc import Callable

from .base import BaseEmbeddingProvider
from .config import EmbeddingConfig

# Registry of available embedding providers
_EMBEDDING_PROVIDERS: dict[str, type[BaseEmbeddingProvider]] = {}


def register_embedding_provider(
    name: str,
) -> Callable[[type[BaseEmbeddingProvider]], type[BaseEmbeddingProvider]]:
    """Decorator to register an embedding provider."""

    def decorator(
        cls: type[BaseEmbeddingProvider],
    ) -> type[BaseEmbeddingProvider]:
        _EMBEDDING_PROVIDERS[name.lower()] = cls
        return cls

    return decorator


def get_embedding_provider(config: EmbeddingConfig) -> BaseEmbeddingProvider:
    """Get an embedding provider instance based on configuration."""
    provider_name = config.provider.lower()

    if provider_name not in _EMBEDDING_PROVIDERS:
        available = ", ".join(_EMBEDDING_PROVIDERS.keys()) or "none"
        raise ValueError(f"Unknown embedding provider: '{provider_name}'. Available: {available}")

    provider_class = _EMBEDDING_PROVIDERS[provider_name]
    return provider_class(config)


def get_available_providers() -> list[str]:
    """Get list of registered embedding provider names."""
    return list(_EMBEDDING_PROVIDERS.keys())
