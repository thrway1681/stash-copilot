"""Embedding providers for vector generation."""

# Import providers to trigger registration
from . import providers
from .base import (
    BaseEmbeddingProvider,
    BaseImageEmbeddingProvider,
    EmbeddingResult,
    ImageEmbeddingResult,
    ImageInput,
)
from .config import EmbeddingConfig
from .provider import (
    get_available_providers,
    get_embedding_provider,
    register_embedding_provider,
)

__all__ = [
    "BaseEmbeddingProvider",
    "BaseImageEmbeddingProvider",
    "EmbeddingConfig",
    "EmbeddingResult",
    "ImageEmbeddingResult",
    "ImageInput",
    "get_available_providers",
    "get_embedding_provider",
    "register_embedding_provider",
]
