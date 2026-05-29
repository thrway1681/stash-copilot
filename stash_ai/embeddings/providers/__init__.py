"""Embedding provider implementations."""

from typing import List

# Always available: Ollama (just uses requests)
from .ollama import OllamaEmbeddingProvider

__all__: list[str] = ["OllamaEmbeddingProvider"]

# Conditionally import CLIP providers based on available dependencies

# CLIP via sentence-transformers
try:
    from .clip import CLIPEmbeddingProvider

    __all__.append("CLIPEmbeddingProvider")
except ImportError:
    # sentence-transformers not installed
    pass

# OpenCLIP
try:
    from .openclip import OpenCLIPEmbeddingProvider

    __all__.append("OpenCLIPEmbeddingProvider")
except ImportError:
    # open-clip-torch not installed
    pass

# SigLIP via HuggingFace transformers
try:
    from .siglip import SigLIPEmbeddingProvider

    __all__.append("SigLIPEmbeddingProvider")
except ImportError:
    # transformers not installed
    pass
