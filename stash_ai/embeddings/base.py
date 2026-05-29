"""Abstract base class for embedding providers."""

import atexit
import signal
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Union

import numpy as np
from numpy.typing import NDArray

from .config import EmbeddingConfig

# =============================================================================
# Global Resource Cleanup Registry
# =============================================================================
# This module provides cleanup functionality for GPU resources when tasks are
# cancelled. When Stash cancels a plugin task, it sends SIGTERM. Without proper
# handling, GPU models (~1-10GB VRAM each) remain allocated.

_cleanup_registry: list[Callable[[], None]] = []
_cleanup_done = False


def register_cleanup(cleanup_fn: Callable[[], None]) -> None:
    """Register a cleanup function to be called on shutdown.

    Args:
        cleanup_fn: Function to call during cleanup (should handle its own errors)
    """
    _cleanup_registry.append(cleanup_fn)


def unregister_cleanup(cleanup_fn: Callable[[], None]) -> None:
    """Unregister a cleanup function.

    Args:
        cleanup_fn: Function to remove from registry
    """
    try:
        _cleanup_registry.remove(cleanup_fn)
    except ValueError:
        pass  # Already removed


def run_cleanup() -> None:
    """Run all registered cleanup functions.

    Called automatically on SIGTERM/SIGINT or normal exit.
    Safe to call multiple times (only runs once).
    """
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    # Run registered cleanup functions in reverse order (LIFO)
    for cleanup_fn in reversed(_cleanup_registry):
        try:
            cleanup_fn()
        except Exception:
            pass  # Best effort cleanup - don't let one failure stop others

    # Final GPU cleanup
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _signal_handler(signum: int, frame: object) -> None:
    """Handle SIGTERM/SIGINT for graceful cleanup.

    Args:
        signum: Signal number received
        frame: Current stack frame (unused)
    """
    import os

    run_cleanup()
    # Use os._exit() instead of SystemExit to force-terminate all threads
    # SystemExit doesn't work when blocked on ThreadPoolExecutor or subprocess
    os._exit(0)


# Register signal handlers for graceful shutdown
# SIGTERM: Sent by Stash when cancelling a task
# SIGINT: Sent on Ctrl+C
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except (ValueError, OSError):
    # Signal handling may fail in some contexts (e.g., non-main thread)
    pass

# Register atexit handler as backup for normal exits
atexit.register(run_cleanup)

if TYPE_CHECKING:
    from PIL import Image

# Type alias for image inputs
ImageInput = Union[str, Path, bytes, "Image.Image"]


class EmbeddingResult(TypedDict):
    """Result from an embedding operation."""

    embedding: list[float]  # The embedding vector
    model: str  # Model used to generate
    dimensions: int  # Vector dimensionality
    tokens_used: int | None  # Token count if available


class ImageEmbeddingResult(TypedDict):
    """Result from an image embedding operation."""

    embedding: list[float]  # The embedding vector
    model: str  # Model used to generate
    dimensions: int  # Vector dimensionality
    image_size: tuple[int, int] | None  # (width, height) if available


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize the provider with configuration."""
        self.config = config
        self.model = config.model
        self.normalize = config.normalize

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text input.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensionality for this model."""
        pass

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        """L2 normalize an embedding vector."""
        if not self.normalize:
            return embedding
        arr: NDArray[np.float32] = np.array(embedding, dtype=np.float32)
        norm: float = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return list(arr.tolist())

    def health_check(self) -> bool:
        """Check if the provider is available and working."""
        try:
            result = self.embed_text("test")
            return len(result["embedding"]) > 0
        except Exception:
            return False

    @property
    def supports_images(self) -> bool:
        """Whether this provider supports image embedding."""
        return False


class BaseImageEmbeddingProvider(BaseEmbeddingProvider):
    """Base class for providers that support direct image embeddings.

    Image embedding providers can embed images directly without converting
    to text first. This is useful for CLIP, OpenCLIP, SigLIP, etc.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the image embedding provider."""
        super().__init__(config)
        self.device = self._resolve_device(config.device)
        self.batch_size = config.batch_size

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_image(self, image: ImageInput) -> "Image.Image":
        """Load an image from various input types.

        Args:
            image: Path string, Path object, bytes, or PIL Image

        Returns:
            PIL Image object
        """
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image
        elif isinstance(image, (str, Path)):
            return PILImage.open(image).convert("RGB")
        elif isinstance(image, bytes):
            import io

            return PILImage.open(io.BytesIO(image)).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    @abstractmethod
    def embed_image(self, image: ImageInput) -> ImageEmbeddingResult:
        """Generate embedding for a single image.

        Args:
            image: Image path, bytes, or PIL Image

        Returns:
            ImageEmbeddingResult with embedding vector
        """
        pass

    @abstractmethod
    def embed_images(self, images: list[ImageInput]) -> list[ImageEmbeddingResult]:
        """Generate embeddings for multiple images (batch).

        Args:
            images: List of image paths, bytes, or PIL Images

        Returns:
            List of ImageEmbeddingResult objects
        """
        pass

    @property
    def supports_images(self) -> bool:
        """Whether this provider supports image embedding."""
        return True

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text (if supported).

        Image embedding models like CLIP also support text embedding.
        Subclasses should override this if they support text.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support text embedding")

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support text embedding")
