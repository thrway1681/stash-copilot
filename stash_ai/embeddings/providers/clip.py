"""CLIP embedding provider using sentence-transformers."""

from ..base import (
    BaseImageEmbeddingProvider,
    EmbeddingResult,
    ImageEmbeddingResult,
    ImageInput,
    register_cleanup,
    unregister_cleanup,
)
from ..config import EmbeddingConfig
from ..provider import register_embedding_provider

# Known CLIP models via sentence-transformers and their dimensions
CLIP_MODELS: dict[str, int] = {
    "clip-ViT-B-32": 512,
    "clip-ViT-B-16": 512,
    "clip-ViT-L-14": 768,
    "clip-ViT-B-32-multilingual-v1": 512,
}


@register_embedding_provider("clip")
class CLIPEmbeddingProvider(BaseImageEmbeddingProvider):
    """CLIP embedding provider via sentence-transformers.

    This is the easiest way to use CLIP embeddings as sentence-transformers
    handles all the preprocessing and model loading automatically.

    Supported models:
        - clip-ViT-B-32 (512 dims, fastest)
        - clip-ViT-B-16 (512 dims)
        - clip-ViT-L-14 (768 dims, best quality)
        - clip-ViT-B-32-multilingual-v1 (512 dims, multilingual text)

    Example:
        config = EmbeddingConfig(provider="clip", model="clip-ViT-B-32")
        provider = CLIPEmbeddingProvider(config)
        result = provider.embed_image("path/to/image.jpg")
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize CLIP provider.

        Args:
            config: Embedding configuration

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        super().__init__(config)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CLIP embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

        # Load model
        self._model = SentenceTransformer(config.model, device=self.device)
        self._dimensions = CLIP_MODELS.get(config.model, config.dimensions or 512)

        # Register cleanup for graceful shutdown
        self._cleanup_registered = True
        register_cleanup(self.cleanup)

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions

    def embed_image(self, image: ImageInput) -> ImageEmbeddingResult:
        """Generate embedding for a single image.

        Args:
            image: Image path, bytes, or PIL Image

        Returns:
            ImageEmbeddingResult with embedding vector
        """
        pil_image = self._load_image(image)
        image_size: tuple[int, int] = pil_image.size

        # sentence-transformers handles preprocessing
        # Note: SentenceTransformer stubs don't include PIL Image support
        embedding = self._model.encode(pil_image, convert_to_numpy=True)  # type: ignore[call-overload]
        embedding_list = self._normalize_embedding(embedding.tolist())

        return {
            "embedding": embedding_list,
            "model": self.model,
            "dimensions": len(embedding_list),
            "image_size": image_size,
        }

    def embed_images(self, images: list[ImageInput]) -> list[ImageEmbeddingResult]:
        """Generate embeddings for multiple images (batch).

        Args:
            images: List of image paths, bytes, or PIL Images

        Returns:
            List of ImageEmbeddingResult objects
        """
        if not images:
            return []

        # Load all images
        pil_images = []
        image_sizes: list[tuple[int, int] | None] = []
        for img in images:
            pil_img = self._load_image(img)
            pil_images.append(pil_img)
            image_sizes.append(pil_img.size)

        # Batch encode (SentenceTransformer stubs don't include PIL Image support)
        embeddings = self._model.encode(
            pil_images,  # type: ignore[arg-type]
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        results: list[ImageEmbeddingResult] = []
        for emb, size in zip(embeddings, image_sizes):
            embedding_list = self._normalize_embedding(emb.tolist())
            results.append(
                {
                    "embedding": embedding_list,
                    "model": self.model,
                    "dimensions": len(embedding_list),
                    "image_size": size,
                }
            )

        return results

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text.

        CLIP models support both image and text embedding in a shared space.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        embedding_list = self._normalize_embedding(embedding.tolist())

        return {
            "embedding": embedding_list,
            "model": self.model,
            "dimensions": len(embedding_list),
            "tokens_used": None,
        }

    def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        results: list[EmbeddingResult] = []
        for emb in embeddings:
            embedding_list = self._normalize_embedding(emb.tolist())
            results.append(
                {
                    "embedding": embedding_list,
                    "model": self.model,
                    "dimensions": len(embedding_list),
                    "tokens_used": None,
                }
            )

        return results

    def health_check(self) -> bool:
        """Check if the provider is working."""
        try:
            # Test with a simple text embedding (faster than image)
            result = self.embed_text("test")
            return len(result["embedding"]) > 0
        except Exception:
            return False

    def cleanup(self) -> None:
        """Release GPU resources.

        Called automatically on SIGTERM/SIGINT or when the provider is deleted.
        Safe to call multiple times.
        """
        # Unregister from global cleanup to avoid double-cleanup
        if getattr(self, "_cleanup_registered", False):
            self._cleanup_registered = False
            unregister_cleanup(self.cleanup)

        # Release model
        if hasattr(self, "_model") and self._model is not None:
            try:
                del self._model
            except Exception:
                pass
            self._model = None  # type: ignore[assignment]

        # Clear GPU cache
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self) -> None:
        """Destructor - backup cleanup if not already done."""
        self.cleanup()
