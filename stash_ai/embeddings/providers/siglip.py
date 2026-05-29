"""SigLIP embedding provider using HuggingFace transformers."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL import Image

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

# Known SigLIP models and their dimensions
SIGLIP_MODELS: dict[str, int] = {
    # Base models (224px input)
    "google/siglip-base-patch16-224": 768,
    "google/siglip-base-patch16-256": 768,
    "google/siglip-base-patch16-384": 768,
    "google/siglip-base-patch16-512": 768,
    # Large models
    "google/siglip-large-patch16-256": 1024,
    "google/siglip-large-patch16-384": 1024,
    # SO400M model (best quality)
    "google/siglip-so400m-patch14-384": 1152,
}


@register_embedding_provider("siglip")
class SigLIPEmbeddingProvider(BaseImageEmbeddingProvider):
    """SigLIP embedding provider via HuggingFace transformers.

    SigLIP (Sigmoid Loss for Language Image Pre-training) is Google's
    improved version of CLIP with better zero-shot performance.

    Supported models:
        - google/siglip-base-patch16-224 (768 dims, fastest)
        - google/siglip-base-patch16-384 (768 dims)
        - google/siglip-large-patch16-256 (1024 dims)
        - google/siglip-large-patch16-384 (1024 dims)
        - google/siglip-so400m-patch14-384 (1152 dims, best quality)

    Example:
        config = EmbeddingConfig(
            provider="siglip",
            model="google/siglip-base-patch16-224",
        )
        provider = SigLIPEmbeddingProvider(config)
        result = provider.embed_image("path/to/image.jpg")

    See https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize SigLIP provider.

        Args:
            config: Embedding configuration

        Raises:
            ImportError: If transformers is not installed
        """
        super().__init__(config)

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for SigLIP embeddings. "
                "Install with: pip install transformers"
            ) from e

        # Load model and processor
        self._processor = AutoProcessor.from_pretrained(config.model)  # type: ignore[no-untyped-call]
        self._model = AutoModel.from_pretrained(config.model).to(self.device)
        self._model.eval()

        # Get dimensions
        self._dimensions = SIGLIP_MODELS.get(config.model, config.dimensions or 768)

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

        # Process image
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

            # GPU normalization for consistency with batch method
            if self.normalize:
                outputs = F.normalize(outputs, p=2, dim=1)

            embedding = outputs.squeeze(0).cpu().numpy()

        return {
            "embedding": embedding.tolist(),
            "model": self.model,
            "dimensions": len(embedding),
            "image_size": image_size,
        }

    def _load_images_parallel(
        self,
        images: list[ImageInput],
        max_workers: int = 8,
    ) -> tuple[list["Image.Image"], list[tuple[int, int]]]:
        """Load images in parallel using thread pool for I/O.

        Args:
            images: List of image inputs (paths, bytes, or PIL Images)
            max_workers: Maximum number of threads for parallel loading

        Returns:
            Tuple of (list of PIL Images, list of image sizes)
        """
        from PIL import Image as PILImage

        # If all images are already PIL, no need for threading
        if all(isinstance(img, PILImage.Image) for img in images):
            pil_images = cast("list[PILImage.Image]", images)
            return pil_images, [img.size for img in pil_images]

        # Parallel load for file paths and bytes
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pil_images = list(executor.map(self._load_image, images))

        image_sizes = [img.size for img in pil_images]
        return pil_images, image_sizes

    def embed_images(self, images: list[ImageInput]) -> list[ImageEmbeddingResult]:
        """Generate embeddings for multiple images (batch).

        Optimized for performance with:
        - Parallel image loading using ThreadPoolExecutor for I/O
        - GPU-based batch normalization with F.normalize()

        Args:
            images: List of image paths, bytes, or PIL Images

        Returns:
            List of ImageEmbeddingResult objects
        """
        if not images:
            return []

        results: list[ImageEmbeddingResult] = []

        # Process in batches
        for batch_start in range(0, len(images), self.batch_size):
            batch_images = images[batch_start : batch_start + self.batch_size]

            # Parallel image loading (8 threads for I/O-bound operations)
            pil_images, image_sizes = self._load_images_parallel(batch_images)

            # Process batch
            inputs = self._processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self._model.get_image_features(**inputs)

                # GPU batch normalization (much faster than per-embedding CPU)
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                embeddings = embeddings.cpu().numpy()

            # Build results (no per-embedding normalization needed)
            for emb, size in zip(embeddings, image_sizes):
                results.append(
                    {
                        "embedding": emb.tolist(),
                        "model": self.model,
                        "dimensions": len(emb),
                        "image_size": size,
                    }
                )

        return results

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text.

        SigLIP supports both image and text embedding in a shared space.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        # Process text
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)

            # GPU normalization for consistency
            if self.normalize:
                outputs = F.normalize(outputs, p=2, dim=1)

            embedding = outputs.squeeze(0).cpu().numpy()

        return {
            "embedding": embedding.tolist(),
            "model": self.model,
            "dimensions": len(embedding),
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

        results: list[EmbeddingResult] = []

        # Process in batches
        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]

            # Process batch
            inputs = self._processor(text=batch_texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self._model.get_text_features(**inputs)

                # GPU batch normalization
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                embeddings = embeddings.cpu().numpy()

            for emb in embeddings:
                results.append(
                    {
                        "embedding": emb.tolist(),
                        "model": self.model,
                        "dimensions": len(emb),
                        "tokens_used": None,
                    }
                )

        return results

    def health_check(self) -> bool:
        """Check if the provider is working."""
        try:
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
            self._model = None

        # Release processor
        if hasattr(self, "_processor"):
            self._processor = None

        # Clear GPU cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self) -> None:
        """Destructor - backup cleanup if not already done."""
        self.cleanup()
