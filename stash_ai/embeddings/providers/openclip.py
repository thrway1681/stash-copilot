"""OpenCLIP embedding provider for advanced CLIP models."""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, TypedDict, cast

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


class ModelInfo(TypedDict):
    """Model information for OpenCLIP models."""

    dim: int
    pretrained: str


# Known OpenCLIP models with their dimensions and default pretrained weights
OPENCLIP_MODELS: dict[str, ModelInfo] = {
    # OpenAI CLIP models
    "ViT-B-32": {"dim": 512, "pretrained": "openai"},
    "ViT-B-16": {"dim": 512, "pretrained": "openai"},
    "ViT-L-14": {"dim": 768, "pretrained": "openai"},
    "ViT-L-14-336": {"dim": 768, "pretrained": "openai"},
    # LAION trained models (larger, often better)
    "ViT-H-14": {"dim": 1024, "pretrained": "laion2b_s32b_b79k"},
    "ViT-g-14": {"dim": 1024, "pretrained": "laion2b_s12b_b42k"},
    "ViT-bigG-14": {"dim": 1280, "pretrained": "laion2b_s39b_b160k"},
    # EVA-CLIP models
    "EVA02-B-16": {"dim": 512, "pretrained": "merged2b_s8b_b131k"},
    "EVA02-L-14": {"dim": 768, "pretrained": "merged2b_s4b_b131k"},
    # DataComp models
    "ViT-L-14-CLIPA-datacomp1B": {"dim": 768, "pretrained": "datacomp1b"},
}


@register_embedding_provider("openclip")
class OpenCLIPEmbeddingProvider(BaseImageEmbeddingProvider):
    """OpenCLIP embedding provider with access to many pretrained models.

    OpenCLIP provides access to many CLIP variants trained on different
    datasets, including the original OpenAI CLIP and larger LAION-trained
    models.

    Supported models include:
        - ViT-B-32, ViT-B-16 (512 dims, OpenAI)
        - ViT-L-14, ViT-L-14-336 (768 dims, OpenAI)
        - ViT-H-14, ViT-g-14 (1024 dims, LAION)
        - ViT-bigG-14 (1280 dims, LAION, best quality)
        - EVA02-B-16, EVA02-L-14 (EVA-CLIP)

    Example:
        config = EmbeddingConfig(
            provider="openclip",
            model="ViT-L-14",
            pretrained="openai",  # or "laion2b_s32b_b79k"
        )
        provider = OpenCLIPEmbeddingProvider(config)
        result = provider.embed_image("path/to/image.jpg")

    See https://github.com/mlfoundations/open_clip for all available models.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize OpenCLIP provider.

        Args:
            config: Embedding configuration

        Raises:
            ImportError: If open_clip is not installed
        """
        super().__init__(config)

        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip is required for OpenCLIP embeddings. "
                "Install with: pip install open-clip-torch"
            ) from e

        # Determine pretrained weights
        model_info = OPENCLIP_MODELS.get(config.model)
        if model_info:
            pretrained = config.pretrained or model_info["pretrained"]
            self._dimensions = model_info["dim"]
        else:
            pretrained = config.pretrained or "openai"
            self._dimensions = config.dimensions or 512

        # Load model
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            config.model,
            pretrained=pretrained,
            device=self.device,
        )
        self._model.eval()

        # Get tokenizer for text embedding
        self._tokenizer = open_clip.get_tokenizer(config.model)

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

        # Preprocess and encode
        image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)

            # GPU normalization for consistency with batch method
            if self.normalize:
                embedding = F.normalize(embedding, p=2, dim=1)

            embedding = embedding.squeeze(0).cpu().numpy()

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

            # Stack preprocessed tensors and move to GPU
            image_tensors = torch.stack([self._preprocess(img) for img in pil_images]).to(
                self.device
            )

            # Encode batch
            with torch.no_grad():
                embeddings = self._model.encode_image(image_tensors)

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

        OpenCLIP models support both image and text embedding in a shared space.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        # Tokenize and encode
        tokens = self._tokenizer([text]).to(self.device)

        with torch.no_grad():
            embedding = self._model.encode_text(tokens)

            # GPU normalization for consistency
            if self.normalize:
                embedding = F.normalize(embedding, p=2, dim=1)

            embedding = embedding.squeeze(0).cpu().numpy()

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

            # Tokenize and encode
            tokens = self._tokenizer(batch_texts).to(self.device)

            with torch.no_grad():
                embeddings = self._model.encode_text(tokens)

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

    def list_models(self) -> list[str]:
        """List available OpenCLIP models."""
        try:
            import open_clip

            return list(open_clip.list_pretrained())
        except (ImportError, AttributeError, RuntimeError):
            return list(OPENCLIP_MODELS.keys())

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

        # Release preprocess transform
        if hasattr(self, "_preprocess"):
            self._preprocess = None

        # Release tokenizer
        if hasattr(self, "_tokenizer"):
            self._tokenizer = None

        # Clear GPU cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self) -> None:
        """Destructor - backup cleanup if not already done."""
        self.cleanup()
