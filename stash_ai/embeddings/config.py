"""Configuration for embedding providers."""

from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider.

    Attributes:
        provider: Provider name (ollama, clip, openclip, siglip)
        model: Model name/identifier
        base_url: API base URL (for remote providers like Ollama)
        api_key: API key (for cloud providers)
        dimensions: Override output dimensions if supported
        normalize: L2 normalize embeddings for cosine similarity
        device: Device for local models (cpu, cuda, mps, auto)
        batch_size: Batch size for image embedding
        pretrained: Pretrained weights name (for OpenCLIP)
    """

    provider: str = "ollama"
    model: str = "nomic-embed-text"  # Default text embedding model
    base_url: str | None = None  # e.g., http://localhost:11434
    api_key: str | None = None
    dimensions: int | None = None  # Override output dimensions if supported
    normalize: bool = True  # L2 normalize embeddings for cosine similarity
    # Image embedding specific options
    device: str = "auto"  # cpu, cuda, mps, or auto (detect)
    batch_size: int = 8  # Batch size for image processing
    pretrained: str | None = None  # Pretrained weights (for OpenCLIP)

    @property
    def model_key(self) -> str:
        """Generate unique key for this model configuration.

        Used to namespace embeddings in storage so multiple models
        can have embeddings for the same scene.

        Returns:
            Unique identifier string like "siglip" or "openclip:ViT-H-14"
        """
        if self.provider == "siglip":
            return "siglip"
        elif self.provider == "openclip":
            return f"openclip:{self.model}"
        elif self.provider == "clip":
            return f"clip:{self.model}"
        else:
            return f"{self.provider}:{self.model}"

    @classmethod
    def from_model_key(cls, model_key: str, device: str = "auto") -> "EmbeddingConfig":
        """Create config from a model_key string.

        Parses model keys like "siglip", "openclip:ViT-H-14", "clip:model-name"
        back into provider and model components.

        Args:
            model_key: Model key string (e.g., "openclip:ViT-H-14")
            device: Device for local models (cpu, cuda, mps, auto)

        Returns:
            EmbeddingConfig instance
        """
        if model_key == "siglip":
            return cls(provider="siglip", model="google/siglip-base-patch16-224", device=device)
        elif ":" in model_key:
            provider, model = model_key.split(":", 1)
            return cls(provider=provider, model=model, device=device)
        else:
            # Fallback: treat as provider name with default model
            return cls(provider=model_key, device=device)


@dataclass
class DenseFrameConfig:
    """Configuration for dense frame extraction (1fps).

    Attributes:
        sampling_rate: Frames per second to extract (default: 1.0)
        deduplication_threshold: Skip frames with cosine similarity > this (default: 0.99)
        use_deduplication: Enable smart deduplication (default: True)
        min_unique_frames: Always keep at least N frames even if duplicates (default: 10)
        max_frames_per_scene: Cap total frames per scene (default: 600 = 10 minutes at 1fps)
        frame_width: Resize width for embedding (default: 640)
        cache_dir: Directory for frame cache (default: "assets/embedded_frames")
    """

    sampling_rate: float = 1.0
    deduplication_threshold: float = 0.99
    use_deduplication: bool = True
    min_unique_frames: int = 10
    max_frames_per_scene: int = 600
    frame_width: int = 640
    cache_dir: str = "assets/embedded_frames"
