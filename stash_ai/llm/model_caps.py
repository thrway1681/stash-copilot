"""Model capabilities and context window information for VLMs."""

import os
from dataclasses import dataclass


@dataclass
class VLMCapabilities:
    """Capabilities for a vision language model."""

    context_tokens: int  # Total context window in tokens
    tokens_per_image: int  # Tokens consumed per image
    max_output_tokens: int  # Maximum output tokens
    optimal_resolution: int  # Optimal image resolution (width)
    supports_multiple_images: bool  # Can handle multiple images per request

    def calculate_max_images(
        self,
        prompt_tokens: int = 2000,
        reserved_output: int = 4000,
    ) -> int:
        """
        Calculate maximum images that fit in context.

        Args:
            prompt_tokens: Estimated tokens for system + user prompts
            reserved_output: Tokens to reserve for model output

        Returns:
            Maximum number of images that fit
        """
        available = self.context_tokens - prompt_tokens - reserved_output
        if available <= 0:
            return 1
        return max(1, available // self.tokens_per_image)


# Shared capability definitions to avoid duplication
_GEMMA3_CAPS = VLMCapabilities(
    context_tokens=131072,
    tokens_per_image=256,
    max_output_tokens=8192,
    optimal_resolution=896,
    supports_multiple_images=True,
)

# Known model capabilities (keyword -> capabilities)
# These are approximate values based on model documentation
MODEL_CAPABILITIES: dict[str, VLMCapabilities] = {
    # Gemma 3 - 128K context, 256 tokens/image at 896x896
    "gemma-3": _GEMMA3_CAPS,
    "gemma3": _GEMMA3_CAPS,  # Alias for non-hyphenated naming
    # Claude Sonnet 4.x - 200K context, ~1600 tokens/image (formula: w*h/750)
    "claude-sonnet": VLMCapabilities(
        context_tokens=200000,
        tokens_per_image=1600,
        max_output_tokens=64000,
        optimal_resolution=1568,
        supports_multiple_images=True,
    ),
    "claude-3": VLMCapabilities(
        context_tokens=200000,
        tokens_per_image=1600,
        max_output_tokens=4096,
        optimal_resolution=1568,
        supports_multiple_images=True,
    ),
    # GPT-4o - 128K context, 85 base + 170 per 512x512 tile (high detail)
    "4o": VLMCapabilities(
        context_tokens=128000,
        tokens_per_image=1000,  # ~765 low detail, ~1100 high detail; 1000 avg
        max_output_tokens=16384,
        optimal_resolution=1024,
        supports_multiple_images=True,
    ),
    # Gemini 1.5/2.0 - 1M context, tile-based: 258 tokens per 768x768 tile
    # At 896px optimal resolution: ceil(896/768)^2 = 4 tiles = 1032 tokens
    "gemini-1.5": VLMCapabilities(
        context_tokens=1000000,
        tokens_per_image=1032,
        max_output_tokens=8192,
        optimal_resolution=896,
        supports_multiple_images=True,
    ),
    "gemini-2": VLMCapabilities(
        context_tokens=1000000,
        tokens_per_image=1032,
        max_output_tokens=8192,
        optimal_resolution=896,
        supports_multiple_images=True,
    ),
    # Gemini 3.x - 1M context, fixed-budget tokenization (media_resolution=HIGH default)
    "gemini-3": VLMCapabilities(
        context_tokens=1000000,
        tokens_per_image=1120,
        max_output_tokens=65536,
        optimal_resolution=896,
        supports_multiple_images=True,
    ),
    # LLaVA variants - 4K context typically, single image
    "llava": VLMCapabilities(
        context_tokens=4096,
        tokens_per_image=576,
        max_output_tokens=2048,
        optimal_resolution=672,
        supports_multiple_images=False,
    ),
    # Pixtral - 128K context, 16x16 patch tokenization (1024px = 64*64 = 4096 tokens)
    "pixtral": VLMCapabilities(
        context_tokens=131072,
        tokens_per_image=4096,
        max_output_tokens=4096,
        optimal_resolution=1024,
        supports_multiple_images=False,  # Works better with grid
    ),
    # Kimi K2.5 - 256K context, native multimodal with video understanding
    "kimi": VLMCapabilities(
        context_tokens=262144,
        tokens_per_image=1000,  # Not publicly documented; estimate
        max_output_tokens=64000,
        optimal_resolution=1024,
        supports_multiple_images=True,
    ),
    # Qwen2-VL - 32K context, multi-image support
    "qwen2-vl": VLMCapabilities(
        context_tokens=32768,
        tokens_per_image=512,
        max_output_tokens=2048,
        optimal_resolution=896,
        supports_multiple_images=True,
    ),
    # Default fallback for unknown models
    "default": VLMCapabilities(
        context_tokens=8192,
        tokens_per_image=512,
        max_output_tokens=2048,
        optimal_resolution=640,
        supports_multiple_images=True,
    ),
}


def get_model_capabilities(model_name: str) -> VLMCapabilities:
    """
    Get capabilities for a model based on its name.

    Args:
        model_name: The model name/ID

    Returns:
        VLMCapabilities for the model (or default if unknown)
    """
    model_lower = model_name.lower()

    # Check each known model keyword
    for keyword, caps in MODEL_CAPABILITIES.items():
        if keyword != "default" and keyword in model_lower:
            return caps

    # Return default if no match
    return MODEL_CAPABILITIES["default"]


def calculate_optimal_frame_count(
    model_name: str,
    video_duration_seconds: float,
    min_frames: int = 4,
    max_frames: int = 50,
) -> int:
    """
    Calculate optimal number of frames based on model context and video duration.

    Args:
        model_name: The VLM model name
        video_duration_seconds: Video duration in seconds
        min_frames: Minimum frames to extract
        max_frames: Maximum frames cap

    Returns:
        Optimal number of frames
    """
    caps = get_model_capabilities(model_name)

    # Calculate max images that fit in context
    context_max = caps.calculate_max_images()

    # If model doesn't support multiple images, return 1
    if not caps.supports_multiple_images:
        return 1

    # Calculate based on video duration (one frame per 10 seconds is good coverage)
    duration_based = max(min_frames, int(video_duration_seconds / 10))

    # Take minimum of context limit and duration-based, capped at max_frames
    optimal = min(context_max, duration_based, max_frames)

    return max(min_frames, optimal)


def is_debug_logging_enabled() -> bool:
    """Check if debug logging is enabled via environment variable."""
    return os.environ.get("STASH_COPILOT_DEBUG", "").lower() in ("1", "true", "yes")
