"""Centralized default values for Stash Copilot configuration.

All default values are defined here as the single source of truth.
These defaults are used when settings are not provided or are invalid.

Categories:
- LLM_DEFAULTS: Language model provider defaults
- EMBEDDING_DEFAULTS: Image embedding defaults
- RECOMMENDATION_DEFAULTS: Recommendation engine weights
- VISION_DEFAULTS: Vision analysis parameters
- FRAME_DEFAULTS: Frame extraction settings
- O_MOMENT_DEFAULTS: O-moment embedding settings
- PERFORMANCE_DEFAULTS: Worker/threading settings
"""

from typing import Final

# =============================================================================
# LLM Provider Defaults
# =============================================================================


class LLMDefaults:
    """Default values for LLM providers."""

    PROVIDER: Final[str] = "ollama"
    MODEL: Final[str] = "llama3.1"
    BASE_URL: Final[str] = "http://localhost:11434"


# =============================================================================
# Embedding Defaults
# =============================================================================


class EmbeddingDefaults:
    """Default values for image embedding."""

    PROVIDER: Final[str] = "openclip"
    MODEL: Final[str] = "ViT-H-14"
    DEVICE: Final[str] = "auto"
    VISUAL_WEIGHT: Final[float] = 0.7  # Blend weight for visual vs metadata


# =============================================================================
# Recommendation Defaults
# =============================================================================


class RecommendationDefaults:
    """Default values for recommendation engine.

    Weights are based on empirical observation:
    - O-count is the strongest engagement signal (20x weight)
    - Replays indicate preference (2x per replay beyond first view)
    - Play duration shows interest (1x per hour)
    - Rating adds if present, no penalty if absent (1.5x per star)
    """

    TOP_SCENES_FOR_PROFILE: Final[int] = 20
    O_WEIGHT: Final[float] = 20.0
    VIEW_WEIGHT: Final[float] = 2.0  # Per replay (views beyond first)
    DURATION_WEIGHT: Final[float] = 1.0  # Per hour of play time
    RATING_WEIGHT: Final[float] = 1.5  # Per star (0-5 scale)
    TIME_DECAY_DAYS: Final[int] = 30  # Half-life for recency decay


# =============================================================================
# Vision Analysis Defaults
# =============================================================================


class VisionDefaults:
    """Default values for vision analysis."""

    AUTO_ANALYZE: Final[bool] = True
    FRAME_INTERVAL: Final[int] = 10  # Seconds between frames
    MIN_FRAMES: Final[int] = 1
    MAX_FRAMES: Final[int] = 0  # 0 = no limit (use model capability)
    HOSTED_MAX_FRAMES: Final[int] = 20  # Limit for cloud providers
    DEBUG: Final[bool] = True


# =============================================================================
# Frame Analysis Defaults
# =============================================================================


class FrameDefaults:
    """Default values for frame extraction and analysis."""

    METHOD: Final[str] = "kmeans"  # Frame selection method
    DYNAMIC: Final[bool] = True  # Use dynamic frame count based on duration
    N_FRAMES: Final[int] = 8  # Fixed frame count when not dynamic
    FRAMES_PER_MINUTE: Final[float] = 1.0  # For dynamic calculation
    MIN_FRAMES: Final[int] = 4
    MAX_FRAMES: Final[int] = 50


# =============================================================================
# O-Moment Defaults
# =============================================================================


class OMomentDefaults:
    """Default values for O-moment embedding extraction."""

    WINDOW_SECONDS: Final[int] = 120  # Total window size around marker
    FRAMES_PER_WINDOW: Final[int] = 12  # Frames extracted per O-moment
    TAG_NAME: Final[str] = "O"  # Tag name for O markers


# =============================================================================
# Performance Defaults
# =============================================================================


class PerformanceDefaults:
    """Default values for performance tuning."""

    EMBED_WORKERS: Final[int] = 2  # Parallel embedding jobs
    FRAME_WORKERS: Final[int] = 4  # Parallel FFmpeg processes


# =============================================================================
# Frontend Config Keys
# =============================================================================


class FrontendConfigKeys:
    """localStorage keys for frontend state persistence."""

    REC_MODE: Final[str] = "stash-copilot-rec-mode"
    SEED_WEIGHT: Final[str] = "stash-copilot-seed-weight"
    ENGAGEMENT_WEIGHT: Final[str] = "stash-copilot-engagement-weight"
    TIME_DECAY_DAYS: Final[str] = "stash-copilot-time-decay-days"
    VISUAL_WEIGHT: Final[str] = "stash-copilot-visual-weight"


class FrontendDefaults:
    """Default values for frontend configuration."""

    REC_MODE: Final[str] = "discover_new"
    SEED_WEIGHT: Final[float] = 0.3
    ENGAGEMENT_WEIGHT: Final[float] = 0.6
    TIME_DECAY_DAYS: Final[int] = 0
    VISUAL_WEIGHT: Final[float] = 0.7


__all__ = [
    "LLMDefaults",
    "EmbeddingDefaults",
    "RecommendationDefaults",
    "VisionDefaults",
    "FrameDefaults",
    "OMomentDefaults",
    "PerformanceDefaults",
    "FrontendConfigKeys",
    "FrontendDefaults",
]
