"""Typed configuration classes for Stash Copilot.

This module provides dataclass-based configuration with:
- Type safety and IDE autocomplete
- Validation at load time
- Single source of defaults (via defaults.py)
- Easy serialization for debugging

Usage:
    from stash_ai.config import PluginConfig

    # Load from Stash plugin settings
    config = PluginConfig.from_plugin_settings(raw_settings)

    # Access typed settings
    print(config.text_llm.provider)  # "ollama"
    print(config.recommendations.o_weight)  # 20.0
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .defaults import (
    EmbeddingDefaults,
    FrameDefaults,
    LLMDefaults,
    OMomentDefaults,
    PerformanceDefaults,
    RecommendationDefaults,
    VisionDefaults,
)


# =============================================================================
# Parsing Helpers
# =============================================================================


def _parse_int(value: Any, default: int) -> int:
    """Parse a value as integer with fallback to default.

    Args:
        value: Value to parse (string, int, or None)
        default: Default value if parsing fails

    Returns:
        Parsed integer or default
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _parse_float(value: Any, default: float) -> float:
    """Parse a value as float with fallback to default.

    Args:
        value: Value to parse (string, float, or None)
        default: Default value if parsing fails

    Returns:
        Parsed float or default
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_bool(value: Any, default: bool) -> bool:
    """Parse a value as boolean with fallback to default.

    Args:
        value: Value to parse (string, bool, or None)
        default: Default value if parsing fails

    Returns:
        Parsed boolean or default
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return default


def _parse_list(value: Any, separator: str = ",") -> list[str]:
    """Parse a comma-separated string into a list.

    Args:
        value: Value to parse (string or None)
        separator: Separator character (default: comma)

    Returns:
        List of stripped strings, or empty list
    """
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(separator) if item.strip()]
    return []


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class LLMProviderConfig:
    """LLM provider configuration (typed version).

    This is the new typed configuration class. For backwards compatibility,
    the legacy LLMSettings class is also exported from stash_ai.config.

    Attributes:
        provider: LLM provider name (ollama, openai, anthropic, openrouter)
        model: Model identifier
        base_url: API endpoint URL
        api_key: Optional API key for cloud providers
    """

    provider: str = LLMDefaults.PROVIDER
    model: str = LLMDefaults.MODEL
    base_url: str = LLMDefaults.BASE_URL
    api_key: Optional[str] = None

    @classmethod
    def from_plugin_settings(
        cls, settings: dict[str, Any], prefix: str = "text_llm"
    ) -> "LLMProviderConfig":
        """Create LLMProviderConfig from raw plugin settings dict.

        Args:
            settings: Raw settings dictionary from Stash
            prefix: Setting name prefix (text_llm or vision_llm)

        Returns:
            Configured LLMProviderConfig instance
        """
        return cls(
            provider=settings.get(f"{prefix}_provider") or LLMDefaults.PROVIDER,
            model=settings.get(f"{prefix}_model") or LLMDefaults.MODEL,
            base_url=settings.get(f"{prefix}_base_url") or LLMDefaults.BASE_URL,
            api_key=settings.get(f"{prefix}_api_key") or None,
        )

    def with_fallback(self, fallback: "LLMProviderConfig") -> "LLMProviderConfig":
        """Create settings with fallback for unset values.

        Used to inherit vision settings from text settings when not specified.

        Args:
            fallback: Settings to use for unset values

        Returns:
            New LLMProviderConfig with fallback values applied
        """
        return LLMProviderConfig(
            provider=self.provider if self.provider else fallback.provider,
            model=self.model if self.model else fallback.model,
            base_url=self.base_url if self.base_url else fallback.base_url,
            api_key=self.api_key if self.api_key else fallback.api_key,
        )

    @property
    def is_configured(self) -> bool:
        """Check if this LLM has explicit configuration."""
        return bool(self.provider and self.model)


@dataclass
class EmbeddingSettings:
    """Image embedding configuration.

    Attributes:
        provider: Embedding provider (openclip, etc.)
        model: Model name (ViT-H-14, etc.)
        device: Compute device (auto, cuda, cpu)
        visual_weight: Weight for visual vs metadata blend (0-1)
    """

    provider: str = EmbeddingDefaults.PROVIDER
    model: str = EmbeddingDefaults.MODEL
    device: str = EmbeddingDefaults.DEVICE
    visual_weight: float = EmbeddingDefaults.VISUAL_WEIGHT

    @classmethod
    def from_plugin_settings(cls, settings: dict[str, Any]) -> "EmbeddingSettings":
        """Create EmbeddingSettings from raw plugin settings dict."""
        return cls(
            provider=settings.get("image_embedding_provider") or EmbeddingDefaults.PROVIDER,
            model=settings.get("image_embedding_model") or EmbeddingDefaults.MODEL,
            device=settings.get("image_embedding_device") or EmbeddingDefaults.DEVICE,
            visual_weight=_parse_float(
                settings.get("embed_visual_weight"), EmbeddingDefaults.VISUAL_WEIGHT
            ),
        )


@dataclass
class RecommendationSettings:
    """Recommendation engine configuration.

    Attributes:
        top_scenes_for_profile: Number of scenes for preference profile
        o_weight: Weight for O-counter in engagement scoring
        view_weight: Weight per replay (views beyond first)
        duration_weight: Weight per hour of play time
        rating_weight: Weight per rating star
        time_decay_days: Half-life for time decay in days
    """

    top_scenes_for_profile: int = RecommendationDefaults.TOP_SCENES_FOR_PROFILE
    o_weight: float = RecommendationDefaults.O_WEIGHT
    view_weight: float = RecommendationDefaults.VIEW_WEIGHT
    duration_weight: float = RecommendationDefaults.DURATION_WEIGHT
    rating_weight: float = RecommendationDefaults.RATING_WEIGHT
    time_decay_days: int = RecommendationDefaults.TIME_DECAY_DAYS

    @classmethod
    def from_plugin_settings(
        cls, settings: dict[str, Any]
    ) -> "RecommendationSettings":
        """Create RecommendationSettings from raw plugin settings dict."""
        return cls(
            top_scenes_for_profile=_parse_int(
                settings.get("rec_top_scenes"),
                RecommendationDefaults.TOP_SCENES_FOR_PROFILE,
            ),
            o_weight=_parse_float(
                settings.get("rec_o_weight"), RecommendationDefaults.O_WEIGHT
            ),
            view_weight=_parse_float(
                settings.get("rec_view_weight"), RecommendationDefaults.VIEW_WEIGHT
            ),
            duration_weight=_parse_float(
                settings.get("rec_duration_weight"),
                RecommendationDefaults.DURATION_WEIGHT,
            ),
            rating_weight=_parse_float(
                settings.get("rec_rating_weight"), RecommendationDefaults.RATING_WEIGHT
            ),
            time_decay_days=_parse_int(
                settings.get("rec_time_decay_days"),
                RecommendationDefaults.TIME_DECAY_DAYS,
            ),
        )


@dataclass
class VisionSettings:
    """Vision analysis configuration.

    Attributes:
        auto_analyze: Enable automatic analysis on scene view
        frame_interval: Seconds between frame samples
        min_frames: Minimum frames to analyze
        max_frames: Maximum frames (0 = use model capability)
        hosted_max_frames: Frame limit for cloud providers
        debug: Enable debug output
    """

    auto_analyze: bool = VisionDefaults.AUTO_ANALYZE
    frame_interval: int = VisionDefaults.FRAME_INTERVAL
    min_frames: int = VisionDefaults.MIN_FRAMES
    max_frames: int = VisionDefaults.MAX_FRAMES
    hosted_max_frames: int = VisionDefaults.HOSTED_MAX_FRAMES
    debug: bool = VisionDefaults.DEBUG

    @classmethod
    def from_plugin_settings(cls, settings: dict[str, Any]) -> "VisionSettings":
        """Create VisionSettings from raw plugin settings dict."""
        return cls(
            auto_analyze=_parse_bool(
                settings.get("vision_auto_analyze"), VisionDefaults.AUTO_ANALYZE
            ),
            frame_interval=_parse_int(
                settings.get("vision_frame_interval"), VisionDefaults.FRAME_INTERVAL
            ),
            min_frames=_parse_int(
                settings.get("vision_min_frames"), VisionDefaults.MIN_FRAMES
            ),
            max_frames=_parse_int(
                settings.get("vision_max_frames"), VisionDefaults.MAX_FRAMES
            ),
            hosted_max_frames=_parse_int(
                settings.get("vision_hosted_max_frames"),
                VisionDefaults.HOSTED_MAX_FRAMES,
            ),
            debug=_parse_bool(settings.get("vision_debug"), VisionDefaults.DEBUG),
        )


@dataclass
class FrameSettings:
    """Frame extraction and selection configuration.

    Attributes:
        method: Frame selection method (kmeans, uniform, etc.)
        dynamic: Use dynamic frame count based on duration
        n_frames: Fixed frame count when not dynamic
        frames_per_minute: Frames per minute for dynamic calculation
        min_frames: Minimum frames to extract
        max_frames: Maximum frames to extract
    """

    method: str = FrameDefaults.METHOD
    dynamic: bool = FrameDefaults.DYNAMIC
    n_frames: int = FrameDefaults.N_FRAMES
    frames_per_minute: float = FrameDefaults.FRAMES_PER_MINUTE
    min_frames: int = FrameDefaults.MIN_FRAMES
    max_frames: int = FrameDefaults.MAX_FRAMES

    @classmethod
    def from_plugin_settings(cls, settings: dict[str, Any]) -> "FrameSettings":
        """Create FrameSettings from raw plugin settings dict."""
        return cls(
            method=settings.get("frame_analysis_method") or FrameDefaults.METHOD,
            dynamic=_parse_bool(
                settings.get("frame_analysis_dynamic"), FrameDefaults.DYNAMIC
            ),
            n_frames=_parse_int(
                settings.get("frame_analysis_n_frames"), FrameDefaults.N_FRAMES
            ),
            frames_per_minute=_parse_float(
                settings.get("frame_analysis_frames_per_minute"),
                FrameDefaults.FRAMES_PER_MINUTE,
            ),
            min_frames=_parse_int(
                settings.get("frame_analysis_min_frames"), FrameDefaults.MIN_FRAMES
            ),
            max_frames=_parse_int(
                settings.get("frame_analysis_max_frames"), FrameDefaults.MAX_FRAMES
            ),
        )


@dataclass
class OMomentSettings:
    """O-moment embedding configuration.

    Attributes:
        window_seconds: Total window size around marker
        frames_per_window: Frames to extract per O-moment
        tag_name: Tag name for O markers
    """

    window_seconds: int = OMomentDefaults.WINDOW_SECONDS
    frames_per_window: int = OMomentDefaults.FRAMES_PER_WINDOW
    tag_name: str = OMomentDefaults.TAG_NAME

    @classmethod
    def from_plugin_settings(cls, settings: dict[str, Any]) -> "OMomentSettings":
        """Create OMomentSettings from raw plugin settings dict."""
        return cls(
            window_seconds=_parse_int(
                settings.get("o_moment_window"), OMomentDefaults.WINDOW_SECONDS
            ),
            frames_per_window=_parse_int(
                settings.get("o_moment_frames"), OMomentDefaults.FRAMES_PER_WINDOW
            ),
            tag_name=settings.get("o_tag_name") or OMomentDefaults.TAG_NAME,
        )


@dataclass
class PerformanceSettings:
    """Performance tuning configuration.

    Attributes:
        embed_workers: Number of parallel embedding jobs
        frame_workers: Number of parallel FFmpeg processes
    """

    embed_workers: int = PerformanceDefaults.EMBED_WORKERS
    frame_workers: int = PerformanceDefaults.FRAME_WORKERS

    @classmethod
    def from_plugin_settings(cls, settings: dict[str, Any]) -> "PerformanceSettings":
        """Create PerformanceSettings from raw plugin settings dict."""
        return cls(
            embed_workers=_parse_int(
                settings.get("embed_num_workers"), PerformanceDefaults.EMBED_WORKERS
            ),
            frame_workers=_parse_int(
                settings.get("frame_extract_workers"),
                PerformanceDefaults.FRAME_WORKERS,
            ),
        )


@dataclass
class PluginConfig:
    """Complete plugin configuration.

    This is the main configuration container. Use from_plugin_settings()
    to create from Stash's raw settings dictionary.

    Attributes:
        text_llm: Text LLM provider settings
        vision_llm: Vision LLM settings (falls back to text_llm if not set)
        embedding: Image embedding settings
        recommendations: Recommendation engine settings
        vision: Vision analysis settings
        frames: Frame extraction settings
        o_moments: O-moment embedding settings
        performance: Performance tuning settings
        excluded_tags: Tags to exclude from AI analysis
    """

    text_llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    vision_llm: Optional[LLMProviderConfig] = None
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    recommendations: RecommendationSettings = field(
        default_factory=RecommendationSettings
    )
    vision: VisionSettings = field(default_factory=VisionSettings)
    frames: FrameSettings = field(default_factory=FrameSettings)
    o_moments: OMomentSettings = field(default_factory=OMomentSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    excluded_tags: list[str] = field(default_factory=list)

    @classmethod
    def from_plugin_settings(cls, settings: dict[str, Any]) -> "PluginConfig":
        """Create PluginConfig from raw Stash plugin settings.

        Args:
            settings: Raw settings dictionary from Stash

        Returns:
            Fully configured PluginConfig instance
        """
        text_llm = LLMProviderConfig.from_plugin_settings(settings, "text_llm")

        # Vision LLM inherits from text LLM if not explicitly configured
        raw_vision = LLMProviderConfig.from_plugin_settings(settings, "vision_llm")
        vision_llm = (
            raw_vision.with_fallback(text_llm)
            if settings.get("vision_llm_provider")
            else None
        )

        return cls(
            text_llm=text_llm,
            vision_llm=vision_llm,
            embedding=EmbeddingSettings.from_plugin_settings(settings),
            recommendations=RecommendationSettings.from_plugin_settings(settings),
            vision=VisionSettings.from_plugin_settings(settings),
            frames=FrameSettings.from_plugin_settings(settings),
            o_moments=OMomentSettings.from_plugin_settings(settings),
            performance=PerformanceSettings.from_plugin_settings(settings),
            excluded_tags=_parse_list(settings.get("excluded_tags")),
        )

    def get_effective_vision_llm(self) -> LLMProviderConfig:
        """Get the effective vision LLM settings.

        Returns vision_llm if explicitly configured, otherwise text_llm.
        """
        return self.vision_llm if self.vision_llm else self.text_llm


__all__ = [
    # Parsing helpers (for use in other modules)
    "_parse_int",
    "_parse_float",
    "_parse_bool",
    "_parse_list",
    # Settings classes
    "LLMProviderConfig",
    "EmbeddingSettings",
    "RecommendationSettings",
    "VisionSettings",
    "FrameSettings",
    "OMomentSettings",
    "PerformanceSettings",
    "PluginConfig",
]
