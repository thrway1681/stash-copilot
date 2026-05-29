"""Configuration module for Stash Copilot.

This module provides typed configuration with centralized defaults.

Backwards Compatibility:
    The legacy classes (LLMConfig, LLMSettings) are preserved for existing code:

    from stash_ai.config import LLMConfig, LLMSettings, get_text_llm_settings

New Usage:
    For new code, use the typed configuration classes:

    from stash_ai.config import PluginConfig, LLMProviderConfig

    # Load from raw plugin settings
    config = PluginConfig.from_plugin_settings(raw_settings)

    # Access typed values
    provider = config.text_llm.provider  # "ollama"
    o_weight = config.recommendations.o_weight  # 20.0

Defaults:
    All default values are defined in defaults.py:

    from stash_ai.config.defaults import RecommendationDefaults
    print(RecommendationDefaults.O_WEIGHT)  # 20.0
"""

# Legacy exports - for backwards compatibility
# Default value classes
from .defaults import (
    EmbeddingDefaults,
    FrameDefaults,
    FrontendConfigKeys,
    FrontendDefaults,
    LLMDefaults,
    OMomentDefaults,
    PerformanceDefaults,
    RecommendationDefaults,
    VisionDefaults,
)
from .legacy import (
    LLMConfig,
    LLMSettingsLegacy as LLMSettings,  # Alias for backwards compatibility
    get_text_llm_settings,
    get_vision_llm_settings,
)

# New typed settings classes
from .settings import (
    EmbeddingSettings,
    FrameSettings,
    LLMProviderConfig,
    OMomentSettings,
    PerformanceSettings,
    PluginConfig,
    RecommendationSettings,
    VisionSettings,
)

__all__ = [
    # Legacy exports (backwards compatibility)
    "LLMConfig",
    "LLMSettings",
    "get_text_llm_settings",
    "get_vision_llm_settings",
    # Default value classes
    "LLMDefaults",
    "EmbeddingDefaults",
    "RecommendationDefaults",
    "VisionDefaults",
    "FrameDefaults",
    "OMomentDefaults",
    "PerformanceDefaults",
    "FrontendConfigKeys",
    "FrontendDefaults",
    # New typed settings classes
    "LLMProviderConfig",
    "EmbeddingSettings",
    "RecommendationSettings",
    "VisionSettings",
    "FrameSettings",
    "OMomentSettings",
    "PerformanceSettings",
    "PluginConfig",
]
