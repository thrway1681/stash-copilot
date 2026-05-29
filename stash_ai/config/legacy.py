"""Legacy configuration classes for backwards compatibility.

This module preserves the original LLMConfig and helper functions that are
actively used throughout the codebase. These will be gradually migrated
to the new typed configuration system.

Note: Import these from stash_ai.config for backwards compatibility.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Map provider names to .env variable names
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "google": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Cache for parsed .env file (avoids re-reading on every call)
_env_cache: dict[str, str] | None = None


def _load_env_file() -> dict[str, str]:
    """Load API keys from the .env file in the plugin directory.

    Parses simple KEY=VALUE lines. Ignores comments and blank lines.
    Cached after first read.

    Returns:
        Dict of env var name → value
    """
    global _env_cache
    if _env_cache is not None:
        return _env_cache

    _env_cache = {}
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and value:
                    _env_cache[key] = value

    return _env_cache


def _get_api_key_for_provider(provider: str) -> str | None:
    """Look up the API key for a provider from the .env file.

    Args:
        provider: Provider name (e.g., "google", "openrouter")

    Returns:
        API key string or None if not found
    """
    env_var = _PROVIDER_ENV_KEYS.get(provider)
    if not env_var:
        return None

    env_data = _load_env_file()
    return env_data.get(env_var) or os.environ.get(env_var)


@dataclass
class LLMConfig:
    """Configuration for LLM provider.

    This is the runtime configuration passed to LLM providers, including
    request-specific parameters like temperature and max_tokens.
    """

    provider: str = "ollama"
    model: str = "llama2"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class LLMSettingsLegacy:
    """Resolved LLM settings from plugin configuration.

    This is the simpler settings structure used by get_text_llm_settings()
    and get_vision_llm_settings() helpers.

    Note: This class is aliased as 'LLMSettings' in stash_ai.config for
    backwards compatibility. The new typed settings are in settings.py.
    """

    provider: str
    model: str
    base_url: str
    api_key: str | None

    def to_config(self, temperature: float = 0.7, max_tokens: int = 1024) -> LLMConfig:
        """Convert to LLMConfig for use with providers."""
        return LLMConfig(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_text_llm_settings(
    plugin_settings: dict[str, Any],
    args: dict[str, Any] | None = None,
) -> LLMSettingsLegacy:
    """
    Resolve Text LLM settings from plugin configuration.

    Args:
        plugin_settings: Settings from Stash plugin configuration
        args: Optional task-specific arguments (override plugin settings)

    Returns:
        LLMSettingsLegacy with resolved values
    """
    args = args or {}

    provider = (
        args.get("text_llm_provider")
        or plugin_settings.get("text_llm_provider")
        or "ollama"
    )

    model = (
        args.get("text_llm_model") or plugin_settings.get("text_llm_model") or "llama3.1"
    )

    base_url = (
        args.get("text_llm_base_url")
        or plugin_settings.get("text_llm_base_url")
        or "http://localhost:11434"
    )

    api_key = (
        args.get("text_llm_api_key")
        or _get_api_key_for_provider(provider)
        or plugin_settings.get("text_llm_api_key")
        or None
    )

    return LLMSettingsLegacy(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def get_vision_llm_settings(
    plugin_settings: dict[str, Any],
    args: dict[str, Any] | None = None,
) -> LLMSettingsLegacy:
    """
    Resolve Vision LLM settings from plugin configuration.

    Falls back to Text LLM settings if vision-specific settings are not provided.

    Args:
        plugin_settings: Settings from Stash plugin configuration
        args: Optional task-specific arguments (override plugin settings)

    Returns:
        LLMSettingsLegacy with resolved values
    """
    args = args or {}
    text_settings = get_text_llm_settings(plugin_settings, args)

    # Vision settings fall back to text settings if not specified
    provider = (
        args.get("vision_llm_provider")
        or plugin_settings.get("vision_llm_provider")
        or text_settings.provider
    )

    model = (
        args.get("vision_llm_model")
        or plugin_settings.get("vision_llm_model")
        or text_settings.model
    )

    base_url = (
        args.get("vision_llm_base_url")
        or plugin_settings.get("vision_llm_base_url")
        or text_settings.base_url
    )

    api_key = (
        args.get("vision_llm_api_key")
        or _get_api_key_for_provider(provider)
        or plugin_settings.get("vision_llm_api_key")
        or text_settings.api_key
    )

    return LLMSettingsLegacy(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


__all__ = [
    "LLMConfig",
    "LLMSettingsLegacy",
    "get_text_llm_settings",
    "get_vision_llm_settings",
]
