# Phase 4: Configuration Architecture Proposal

**Generated:** 2026-02-15

## Current Configuration Landscape

### Configuration Sources (7 Total)

| Source | Location | Count | Editable By |
|--------|----------|-------|-------------|
| Plugin Settings | `stash-copilot.yml` | 39 | User (Stash UI) |
| Build Config | `pyproject.toml` | ~50 | Developer |
| Prompt Templates | `prompts/*.yaml` | 7 | Developer |
| Database Schema | SQLite `schema_info` | 2+ | System |
| Environment Variables | Shell | 2 | User/System |
| Frontend localStorage | Browser | 5 | User (implicit) |
| Runtime Arguments | Task args | Many | System |

### Current Plugin Settings by Category

```yaml
# stash-copilot.yml - 39 settings

# Text LLM (4)
text_llm_provider, text_llm_model, text_llm_base_url, text_llm_api_key

# Vision LLM (4)
vision_llm_provider, vision_llm_model, vision_llm_base_url, vision_llm_api_key

# General (1)
excluded_tags

# Vision Analysis (6)
vision_auto_analyze, vision_frame_interval, vision_min_frames,
vision_max_frames, vision_debug, vision_hosted_max_frames

# Embeddings (7)
embedding_model, embed_visual_weight, image_embedding_provider,
image_embedding_model, image_embedding_device

# Frame Analysis (7)
frame_analysis_method, frame_analysis_n_frames, frame_analysis_dynamic,
frame_analysis_frames_per_minute, frame_analysis_min_frames,
frame_analysis_max_frames, frame_analysis_compare

# Performance (2)
embed_num_workers, frame_extract_workers

# Recommendations (6)
rec_top_scenes, rec_o_weight, rec_view_weight, rec_duration_weight,
rec_rating_weight, rec_time_decay_days

# O-Moments (3)
o_moment_window, o_moment_frames, o_tag_name
```

## Analysis

### Issues with Current Configuration

1. **Flat Namespace:** All 39 settings at same level, hard to find related settings
2. **String-Based Values:** No validation, type coercion in code
3. **Scattered Defaults:** Defaults in multiple places (yml, code, CLAUDE.md)
4. **Inconsistent Naming:** Mix of `snake_case` and `prefix_snake_case`
5. **Hidden Settings:** Some behavior controlled by code, not exposed

### Settings Exposure Analysis

| Setting | Should Expose | Current | Reason |
|---------|---------------|---------|--------|
| LLM Provider/Model | ✅ Yes | ✅ Exposed | User choice |
| API Keys | ✅ Yes | ✅ Exposed | User credential |
| Excluded Tags | ✅ Yes | ✅ Exposed | Privacy control |
| Frame Interval | ⚠️ Advanced | ✅ Exposed | Technical tuning |
| Worker Counts | ⚠️ Advanced | ✅ Exposed | Performance tuning |
| Rec Weights | ⚠️ Advanced | ✅ Exposed | Algorithm tuning |
| Dedup Threshold | ❌ Hide | ❌ Hidden | Internal |
| Schema Version | ❌ Hide | ❌ Hidden | System |

## Proposed Configuration Architecture

### 1. Tiered Configuration System

```
┌─────────────────────────────────────────────────┐
│ Tier 1: User-Facing (Stash Plugin Settings)    │
│ - LLM provider selection                        │
│ - API keys                                      │
│ - Privacy settings (excluded tags)              │
│ - Feature toggles                               │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Tier 2: Advanced (Stash Plugin Settings)        │
│ - Performance tuning                            │
│ - Algorithm parameters                          │
│ - Frame extraction settings                     │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Tier 3: Developer (Code/Config Files)           │
│ - Prompt templates                              │
│ - Model capabilities                            │
│ - Internal thresholds                           │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ Tier 4: System (Computed/Database)              │
│ - Schema versions                               │
│ - Cached states                                 │
│ - Model-specific thresholds                     │
└─────────────────────────────────────────────────┘
```

### 2. Reorganized Plugin Settings

**Proposed `stash-copilot.yml` structure:**

```yaml
settings:
  # ========== TIER 1: ESSENTIAL ==========
  # These appear first in Stash UI, most users need these

  text_llm_provider:
    displayName: "🤖 Text AI Provider"
    description: "AI provider for chat and analysis. Options: ollama (local), openai, anthropic, openrouter"
    type: STRING

  text_llm_model:
    displayName: "🤖 Text AI Model"
    description: "Model name (e.g., llama3.1, gpt-4o, claude-3-5-sonnet)"
    type: STRING

  text_llm_api_key:
    displayName: "🔑 Text AI API Key"
    description: "API key for cloud providers. Leave empty for Ollama."
    type: STRING

  text_llm_base_url:
    displayName: "🌐 Text AI URL"
    description: "API endpoint. Default: http://localhost:11434 for Ollama"
    type: STRING

  excluded_tags:
    displayName: "🏷️ Excluded Tags"
    description: "Tags to exclude from AI analysis (comma-separated)"
    type: STRING

  # ========== TIER 1: VISION (if different from text) ==========

  vision_llm_provider:
    displayName: "👁️ Vision AI Provider"
    description: "Provider for vision analysis. Leave empty to use Text AI."
    type: STRING

  # ... (other vision settings)

  # ========== TIER 2: ADVANCED - RECOMMENDATIONS ==========
  # Collapsed section in UI (if Stash supports it)

  rec_o_weight:
    displayName: "⚖️ O-Count Weight"
    description: "Weight for O-counter in engagement scoring (default: 20.0)"
    type: STRING

  rec_view_weight:
    displayName: "⚖️ Replay Weight"
    description: "Weight per replay in engagement scoring (default: 2.0)"
    type: STRING

  # ... etc
```

### 3. Typed Configuration Classes

**Current:** Settings parsed from strings with defaults scattered in code
**Proposed:** Centralized typed configuration

```python
# stash_ai/config/settings.py
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class LLMSettings:
    """LLM provider configuration."""
    provider: str = "ollama"
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None

    @classmethod
    def from_plugin_settings(cls, settings: dict, prefix: str = "text_llm") -> "LLMSettings":
        return cls(
            provider=settings.get(f"{prefix}_provider") or "ollama",
            model=settings.get(f"{prefix}_model") or "llama3.1",
            base_url=settings.get(f"{prefix}_base_url") or "http://localhost:11434",
            api_key=settings.get(f"{prefix}_api_key") or None,
        )

@dataclass
class RecommendationSettings:
    """Recommendation engine configuration."""
    top_scenes_for_profile: int = 20
    o_weight: float = 20.0
    view_weight: float = 2.0
    duration_weight: float = 1.0
    rating_weight: float = 1.5
    time_decay_days: int = 30

    @classmethod
    def from_plugin_settings(cls, settings: dict) -> "RecommendationSettings":
        return cls(
            top_scenes_for_profile=_parse_int(settings.get("rec_top_scenes"), 20),
            o_weight=_parse_float(settings.get("rec_o_weight"), 20.0),
            view_weight=_parse_float(settings.get("rec_view_weight"), 2.0),
            duration_weight=_parse_float(settings.get("rec_duration_weight"), 1.0),
            rating_weight=_parse_float(settings.get("rec_rating_weight"), 1.5),
            time_decay_days=_parse_int(settings.get("rec_time_decay_days"), 30),
        )

@dataclass
class PluginConfig:
    """Complete plugin configuration."""
    text_llm: LLMSettings = field(default_factory=LLMSettings)
    vision_llm: Optional[LLMSettings] = None
    recommendations: RecommendationSettings = field(default_factory=RecommendationSettings)
    excluded_tags: List[str] = field(default_factory=list)
    # ... other settings

    @classmethod
    def load(cls, stash: StashInterface) -> "PluginConfig":
        """Load configuration from Stash plugin settings."""
        raw = stash.get_plugin_settings("stash-copilot") or {}
        return cls(
            text_llm=LLMSettings.from_plugin_settings(raw, "text_llm"),
            vision_llm=LLMSettings.from_plugin_settings(raw, "vision_llm")
                       if raw.get("vision_llm_provider") else None,
            recommendations=RecommendationSettings.from_plugin_settings(raw),
            excluded_tags=_parse_list(raw.get("excluded_tags")),
        )
```

**Pros:**
- Type safety with IDE autocomplete
- Validation at load time
- Single source of defaults
- Easy to test

---

### 4. Settings Categories Proposal

#### Category 1: Provider Selection (Essential)

| Setting | Display Name | Type | Default |
|---------|--------------|------|---------|
| `text_llm_provider` | Text AI Provider | enum | `ollama` |
| `text_llm_model` | Text AI Model | string | `llama3.1` |
| `text_llm_base_url` | Text AI URL | string | `http://localhost:11434` |
| `text_llm_api_key` | Text AI API Key | password | - |
| `vision_llm_provider` | Vision AI Provider | enum | (use text) |
| `vision_llm_model` | Vision AI Model | string | (use text) |
| `vision_llm_base_url` | Vision AI URL | string | (use text) |
| `vision_llm_api_key` | Vision AI API Key | password | (use text) |
| `image_embedding_provider` | Image Embedding Provider | enum | `openclip` |
| `image_embedding_model` | Image Embedding Model | string | `ViT-H-14` |

#### Category 2: Privacy & Filtering (Essential)

| Setting | Display Name | Type | Default |
|---------|--------------|------|---------|
| `excluded_tags` | Excluded Tags | string | - |

#### Category 3: Recommendation Weights (Advanced)

| Setting | Display Name | Type | Default | Description |
|---------|--------------|------|---------|-------------|
| `rec_top_scenes` | Profile Scenes | int | 20 | Scenes for preference profile |
| `rec_o_weight` | O-Count Weight | float | 20.0 | Engagement multiplier |
| `rec_view_weight` | Replay Weight | float | 2.0 | Per-replay multiplier |
| `rec_duration_weight` | Duration Weight | float | 1.0 | Per-hour multiplier |
| `rec_rating_weight` | Rating Weight | float | 1.5 | Per-star multiplier |
| `rec_time_decay_days` | Time Decay Days | int | 30 | Half-life for decay |

#### Category 4: Performance Tuning (Advanced)

| Setting | Display Name | Type | Default | Description |
|---------|--------------|------|---------|-------------|
| `embed_num_workers` | Embedding Workers | int | 2 | Parallel embedding jobs |
| `frame_extract_workers` | FFmpeg Workers | int | 4 | Parallel frame extraction |
| `image_embedding_device` | Embedding Device | enum | `auto` | GPU selection |

#### Category 5: Frame Analysis (Advanced)

| Setting | Display Name | Type | Default |
|---------|--------------|------|---------|
| `frame_analysis_method` | Selection Method | enum | `kmeans` |
| `frame_analysis_dynamic` | Dynamic Count | bool | true |
| `frame_analysis_n_frames` | Fixed Frame Count | int | 8 |
| `frame_analysis_frames_per_minute` | Frames/Minute | float | 1.0 |
| `frame_analysis_min_frames` | Min Frames | int | 4 |
| `frame_analysis_max_frames` | Max Frames | int | 50 |

#### Category 6: Vision Analysis (Advanced)

| Setting | Display Name | Type | Default |
|---------|--------------|------|---------|
| `vision_auto_analyze` | Auto-Analyze | bool | true |
| `vision_frame_interval` | Frame Interval | int | 10 |
| `vision_min_frames` | Min Frames | int | 1 |
| `vision_max_frames` | Max Frames | int | 0 (no limit) |
| `vision_debug` | Debug Mode | bool | true |

#### Category 7: O-Moments (Advanced)

| Setting | Display Name | Type | Default |
|---------|--------------|------|---------|
| `o_moment_window` | Window Seconds | int | 120 |
| `o_moment_frames` | Frames Per Window | int | 12 |
| `o_tag_name` | O Marker Tag | string | `O` |

---

### 5. Configuration Inheritance

**Current:** Vision LLM falls back to Text LLM via code logic
**Proposed:** Explicit inheritance chain

```python
@dataclass
class LLMSettings:
    provider: str = "ollama"
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None

    def with_fallback(self, fallback: "LLMSettings") -> "LLMSettings":
        """Return settings with fallback for unset values."""
        return LLMSettings(
            provider=self.provider or fallback.provider,
            model=self.model or fallback.model,
            base_url=self.base_url or fallback.base_url,
            api_key=self.api_key or fallback.api_key,
        )

# Usage
config.vision_llm = raw_vision_settings.with_fallback(config.text_llm)
```

---

### 6. Frontend Configuration State

**Current:** localStorage keys scattered throughout code
**Proposed:** Centralized frontend config

```javascript
// frontend/core/config.js
const CONFIG_KEYS = {
    // User preferences
    REC_MODE: 'stash-copilot-rec-mode',
    SEED_WEIGHT: 'stash-copilot-seed-weight',
    ENGAGEMENT_WEIGHT: 'stash-copilot-engagement-weight',
    TIME_DECAY_DAYS: 'stash-copilot-time-decay-days',
    VISUAL_WEIGHT: 'stash-copilot-visual-weight',
};

const DEFAULT_CONFIG = {
    recMode: 'discover_new',
    seedWeight: 0.3,
    engagementWeight: 0.6,
    timeDecayDays: 0,
    visualWeight: 0.7,
};

export function loadConfig() {
    return {
        recMode: localStorage.getItem(CONFIG_KEYS.REC_MODE) || DEFAULT_CONFIG.recMode,
        seedWeight: parseFloat(localStorage.getItem(CONFIG_KEYS.SEED_WEIGHT)) || DEFAULT_CONFIG.seedWeight,
        // ...
    };
}

export function saveConfig(key, value) {
    localStorage.setItem(CONFIG_KEYS[key], value);
}
```

---

### 7. Environment Variables

**Keep minimal:**

| Variable | Purpose | Default |
|----------|---------|---------|
| `STASH_COPILOT_DEBUG` | Enable debug logging | `false` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Fast model downloads | `false` |

**Don't add:** API keys as env vars (use Stash settings instead for consistency)

---

## Recommendations Summary

### Settings to Keep Exposed (Tier 1-2)

1. All LLM provider settings (essential)
2. Excluded tags (privacy)
3. Recommendation weights (power users)
4. Performance tuning (power users)
5. Frame analysis settings (power users)

### Settings to Hide (Move to Code)

1. `vision_hosted_max_frames` → Derive from model capabilities
2. `frame_analysis_compare` → Remove, always use best method
3. Internal thresholds (deduplication, coverage) → Keep in code

### Settings to Add

1. `enable_recommendations` (bool) - Feature toggle
2. `enable_preference_learning` (bool) - Feature toggle
3. `enable_taste_map` (bool) - Feature toggle

### Implementation Priority

| Change | Effort | Impact | Priority |
|--------|--------|--------|----------|
| Typed configuration classes | Low | High | 1 |
| Centralize defaults | Low | Medium | 2 |
| Frontend config module | Low | Low | 3 |
| Reorganize yml categories | Low | Medium | 4 |
| Add feature toggles | Low | Medium | 5 |

## Decision Required

Please review and approve or suggest modifications to:

1. Tiered configuration structure
2. Settings to hide vs. expose
3. New feature toggle settings
4. Frontend configuration centralization
