# Phase 6: Module Restructure Plan

**Generated:** 2026-02-15

## Overview

Based on the approved architecture (Phase 3), this document outlines the module restructuring approach. Given the scope, this will be done incrementally to minimize risk.

## Current State

### Python Backend
- 71 files in `stash_ai/`
- Monolithic task handlers
- Direct storage access from tasks
- No service layer

### JavaScript Frontend
- Single 13,500+ line file
- Multiple state objects
- Mixed feature code

### CSS
- Single 10,000+ line file
- Feature styles mixed together

## Phase 6 Implementation Strategy

### Priority 1: Python Service Layer (Immediate)

Create service abstractions without breaking existing code:

```
stash_ai/
├── services/
│   ├── __init__.py
│   ├── embedding_service.py    # Extract from embed_scenes.py
│   └── recommendation_service.py # Extract from recommendations.py
```

**Approach:** Wrap existing logic in service classes, then refactor tasks to use services.

### Priority 2: Python Exception Hierarchy (Immediate)

```
stash_ai/
├── exceptions.py               # New file
```

```python
class StashCopilotError(Exception):
    """Base exception for all plugin errors."""
    pass

class LLMError(StashCopilotError):
    """LLM provider errors."""
    pass

class EmbeddingError(StashCopilotError):
    """Embedding generation errors."""
    pass

class StorageError(StashCopilotError):
    """Database operation errors."""
    pass
```

### Priority 3: Configuration Classes (Short-term)

```
stash_ai/
├── config/
│   ├── __init__.py
│   ├── settings.py             # Typed configuration
│   └── defaults.py             # Centralized defaults
```

### Priority 4: Frontend Module Split (Requires Build System)

**Decision Required:** Implement esbuild for frontend bundling?

If yes:
```
frontend/
├── core/
│   ├── api.js
│   ├── state.js
│   └── utils.js
├── components/
│   ├── cards.js
│   └── modals.js
├── features/
│   ├── vision/
│   ├── similar/
│   └── recommendations/
└── index.js
```

If no (keep monolithic):
- Add section comments for better organization
- Extract shared utilities to top of file

## Implementation Log

### Step 1: Create Exception Hierarchy

**File:** `stash_ai/exceptions.py`

```python
"""Custom exceptions for Stash Copilot."""

class StashCopilotError(Exception):
    """Base exception for all plugin errors."""
    pass

class LLMError(StashCopilotError):
    """LLM provider errors (network, API, parsing)."""
    pass

class VisionError(LLMError):
    """Vision analysis specific errors."""
    pass

class EmbeddingError(StashCopilotError):
    """Embedding generation or storage errors."""
    pass

class StorageError(StashCopilotError):
    """Database operation errors."""
    pass

class ConfigurationError(StashCopilotError):
    """Invalid configuration or missing settings."""
    pass

class TaskError(StashCopilotError):
    """Task execution errors."""
    pass
```

### Step 2: Create Configuration Module

**File:** `stash_ai/config/__init__.py`

```python
from .settings import PluginConfig, LLMSettings, RecommendationSettings

__all__ = ["PluginConfig", "LLMSettings", "RecommendationSettings"]
```

**File:** `stash_ai/config/settings.py`

(Implementation per Phase 4 proposal)

### Step 3: JavaScript Organization (No Build System)

Add section markers to `stash-copilot.js`:

```javascript
// ============================================================
// SECTION: Core Utilities
// ============================================================

// ============================================================
// SECTION: State Management
// ============================================================

// ============================================================
// SECTION: API & GraphQL
// ============================================================

// ============================================================
// SECTION: Components - Cards
// ============================================================

// ============================================================
// SECTION: Components - Modals
// ============================================================

// ============================================================
// SECTION: Feature - Vision Analysis
// ============================================================

// etc.
```

## What Was Implemented

1. ✅ Exception hierarchy (`stash_ai/exceptions.py`) - COMPLETE
2. ✅ Configuration module (`stash_ai/config/`) - COMPLETE
3. ⏸️ Service layer - Deferred (larger refactoring)
4. ⏸️ Frontend split - Deferred (requires build system setup)

### Implementation Details

**Exception Hierarchy** (`stash_ai/exceptions.py`):
- `StashCopilotError` - Base exception with message and details
- `LLMError` - LLM provider errors with provider/model context
- `VisionError` - Vision-specific errors (extends LLMError)
- `EmbeddingError` - Embedding errors with model_key/scene_id context
- `StorageError` - Database errors with operation/table context
- `ConfigurationError` - Config errors with setting/expected/actual context
- `TaskError` - Task execution errors with task_name/scene_id context

**Configuration Module** (`stash_ai/config/`):
- `defaults.py` - Centralized default values (single source of truth)
- `settings.py` - Typed configuration dataclasses with validation
- `legacy.py` - Backwards-compatible exports (LLMConfig, LLMSettings)
- `__init__.py` - Public API with both legacy and new exports

**Configuration Classes Created**:
- `LLMProviderConfig` - LLM provider settings (new typed version)
- `EmbeddingSettings` - Image embedding configuration
- `RecommendationSettings` - Recommendation engine weights
- `VisionSettings` - Vision analysis parameters
- `FrameSettings` - Frame extraction settings
- `OMomentSettings` - O-moment embedding settings
- `PerformanceSettings` - Worker/threading settings
- `PluginConfig` - Complete plugin configuration container

## Verification

After each change:
1. Run `uv run pytest` - All tests must pass
2. Run `uv run mypy stash_ai` - Type checking must pass
3. Manual test of key features if structural changes made

## Risks

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Small incremental changes, tests after each |
| Import cycles | Careful dependency ordering |
| Missing exports | Update `__init__.py` files |

## Future Work (Post-Phase 6)

1. Complete service layer implementation
2. Evaluate esbuild for frontend bundling
3. Repository pattern for database access
4. CSS module splitting with build system
