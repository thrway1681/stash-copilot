# Phase 5: Duplicate Code Consolidation

**Generated:** 2026-02-15

## Summary

This document tracks duplicate code identified and consolidated during Phase 5.

## JavaScript Duplicates

### 1. escapeHtml Function

**Status:** ✅ Consolidated

**Locations Found:**
- Line 2813: Primary definition with null check
- Line 8427: Duplicate definition (removed)
- Line 4615: `escapeHtmlChars` - regex-based alternative

**Resolution:**
- Removed duplicate at line 8427
- Keep line 2813 as the canonical `escapeHtml`
- Keep `escapeHtmlChars` as alternative (uses regex instead of DOM)

### 2. formatDuration Functions

**Status:** ✅ Not duplicates (different purposes)

**Analysis:**
| Location | Function | Output Format | Purpose |
|----------|----------|---------------|---------|
| Line 645 | `formatDuration()` | `"2h 30m"` | Human-readable summary |
| Line 2948 | `SceneCardUtils.formatDuration()` | `"2:30:45"` | Video timestamp display |
| Line 4603 | `formatDurationMs()` | `"Xh Ym"` | Milliseconds to human |

**Resolution:** These serve different formatting needs. Not consolidated.

**Recommendation:** Consider renaming for clarity:
- `formatDuration` → `formatDurationHuman`
- `SceneCardUtils.formatDuration` → Keep as-is (context makes purpose clear)

### 3. formatRating Functions

**Status:** ✅ Consolidated into SceneCardUtils

**Locations:**
- `SceneCardUtils.formatRating()` - Canonical implementation

**Resolution:** Already consolidated in previous cleanup.

## Python Duplicates

### 1. Model Capability Definitions

**Status:** ✅ Already using shared constant

**Location:** `stash_ai/llm/model_caps.py`

```python
# Shared capability definitions to avoid duplication
_GEMMA3_CAPS = VLMCapabilities(...)

MODEL_CAPABILITIES = {
    "gemma-3": _GEMMA3_CAPS,
    "gemma3": _GEMMA3_CAPS,  # Alias
}
```

**Resolution:** Already DRY - both keys reference same constant.

### 2. Exception Handling Patterns

**Status:** 📋 Documented for future refactoring

**Findings:** 85+ instances of `except Exception` across codebase

**Recommendation:** Create typed exception hierarchy (documented in Phase 3 architecture proposal):

```python
class StashCopilotError(Exception): pass
class LLMError(StashCopilotError): pass
class EmbeddingError(StashCopilotError): pass
class StorageError(StashCopilotError): pass
```

**Priority:** Low - functional but masks specific errors

### 3. LLM Provider Error Handling

**Status:** 📋 Documented for future refactoring

**Pattern observed in providers:**
```python
try:
    response = requests.post(...)
except requests.RequestException as e:
    raise LLMError(f"Network error: {e}")
except json.JSONDecodeError as e:
    raise LLMError(f"Invalid JSON: {e}")
```

**Recommendation:** Extract to helper:
```python
def _handle_request(url, data, timeout):
    """Common HTTP request handling for LLM providers."""
    ...
```

## CSS Duplicates

### Analysis

The CSS file (11,156 lines) was analyzed for duplicates in the previous cleanup session. Major duplicate styles (Similar Modal, Scene Recs Modal) were removed, saving ~1,164 lines.

**Remaining opportunities:**
- CSS custom properties could be more widely used
- Some color values are hardcoded instead of using variables

## Consolidation Summary

| Category | Duplicates Found | Consolidated | Remaining |
|----------|------------------|--------------|-----------|
| JS escapeHtml | 2 | 1 | 0 |
| JS formatDuration | 3 | 0 (different purposes) | N/A |
| Python model caps | 2 | 0 (already shared) | 0 |
| Exception handling | 85+ | 0 | 85+ (future work) |

## Lines Changed

| File | Lines Removed | Notes |
|------|---------------|-------|
| stash-copilot.js | 5 | Duplicate escapeHtml |

## Recommendations for Future

1. **Exception Handling:** Implement typed exceptions per Phase 3 proposal
2. **Utility Module:** Consider extracting shared utilities to separate module
3. **CSS Variables:** Expand use of CSS custom properties for colors/spacing

## Verification

Run tests to verify no regressions:
```bash
uv run pytest
```
