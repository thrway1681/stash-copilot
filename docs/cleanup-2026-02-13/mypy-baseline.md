# Mypy Baseline Report

**Date:** 2026-02-13
**Mypy Version:** 1.19.1
**Python Version:** 3.10

## Summary

| Metric | Initial | Current |
|--------|---------|---------|
| Total errors | 234 | 76 |
| Files with errors | 43 | 19 |
| Files checked | 72 | 72 |

**Progress:** 158 errors fixed (68% reduction)

## Error Categories

### 1. Missing Type Annotations (High Priority)

| Error Code | Count | Description |
|------------|-------|-------------|
| `no-untyped-def` | ~50+ | Functions missing return type annotations |
| `type-arg` | ~25+ | Generic types without parameters (`set`, `dict`, `list`) |
| `no-any-return` | ~20+ | Functions returning `Any` instead of specific type |

**Fix approach:** Add explicit type annotations progressively.

### 2. Type Mismatches

| Error Code | Count | Description |
|------------|-------|-------------|
| `assignment` | ~10+ | Type mismatch in assignments |
| `return-value` | ~5+ | Return type doesn't match annotation |
| `union-attr` | ~5+ | Accessing attribute on possibly-None value |

**Fix approach:** Add proper type guards, Optional handling.

### 3. Import Issues (Resolved)

| Library | Status |
|---------|--------|
| yaml | ✅ Fixed with `types-PyYAML` |
| requests | ✅ Fixed with `types-requests` |

## Top Files by Error Count

Based on error output, these files need the most attention:

1. `stash_ai/tools/database.py` - Generic type parameters, variable redefinition
2. `stash-copilot.py` - Missing return types on plugin classes
3. `stash_ai/llm/providers/*.py` - Return type issues, kwargs handling
4. `stash_ai/tasks/*.py` - Missing type annotations

## Configuration

The following mypy configuration was added to `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.10"
strict = true
files = ["stash_ai", "stash-copilot.py"]
exclude = ["^tests/", "^\\.venv/", "^venv/", "^assets/"]

# Third-party libraries without stubs
[[tool.mypy.overrides]]
module = ["stashapi.*", "open_clip.*", "faiss.*", ...]
ignore_missing_imports = true
```

## Recommended Fix Order

1. **Phase 2a:** Fix high-impact, low-effort errors
   - Add `-> None` to functions that don't return
   - Add type parameters to generics (`set[int]`, `dict[str, Any]`)

2. **Phase 2b:** Fix function signatures
   - Add return type annotations
   - Add parameter type annotations for `**kwargs`

3. **Phase 2c:** Fix complex type issues
   - TypedDict for dictionary structures
   - Proper Optional handling

## Running Mypy

```bash
# Run type checking
uv run mypy stash_ai stash-copilot.py

# Run with specific error codes only
uv run mypy stash_ai stash-copilot.py --disable-error-code=no-any-return
```
