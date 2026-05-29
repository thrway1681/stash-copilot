# Phase 0: Pre-Cleanup Checklist

**Date:** 2026-02-13
**Branch:** cleanup/productionalize-2026-02-13

## Type Checking

- **mypy:** NOT installed (to be added in Phase 2)
- **Current state:** No static type checking configured
- **Action required:** Add mypy as dev dependency, configure in pyproject.toml

## Test Suite

| Metric | Value |
|--------|-------|
| Test files | ~17 |
| Total test lines | ~4,411 |
| Coverage | TBD (pytest-cov not configured) |

**Test Structure:**
- `tests/conftest.py` - Shared fixtures (mock DB, mock Stash interface)
- `tests/fixtures/` - Schema and mock data generators
- `tests/tools/` - Database tool tests (6 files)
- `tests/tasks/` - Task handler tests (2 files)
- `tests/test_preferences.py` - Preference system tests

## Configuration Locations

### 1. Plugin Configuration (`stash-copilot.yml`)
Main plugin manifest with ~50 configurable settings:

| Category | Settings Count |
|----------|----------------|
| Text LLM | 4 (provider, model, base_url, api_key) |
| Vision LLM | 4 (separate from text LLM) |
| General | 1 (excluded_tags) |
| Vision Analysis | 5 (frame interval, min/max frames, debug) |
| Embedding | 8 (model, weight, provider, device, workers) |
| Frame Analysis | 6 (method, dynamic settings) |
| Recommendation | 6 (weights, time decay) |
| O-Moment | 3 (window, frames, tag name) |

### 2. Python Project (`pyproject.toml`)
- Project name: `stash-plugin-boilerplate` (needs rename)
- Python: >=3.10
- Dependencies: torch, open-clip, sentence-transformers, faiss, sklearn, etc.
- Optional deps: ollama, openai, anthropic, litellm, clip, openclip, siglip
- Dev deps: pytest, pytest-asyncio

### 3. Prompt Templates (`prompts/`)
```
prompts/
├── ask/system.yaml       # Agentic Q&A system prompt
├── chat/system.yaml      # Multi-turn chat prompt (17KB)
├── embed/visual_description.yaml
├── stats/summary.yaml    # Library summary generation
├── tags/suggestion.yaml  # Tag suggestion prompt
└── vision/
    ├── description.yaml  # Scene description prompt
    └── system.yaml       # Vision analysis system prompt
```

### 4. GitHub CI/CD (`.github/workflows/`)
- `claude-code-review.yml` - PR code review via Claude
- `claude.yml` - @claude mention triggers

**Missing CI/CD:**
- No linting (ruff)
- No type checking (mypy)
- No test runner
- No coverage reporting

## Performance-Critical Paths

Based on code structure, these are likely hotspots:

| Path | Description | Why Critical |
|------|-------------|--------------|
| `stash_ai/embeddings/provider.py` | Image embedding generation | GPU-intensive, batch processing |
| `stash_ai/tasks/embed_scenes.py` | Scene embedding pipeline | FFmpeg extraction + GPU inference |
| `stash_ai/tasks/frame_extractor.py` | Frame extraction | FFmpeg subprocess management |
| `stash_ai/embeddings/storage.py` | Embedding storage | Large BLOB read/write |
| `stash_ai/recommendations/engine.py` | Similarity search | Cosine similarity over embeddings |

## Database Files Found

| File | Size | Status |
|------|------|--------|
| `assets/stash_copilot.sqlite` | Active | Main embedding DB (ViT-H-14) |
| `assets/stash_copilot_ViT-bigG-14-dense.sqlite` | Active | Dense embeddings |
| `assets/stash_copilot_ViT-H-14-sparse.sqlite` | Active | Multiple models |
| `assets/embeddings.db` | TBD | Investigate usage |
| `stash_ai/embeddings/embeddings.db` | TBD | Investigate usage |
| `embeddings.db` (root) | 0 bytes | Empty, likely stale |

## Codebase Size

| Component | Files | Lines |
|-----------|-------|-------|
| stash-copilot.py | 1 | 3,275 |
| stash-copilot.js | 1 | 15,411 |
| stash-copilot.css | 1 | 11,156 |
| stash_ai/ package | 71 | TBD |
| tests/ | 17 | ~4,411 |
| **Total** | ~91 | ~34,000+ |

## Next Steps

1. Wait for exploration agents to complete
2. Document database schemas
3. Create current architecture diagram
4. Build feature dependency table
5. Interview user about feature inclusion
