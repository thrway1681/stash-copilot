# Phase 0: Pre-cleanup Checklist

**Completed:** 2026-02-15

## Type Checking Status

**Tool:** mypy (strict mode)
**Result:** ✅ 0 errors

```
Success: no issues found in 71 source files
```

## Test Suite Status

**Tool:** pytest with pytest-cov
**Result:** ✅ 247 tests passed

```
Coverage Summary:
- Total lines: ~40,740
- Lines covered: ~11,800
- Coverage: 29%
```

Key coverage by module:
| Module | Coverage |
|--------|----------|
| stash_ai/llm | 45% |
| stash_ai/embeddings | 38% |
| stash_ai/recommendations | 32% |
| stash_ai/tasks | 25% |
| stash_ai/tools | 20% |

## Configuration & Settings Locations

### 1. Plugin Settings (Stash UI)

**File:** `stash-copilot.yml`
**Access:** Settings → Plugins → Stash Copilot

| Category | Settings Count | Examples |
|----------|---------------|----------|
| Text LLM | 4 | provider, model, base_url, api_key |
| Vision LLM | 4 | provider, model, base_url, api_key |
| General | 1 | excluded_tags |
| Vision Analysis | 6 | auto_analyze, frame_interval, min/max frames, debug, hosted_max_frames |
| Embeddings | 7 | embedding_model, visual_weight, image_embedding_provider/model/device |
| Frame Analysis | 6 | method, n_frames, dynamic, frames_per_minute, min/max_frames, compare |
| Performance | 2 | embed_num_workers, frame_extract_workers |
| Recommendations | 6 | top_scenes, o_weight, view_weight, duration_weight, rating_weight, time_decay_days |
| O-Moments | 3 | window, frames, tag_name |

**Total:** 39 configurable settings via Stash UI

### 2. Build & Tool Configuration

**File:** `pyproject.toml`

| Section | Purpose |
|---------|---------|
| `[project]` | Package metadata, dependencies |
| `[tool.mypy]` | Type checking (strict mode) |
| `[tool.ruff]` | Linting & formatting |
| `[tool.pytest]` | Test configuration |
| `[tool.uv]` | UV package manager, PyTorch CUDA indexes |

### 3. LLM Prompt Templates

**Directory:** `prompts/`

| File | Purpose |
|------|---------|
| `ask/system.yaml` | System prompt for "Ask AI" mode |
| `chat/system.yaml` | System prompt for multi-turn chat |
| `vision/system.yaml` | System prompt for vision analysis |
| `vision/description.yaml` | Scene description generation |
| `tags/suggestion.yaml` | Tag suggestion prompts |
| `stats/summary.yaml` | Library statistics summary |
| `embed/visual_description.yaml` | Visual description for embeddings |

### 4. Database Configuration

**Primary Database:** `assets/stash_copilot.sqlite`
- Schema version: 10
- Model key scoping for multi-model support
- Auto-migration on startup

**Stale/Experimental Databases:**
- `assets/stash_copilot_ViT-bigG-14-dense.sqlite` (v4, last: 2026-01-18)
- `assets/stash_copilot_ViT-H-14-sparse.sqlite` (v4, last: 2026-01-04)

### 5. Runtime Configuration

**Environment Variables:**
| Variable | Purpose |
|----------|---------|
| `STASH_COPILOT_DEBUG` | Enable debug logging for LLM providers |
| `HF_HUB_ENABLE_HF_TRANSFER` | Fast Hugging Face downloads |

**Stash Connection:**
- Passed via stdin JSON (`server_connection` fragment)
- Auto-configured by StashInterface from stashapi library

### 6. Frontend State Persistence

**localStorage keys (JavaScript):**
| Key | Purpose |
|-----|---------|
| `stash-copilot-seed-weight` | Scene recs seed weight slider |
| `stash-copilot-engagement-weight` | Engagement vs similarity slider |
| `stash-copilot-sidebar-rec-decay` | Time decay setting |
| `stash-copilot-rec-mode` | Last selected recommendation mode |
| `stash-copilot-visual-weight` | Visual vs metadata weight |

### 7. Asset Output Files

**Directory:** `assets/`

| File Pattern | Purpose |
|--------------|---------|
| `last_summary.json` | Cached library summary |
| `chat_history.json` | Chat conversation history |
| `scene_vision/vision_history_*.json` | Scene vision analysis results |
| `recommendations_*.json` | Recommendation results |
| `taste_map_*.json` | Taste map visualization data |
| `tag_gaps_*.json` | Tag gap detection results |
| `search_results_*.json` | Search result caches |
| `embedding_models_*.json` | Available embedding models |
| `o_moment_stats.json` | O-moment statistics |

## Performance-Critical Paths

### 1. Frame Embedding Generation (HIGHEST IMPACT)

**Path:** `stash_ai/tasks/embed_scenes.py` → `DenseFrameExtractor` → `EmbeddingProvider`

**Bottlenecks:**
- FFmpeg frame extraction (I/O bound)
- GPU embedding generation (compute bound)
- SQLite writes for 4M+ frames (I/O bound)

**Optimizations in place:**
- Batch processing (32 frames default)
- Parallel workers (configurable: `embed_num_workers`)
- Frame deduplication (skip similar consecutive frames)

**Metrics:**
- ~300-500 frames/scene average
- 12,760 scenes embedded
- 4,073,927 total frames
- 19 GB database

### 2. Similarity Search (HIGH IMPACT)

**Path:** `EmbeddingStorage.find_similar_scenes()` → FAISS (optional) or SQLite

**Bottlenecks:**
- Loading all embeddings into memory
- Cosine similarity computation
- Sorting and ranking

**Optimizations in place:**
- FAISS index for frame-level search
- Pre-computed composite embeddings
- Pagination with server-side limits

### 3. Recommendation Generation (MEDIUM IMPACT)

**Path:** `RecommendationEngine.generate_recommendations()`

**Bottlenecks:**
- User profile building (top 20 scenes)
- Similarity computation against all scenes
- Engagement score calculation

**Optimizations in place:**
- Weighted embedding averaging
- Configurable profile size
- Cached scene details

### 4. Vision Analysis (MEDIUM IMPACT)

**Path:** `SceneVisionTask` → Frame extraction → VLM API

**Bottlenecks:**
- Frame extraction (FFmpeg)
- VLM inference (model dependent)
- Image encoding/transmission

**Optimizations in place:**
- Smart frame selection (representative frames)
- Grid mode for single-image models
- Model capability auto-detection

### 5. Tag Gap Detection (MEDIUM IMPACT)

**Path:** `TagGapDetectionTask` → Frame/tag similarity

**Bottlenecks:**
- Loading all frame embeddings
- Tag embedding comparison
- Coverage threshold calculation

**Optimizations in place:**
- Pre-computed frame_tag_coverage table
- Batch processing

### 6. Preference Learning (LOW IMPACT)

**Path:** `BayesianPreferenceModel.update_from_comparison()`

**Complexity:** O(d) where d = embedding dimensions
- Diagonal Laplace approximation (not O(d²))
- Incremental updates per comparison

## Summary

| Item | Status |
|------|--------|
| Type checking (mypy) | ✅ 0 errors |
| Test suite | ✅ 247 passed, 29% coverage |
| Config locations documented | ✅ 7 categories |
| Performance paths documented | ✅ 6 critical paths |

**Phase 0 Status: COMPLETE**
