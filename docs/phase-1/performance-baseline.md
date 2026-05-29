# Performance Baseline

**Generated:** 2026-02-15

## Overview

This document establishes performance baselines for the key features of Stash Copilot. These metrics serve as reference points for evaluating optimizations and regressions.

## Current System Configuration

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen (multiple cores) |
| GPU | NVIDIA with CUDA support |
| Storage | SSD (assets on local disk) |
| Python | 3.12 |
| Database | SQLite 3.x |
| Embedding Model | OpenCLIP ViT-H-14 |

## Feature Performance Metrics

### 1. Frame Embedding Generation

**Task:** Embed All Scenes

| Metric | Value | Notes |
|--------|-------|-------|
| Scenes Processed | 12,760 | Full library |
| Frames Extracted | 4,073,927 | At 1 fps |
| Avg Frames/Scene | 319 | Varies by duration |
| Database Size | 19 GB | After full embedding |
| Sampling Rate | 1 fps | Configurable |

**Bottlenecks:**
- FFmpeg frame extraction (I/O)
- GPU embedding batch processing
- SQLite writes for large batches

**Parallelization:**
- `embed_num_workers`: Default 2 (configurable 1-8)
- `frame_extract_workers`: Default 4

---

### 2. Similarity Search

**Task:** Find Similar Scenes

| Metric | Value | Notes |
|--------|-------|-------|
| Query Time | <100ms | Single scene query |
| Scenes Searched | 12,812 | All scenes |
| Embedding Dimensions | 1024 | ViT-H-14 |
| Min Similarity Threshold | 0.5 | Configurable |

**Algorithm:** Cosine similarity (dot product of normalized vectors)

---

### 3. Recommendation Generation

**Task:** Get Recommendations

| Metric | Value | Notes |
|--------|-------|-------|
| Profile Build Time | ~500ms | Top 20 scenes |
| Similarity Search | ~100ms | Against all scenes |
| Total Response Time | <1s | End-to-end |
| Results Per Request | 24 | Paginated |

**Engagement Score Calculation:**
```
base_score = (o_count × 20) + (replays × 2) + (hours × 1) + (stars × 1.5)
```

---

### 4. Vision Analysis

**Task:** Scene Vision Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Frame Extraction | 2-5s | Depends on video length |
| Smart Selection | <1s | K-means on embeddings |
| VLM Inference | 5-30s | Model dependent |
| Total Time | 10-60s | Full analysis |

**Model-specific performance:**
| Model | Avg Time | Quality |
|-------|----------|---------|
| Gemma 3 27B | 15-20s | High |
| LLaVA 13B | 10-15s | Medium |
| GPT-4o | 5-10s | High |

---

### 5. Preference Learning

**Task:** Preference Training Session

| Metric | Value | Notes |
|--------|-------|-------|
| Model Update | <10ms | Per comparison |
| Pair Selection | <50ms | Information gain calc |
| Convergence | ~100-200 | Comparisons to converge |

**Algorithm:** Bradley-Terry with diagonal Laplace approximation
- Memory: O(d) where d = embedding dimensions
- Compute: O(d) per update

---

### 6. Taste Map Generation

**Task:** Build Taste Map

| Metric | Value | Notes |
|--------|-------|-------|
| UMAP Reduction | 5-30s | Depends on scene count |
| K-means Clustering | <5s | Optimal k by silhouette |
| Total Time | 10-60s | Full generation |
| Cached Load | <1s | After first generation |

---

### 7. Tag Gap Detection

**Task:** Detect Tag Gaps

| Metric | Value | Notes |
|--------|-------|-------|
| First Run | 30-120s | Computes all coverage |
| Cached Load | <5s | Uses precomputed table |
| Frame-Tag Comparisons | 4M+ | All frames vs all tags |

**Optimization:** Pre-computed `frame_tag_coverage` table

---

### 8. Semantic Search

**Task:** Search Scenes by Text

| Metric | Value | Notes |
|--------|-------|-------|
| Query Embedding | 50-100ms | Text to vector |
| Similarity Search | <100ms | Against scene embeddings |
| Frame-level Search | 500ms-2s | Against frame embeddings |
| Results | 240 | Pre-fetched for pagination |

---

## Database Performance

### Table Sizes

| Table | Rows | Est. Size |
|-------|------|-----------|
| frame_embeddings | 4,073,927 | ~15 GB |
| frame_tag_coverage | 4,073,278 | ~2 GB |
| scene_embeddings | 12,812 | ~500 MB |
| scene_umap_coords | 12,756 | ~50 MB |
| frame_embedding_metadata | 12,362 | ~20 MB |
| tag_embeddings | 507 | ~5 MB |
| performer_embeddings | 313 | ~5 MB |
| preference_comparisons | 1,296 | <1 MB |

### Index Performance

Key indexes:
- `idx_frame_emb_scene_model` - Frame lookups by scene
- `idx_frame_tag_coverage_scene` - Coverage by scene
- `idx_frame_tag_coverage_uncovered` - Find uncovered frames
- `idx_performer_embedding_performer_id` - Performer lookups

---

## Frontend Performance

### JavaScript Bundle

| Metric | Value |
|--------|-------|
| Lines of Code | 15,411 |
| Load Time | <100ms |
| Memory Usage | ~50-100 MB |

### CSS Bundle

| Metric | Value |
|--------|-------|
| Lines of Code | 11,156 |
| Class Definitions | 1,521 |

### State Objects

| Object | Purpose | Memory |
|--------|---------|--------|
| `state` | Global plugin state | Light |
| `visionState` | Vision workflow | Light |
| `searchState` | Search results | Medium (caches results) |
| `similarState` | Similar results | Medium (caches results) |
| `sceneRecsState` | Recommendations | Medium |
| `preferenceState` | Training state | Light |

---

## Polling Patterns

All long-running tasks use polling for results:

| Task | Interval | Max Wait |
|------|----------|----------|
| Vision Analysis | 200ms | 60s |
| Similar Scenes | 200ms | 60s |
| Recommendations | 200ms | 60s |
| Taste Map | 200ms | 120s |
| Tag Gaps | 200ms | 120s |
| Search | 200ms | 60s |

---

## Memory Considerations

### GPU Memory (Embedding Generation)

| Model | VRAM | Batch Size |
|-------|------|------------|
| ViT-H-14 | ~4 GB | 32 |
| ViT-bigG-14 | ~8 GB | 16 |
| SigLIP | ~2 GB | 64 |

### CPU Memory (Similarity Search)

Loading all embeddings into memory:
- 12,812 scenes × 1024 dims × 4 bytes = ~52 MB
- Frame-level: ~16 GB (not loaded entirely)

---

## Optimization Notes

### Currently Implemented

1. **Batch Processing:** Frame embeddings processed in batches of 32
2. **Parallel Workers:** Configurable scene-level parallelism
3. **Frame Deduplication:** Skip similar consecutive frames
4. **Pre-computation:** frame_tag_coverage table
5. **Pagination:** Server-side limiting for large result sets
6. **Pre-fetching:** 10 pages (240 results) for search

### Potential Improvements

1. FAISS index for frame-level search (partially implemented)
2. Incremental embedding updates (only new/changed scenes)
3. Streaming embeddings for very large libraries
4. WebSocket-based progress instead of polling

---

## Test Suite Performance

| Metric | Value |
|--------|-------|
| Total Tests | 247 |
| Pass Rate | 100% |
| Coverage | 29% |
| Run Time | ~60s |

---

## Baseline Summary

| Feature | Performance | Status |
|---------|-------------|--------|
| Frame Embedding | 12K scenes, 4M frames embedded | ✅ Complete |
| Similarity Search | <100ms query time | ✅ Optimized |
| Recommendations | <1s response time | ✅ Optimized |
| Vision Analysis | 10-60s per scene | ⚠️ Model-dependent |
| Preference Learning | <10ms per update | ✅ Optimized |
| Taste Map | 10-60s generation | ⚠️ UMAP bottleneck |
| Tag Gap Detection | <5s with cache | ✅ Pre-computed |
| Semantic Search | <200ms scene-level | ✅ Optimized |
