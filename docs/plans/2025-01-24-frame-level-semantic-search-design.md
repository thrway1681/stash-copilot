# Frame-Level Semantic Search Design

**Date:** 2025-01-24
**Status:** Approved

## Overview

Add frame-level semantic search to the existing AI Semantic Search feature. This enables finding scenes that contain specific poses, actions, or moments—even if they appear briefly—by searching against individual frame embeddings rather than scene-level composites.

## Goals

1. Find specific moments/poses in scenes, even if they appear briefly
2. Potentially replace scene-level search if frame-level produces better results
3. Maintain fast, interactive search latency (<5 seconds)

## Architecture

### Data Flow

```
User Query
    │
    ├─► [Toggle OFF] Scene Search (existing)
    │       └─► Query scene_embeddings table
    │       └─► Return scenes by composite similarity
    │
    └─► [Toggle ON] Frame Search (new)
            └─► Query FAISS index of frame embeddings
            └─► Get top-K matching frames across all scenes
            └─► Aggregate to scenes using max-score
            └─► Return scenes with best-matching frame info
```

### New Components

| Component | Path | Description |
|-----------|------|-------------|
| Frame Search Module | `stash_ai/embeddings/frame_search.py` | FAISS index management and search |
| Index File | `assets/frame_search_{model_key}.index` | FAISS index binary |
| Index Metadata | `assets/frame_search_{model_key}_meta.json` | Vector ID → frame mapping |

### Key Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Aggregation | Max-score | Duration-agnostic; finds brief moments |
| Index Type | FAISS IndexFlatIP | Exact search, normalized embeddings |
| Index Building | Manual task | User control, simple implementation |
| Device | CPU (GPU upgrade path available) | Sufficient for ~4M vectors |

## Backend Implementation

### New File: `stash_ai/embeddings/frame_search.py`

```python
from typing import TypedDict
import numpy as np

class FrameMetadata(TypedDict):
    scene_id: int
    frame_index: int
    timestamp: float

class FrameMatch(TypedDict):
    scene_id: int
    frame_index: int
    timestamp: float
    similarity: float

class SceneMatch(TypedDict):
    scene_id: int
    best_frame_index: int
    best_timestamp: float
    similarity: float

class FrameSearchIndex:
    """FAISS-based frame-level search index."""

    def __init__(self, index_path: str, meta_path: str):
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[FrameMetadata] = []

    def build(self, storage: EmbeddingStorage, model_key: str) -> None:
        """Build index from all frame_embeddings for given model."""
        pass

    def load(self) -> None:
        """Load index into memory (lazy, on first search)."""
        pass

    def search(self, query_embedding: np.ndarray, top_k: int = 1000) -> list[FrameMatch]:
        """Find top-K matching frames across all scenes."""
        pass

    def aggregate_to_scenes(self, frame_matches: list[FrameMatch]) -> list[SceneMatch]:
        """Aggregate frame matches to scene-level using max-score."""
        pass
```

### New Task: Build Frame Search Index

**Registration** (`stash-copilot.yml`):
```yaml
- name: Build Frame Search Index
  description: Build FAISS index for frame-level semantic search
  defaultArgs:
    mode: build_frame_index
    model_key: ""
```

**Build Process:**
1. Query all frame embeddings for model_key from `frame_embeddings` table
2. Load embeddings in batches (10,000 at a time) to manage memory
3. Stack into numpy array, normalize if needed
4. Create `faiss.IndexFlatIP` and add vectors
5. Save index to `assets/frame_search_{model_key}.index`
6. Save metadata to `assets/frame_search_{model_key}_meta.json`
7. Report progress throughout

### Index Metadata Structure

```json
{
  "model_key": "siglip",
  "frame_count": 4012620,
  "scene_count": 12660,
  "dimensions": 768,
  "created_at": "2025-01-24T10:30:00Z",
  "frames": [
    {"scene_id": 1, "frame_index": 0, "timestamp": 0.0},
    {"scene_id": 1, "frame_index": 1, "timestamp": 1.0}
  ]
}
```

### Search Response Format

```json
{
  "status": "complete",
  "query": "blonde in red lingerie",
  "model_key": "siglip",
  "frame_search": true,
  "results": [
    {
      "scene_id": 123,
      "similarity": 0.87,
      "best_frame_index": 154,
      "best_timestamp": 154.0,
      "frame_path": "embedded_frames/scene_123/frame_0154.jpg",
      "scene": { "id": 123, "title": "...", "performers": [...] }
    }
  ]
}
```

## Frontend Implementation

### Search Page Toggle

Add toggle next to model selector:

```
[Search box: "query"________________________] [🔍]

[Model: siglip ▾]  [○ Scene Search  ● Frame Search]
```

### State Update

```javascript
const semanticSearchState = {
    // ... existing fields
    frameSearch: false,  // NEW: toggle state
};
```

### Result Card Changes

When `frameSearch` is enabled:

1. **Thumbnail:** Load from `assets/embedded_frames/scene_{id}/frame_{index}.jpg`
2. **Timestamp badge:** Display "Best match at 2:34" overlay
3. **Fallback:** Use scene default thumbnail if frame image missing

```html
<div class="stash-copilot-card" data-theme="similar">
  <div class="card-thumbnail">
    <img src="/plugin/stash-copilot/assets/embedded_frames/scene_123/frame_0154.jpg">
    <span class="match-timestamp">2:34</span>
  </div>
  <!-- rest unchanged -->
</div>
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| Index not built | Return error: "Frame search index not built. Run 'Build Frame Search Index' task first." |
| Model mismatch | Return error: "Frame index not available for model X. Available: siglip" |
| Stale index | User must manually rebuild (no auto-detection) |
| Missing frame image | Fallback to scene default thumbnail, log warning |
| Empty results | Same as current: "No matching scenes found" |

## Resource Estimates

| Resource | Estimate |
|----------|----------|
| Index file size | ~12GB (4M vectors × 768 dims × 4 bytes) |
| Metadata JSON | ~200MB |
| Memory (loaded) | ~12GB |
| Search latency | 100-500ms (CPU) |

## Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `faiss-cpu` dependency |
| `stash-copilot.yml` | Add "Build Frame Search Index" task |
| `stash-copilot.py` | Add `build_frame_index` and update `search_by_text` handlers |
| `stash-copilot.js` | Add toggle, update result rendering |
| `stash-copilot.css` | Timestamp badge styling |

## New Files

| File | Purpose |
|------|---------|
| `stash_ai/embeddings/frame_search.py` | FAISS index management and search logic |

## Future Enhancements

- GPU support via `faiss-gpu` (simple code change)
- Auto-rebuild index after embedding task
- UI indicator showing index freshness
- Config option: `frame_search_device: cpu | gpu | auto`
