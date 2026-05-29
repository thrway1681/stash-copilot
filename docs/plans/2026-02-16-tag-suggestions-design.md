# Embedding-Based Tag Suggestions - Design Document

**Date:** 2026-02-16
**Goal:** Suggest tags for scenes based on frame-to-tag embedding similarity, showing visual evidence (matching frame thumbnails) to help users approve or dismiss suggestions.

## Overview

A new "Tags" tab in the scene sidebar that:
1. Computes tag suggestions using frame embeddings matched against tag embeddings
2. Shows top 5 matching frames as evidence for each suggested tag
3. Allows users to apply tags immediately or dismiss them (remembered per-scene)

No LLM/VLM calls required. Pure embedding-based similarity matching.

## Algorithm: Frame-Centric Voting

### Core Flow

1. **Load data:**
   - Frame embeddings for the scene (N frames)
   - Tag embeddings for all Stash tags (T tags)
   - Existing scene tags (to exclude)
   - Dismissed tags for this scene (to exclude)

2. **Compute similarity matrix:**
   ```
   similarity_matrix[N×T] = frame_embeddings @ tag_embeddings.T
   ```

3. **Aggregate votes per tag:**
   - Count frames with similarity ≥ 0.30
   - Record top-5 frame indices (evidence)
   - Compute max similarity (primary ranking)
   - Compute mean similarity (secondary)

4. **Rank and filter:**
   - Primary sort: frame_count (more frames = stronger signal)
   - Secondary sort: max_similarity (strongest single match)
   - Filter: max_similarity ≥ 0.30
   - Filter: not already on scene
   - Filter: not dismissed
   - Take top 20

### Output Structure

```python
@dataclass
class TagSuggestion:
    tag_id: int
    tag_name: str
    max_similarity: float      # Highest single-frame match
    mean_similarity: float     # Average across all matching frames
    frame_count: int           # Frames with similarity >= threshold
    evidence_frames: list[EvidenceFrame]  # Top 5 matching frames

@dataclass
class EvidenceFrame:
    frame_index: int
    similarity: float
    timestamp: str             # "MM:SS" format
    thumbnail_path: str        # Relative path to frame image
```

## Storage

### New Table: Dismissed Suggestions

```sql
CREATE TABLE dismissed_tag_suggestions (
    scene_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    dismissed_at TEXT NOT NULL,  -- ISO timestamp
    PRIMARY KEY (scene_id, tag_id)
);

CREATE INDEX idx_dismissed_scene ON dismissed_tag_suggestions(scene_id);
```

### Existing Tables Used (No Changes)

| Table | Purpose |
|-------|---------|
| `frame_embeddings` | Source frame vectors |
| `tag_embeddings` | Precomputed tag vectors |
| `scenes_tags` (Stash DB) | Existing scene tags (to exclude) |
| `tags` (Stash DB) | Tag names and IDs |

## Backend

### New File: `stash_ai/tasks/tag_suggestions.py`

**Class:** `TagSuggestionsTask`

**Methods:**
- `run(scene_id: int) -> TagSuggestionsResult` - Main computation
- `_load_embeddings()` - Load frame + tag embeddings
- `_compute_similarities()` - Batch matrix operation
- `_aggregate_votes()` - Count frames per tag
- `_build_evidence()` - Get top-5 frames per tag

### New Plugin Modes

| Mode | Description |
|------|-------------|
| `get_tag_suggestions` | Compute and return suggestions for scene |
| `apply_suggested_tag` | Add tag to scene via Stash GraphQL |
| `dismiss_suggested_tag` | Record dismissal in storage |
| `clear_dismissed_tags` | Clear all dismissals for scene |

### Storage Methods (in `EmbeddingStorage`)

```python
def save_dismissed_tag(self, scene_id: int, tag_id: int) -> None:
    """Record that a tag suggestion was dismissed."""

def get_dismissed_tags(self, scene_id: int) -> set[int]:
    """Get all dismissed tag IDs for a scene."""

def clear_dismissed_tags(self, scene_id: int) -> int:
    """Clear dismissals for a scene. Returns count deleted."""
```

## UI

### Location

New "Tags" tab in scene sidebar (4th tab alongside Analyze/Similar/Recs).

### Color Theme

**Cyan** (#06b6d4) - distinct from existing green/purple/orange themes.

### Layout (350-400px sidebar width)

```
┌─────────────────────────────────────┐
│  🏷️ Tags                            │
│  ─────────────────────────────────  │
│  [Suggest Tags]  [Clear Dismissed]  │
│                                     │
│  ┌─────────────────────────────────┐│
│  │ BLOWJOB                    82%  ││
│  │ ┌─────┬─────┬─────┬─────┬─────┐ ││
│  │ │ 🖼️  │ 🖼️  │ 🖼️  │ 🖼️  │ 🖼️  │ ││
│  │ │ 82% │ 79% │ 75% │ 71% │ 68% │ ││
│  │ └─────┴─────┴─────┴─────┴─────┘ ││
│  │ 12 frames matched               ││
│  │ [✓ Apply] [✕ Dismiss]           ││
│  └─────────────────────────────────┘│
│                                     │
│  ... more suggestions ...           │
│                                     │
│  ◀ 1 / 4 ▶                          │
└─────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Header** | Tab title with action buttons |
| **Suggest Tags button** | Triggers suggestion computation |
| **Clear Dismissed button** | Removes all dismissals for scene |
| **Suggestion Card** | One per suggested tag |
| **Tag name + score** | Name in caps, max similarity % |
| **Evidence thumbnails** | Top 5 frames, 60x40px, similarity % overlay |
| **Frame count** | "X frames matched" |
| **Apply button** | Green, adds tag immediately |
| **Dismiss button** | Muted red, hides and remembers |
| **Pagination** | 5 suggestions per page |

### Interactions

- **Thumbnail hover:** Show larger preview with timestamp
- **Thumbnail click:** Seek video to that frame timestamp
- **Apply click:** API call → tag added → card removed with animation
- **Dismiss click:** Recorded in storage → card removed with animation
- **High confidence (≥70%):** Subtle pulse on score badge

## Error Handling

| Scenario | Handling |
|----------|----------|
| No embeddings | "Scene needs embedding first" + Embed button |
| No tag embeddings | "Run tag embedding first" + Embed Tags button |
| No matching tags | "No tag suggestions found above threshold" |
| All tags dismissed | "All suggestions dismissed" + Clear button |
| Stash API error | Toast notification with error, retry button |
| Apply fails | Toast error, keep suggestion visible |

### Loading States

| State | UI |
|-------|-----|
| Computing | "Analyzing scene..." with spinner |
| Loading frames | Progress: "Loading frames (12/50)..." |
| Applying tag | Button disabled with spinner |

## Constraints

- **Tag sources:** Stash tags only (no curated vocabulary)
- **Scope:** Per-scene only (no library-wide batch)
- **Evidence:** Frame thumbnails only (no VLM reasoning)
- **Threshold:** ≥0.30 similarity, top 20 suggestions
- **Prerequisites:** Scene must have frame embeddings

## Files to Create/Modify

| File | Change |
|------|--------|
| `stash_ai/tasks/tag_suggestions.py` | **New** - Core suggestion logic |
| `stash_ai/embeddings/storage.py` | Add dismissed_tag_suggestions table + methods |
| `stash-copilot.py` | Add new task modes |
| `stash-copilot.js` | Add Tags tab UI |
| `stash-copilot.css` | Add Tags tab styles (cyan theme) |

## Out of Scope (YAGNI)

- Library-wide batch suggestions
- Tag creation from curated vocabulary
- Global tag blacklist
- VLM reasoning integration
- Cluster-based analysis
- Caching of suggestions
