# Tag Gap Detection - Design Document

**Date:** 2026-02-13
**Goal:** Detect visual content in scenes that isn't covered by any existing tag, flag those scenes, and provide helpful context so the user can decide what new tags to create.

## Approach

**Novelty detection via frame-level embeddings.** For each frame in a scene, compare its OpenCLIP image embedding against all tag text embeddings. If no tag is a close match, that frame represents "uncovered" content. If enough frames in a scene are uncovered, the scene is flagged.

No LLM/VLM calls. No large curated vocabulary. No tag name suggestions - the system flags gaps and provides visual + semantic context; the user decides what to name the tag.

## Core Algorithm

### Per-Frame Gap Detection

1. Ensure all Stash tags have text embeddings via `TagVocabulary` + OpenCLIP `encode_text()`
2. For each frame embedding in a scene, compute cosine similarity against every tag embedding
3. Record the best-matching tag and its similarity score
4. If the best similarity falls below an adaptive threshold, mark the frame as "uncovered"

### Adaptive Threshold

Rather than a hard cutoff (OpenCLIP image-to-text cosine similarity for short tag names typically ranges 0.15-0.35), use a **percentile-based approach**:

- Compute the distribution of best-match scores across all frames in the library
- Flag frames in the bottom quartile as uncovered
- This auto-adapts to the user's tag vocabulary size

### Scene Coverage Score

- **Coverage ratio** = covered frames / total frames
- Scenes ranked by coverage ratio (lowest = most uncovered content)

### Cross-Scene Similarity (On-Demand)

No pre-computed clusters (avoids staleness when scenes/tags change). Instead:

1. Take the uncovered frames' embeddings for the current scene
2. Average them into an "uncovered concept" vector
3. Compare against other scenes' averaged uncovered frame embeddings
4. Return the most similar scenes - "these scenes share uncovered content"

Computed at query time, always fresh.

## Storage

### New SQLite Table

Single new table in `assets/stash_copilot.sqlite`:

```sql
CREATE TABLE frame_tag_coverage (
    scene_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    model_key TEXT NOT NULL,
    best_tag TEXT NOT NULL,
    best_similarity REAL NOT NULL,
    is_covered BOOLEAN NOT NULL,
    PRIMARY KEY (scene_id, frame_index, model_key)
);

CREATE INDEX idx_frame_tag_coverage_scene ON frame_tag_coverage(scene_id, model_key);
CREATE INDEX idx_frame_tag_coverage_uncovered ON frame_tag_coverage(is_covered, model_key);
```

### Query Patterns

- **Per-scene uncovered frames:** `SELECT * FROM frame_tag_coverage WHERE scene_id = ? AND NOT is_covered AND model_key = ?`
- **Bulk report (flagged scenes):** `SELECT scene_id, COUNT(*) as total, SUM(CASE WHEN NOT is_covered THEN 1 ELSE 0 END) as uncovered FROM frame_tag_coverage WHERE model_key = ? GROUP BY scene_id ORDER BY CAST(uncovered AS REAL) / total DESC`
- **Uncovered frame embeddings for similarity:** Join with `frame_embeddings` table on (scene_id, frame_index, model_key)

## Backend

### New File: `stash_ai/tasks/tag_gap_detection.py`

**Class:** `TagGapDetectionTask`

**Bulk task flow ("Detect Tag Gaps"):**

1. Ensure tag embeddings via `TagVocabulary.ensure_embeddings()`
2. Load all tag embeddings into a matrix (N_tags x 1024)
3. Iterate all scenes with frame embeddings, with progress reporting:
   - Load frame embeddings from `frame_embeddings` table
   - Batch cosine similarity: each frame against all tag embeddings
   - Record best tag + similarity per frame
4. Compute adaptive threshold (bottom quartile of all best-similarity scores)
5. Write `frame_tag_coverage` rows
6. Report summary (total scenes, average coverage, flagged count)

**Incremental support:**
- Skip scenes already in `frame_tag_coverage` unless `force_recompute=True`
- When new tags are added, recompute threshold + `is_covered` flags only

### New Method in `EmbeddingStorage`

`find_similar_uncovered_scenes(scene_id, model_key, limit)`:
- Average uncovered frame embeddings for the query scene
- Compare against other scenes' averaged uncovered embeddings
- Return ranked scene list with similarity scores

### Storage Methods

- `save_frame_tag_coverage(rows: List[FrameTagCoverage])` - Batch insert/replace
- `get_scene_tag_coverage(scene_id, model_key)` - Per-scene query
- `get_coverage_summary(model_key)` - Bulk report (all scenes ranked)
- `get_uncovered_frame_embeddings(scene_id, model_key)` - Join with frame_embeddings for similarity search

## Frontend

### AI Insights Modal: "Tag Gaps" Tab

New tab in the modal tab bar alongside Summary, Chat, Tools, Recs, Peak, Taste Map, Train.

**Three states:**

#### Empty State (no analysis run)
- Coverage stats placeholder ("X scenes embedded, X tags in library")
- Description: "Detect visual content in your scenes that isn't covered by any existing tag"
- **"Detect Tag Gaps" button**
- Tip text

#### Running State (task in progress)
- Progress bar: "Analyzing scene 47 / 312..."
- Animated spinner
- Button disabled

#### Results State (analysis complete)
- **Summary stats:** "87% average coverage. 45 scenes have uncovered content."
- **Scene list:** Ranked by coverage ratio (lowest first), each row shows:
  - Scene thumbnail, title, coverage bar (e.g., "62% covered")
  - Nearest tags for uncovered frames (faded, as hints)
- **Click a scene:** Navigates to scene page where sidebar shows full detail
- **Re-run button** to refresh analysis

### Scene Page Sidebar: "Gaps" Tab

New 4th sidebar tab alongside Analyze / Similar / Recs.

#### When data exists:
- **Coverage bar:** Visual bar showing "68% covered" (green -> yellow -> red gradient)
- **Uncovered frames strip:** Horizontal scrollable row of frame thumbnails with timestamps
- **Nearest tags per frame:** Top 2-3 closest tags with similarity scores (e.g., "closest: *outdoor* 0.19, *solo* 0.17")
- **Similar uncovered scenes:** Compact scene cards (unified card system) showing scenes with similar uncovered content, computed on-the-fly

#### When no data exists:
- "Run **Detect Tag Gaps** from AI Insights to analyze this scene"
- Button/link to open AI Insights modal Tag Gaps tab

## Design Principles

- **No automation of tag creation** - system flags gaps, user decides
- **No LLM/VLM calls** - purely embedding-based, fast and free
- **No stored clusters** - on-demand similarity avoids staleness
- **Incremental** - only processes new/changed scenes
- **Adaptive threshold** - auto-adjusts to vocabulary size
