# Find Similar Scenes by Current Frame

**Date:** 2026-03-17
**Status:** Design approved

## Summary

Add a "Search by Current Frame" button to the existing Similar sidebar tab on the scene page. When clicked, it extracts the frame at the current video playback position, embeds it with OpenCLIP, and searches the FAISS frame index for visually similar frames across the library. Results replace the existing similar-scenes list in-place, with a back button to restore the original results.

## Requirements

- Button lives inside the existing Similar sidebar tab, above the sub-tabs
- Extracts the frame at `video.currentTime` from the playing scene
- Embeds the frame using the configured image embedding provider (OpenCLIP ViT-bigG-14)
- Searches the pre-built FAISS frame index (`FrameSearchIndex`) for similar frames
- Results are frame-level (not scene-level composite), each showing the matched timestamp
- Results replace the existing similar results in-place with a "Back to Similar" toggle
- Requires the frame index to be built; shows a clear error if not
- Extracted frame is ephemeral (in-memory only, no disk caching)

## Architecture

### Python Backend

#### New task mode: `find_similar_by_frame`

Added to `run_task()` dispatch in `stash-copilot.py`. New method `run_find_similar_by_frame()`.

**Arguments** (all strings per existing convention):

| Arg | Type | Required | Default | Description |
|---|---|---|---|---|
| `scene_id` | str | yes | — | The scene currently playing |
| `timestamp` | str | yes | — | Seconds (float) from `video.currentTime` |
| `limit` | str | no | `"20"` | Max results |
| `request_id` | str | yes | — | For poll matching |

**Processing flow:**

1. Resolve video file path from Stash SQLite (Pattern A from `embed_scenes.py`: `folders.path || '/' || files.basename`)
2. Instantiate `FrameExtractor(config=FrameExtractionConfig(), cache_dir=assets_dir)` — minimal config since this is a one-shot extraction, not a cache-and-reuse scenario. The `cache_dir` points to the existing `assets/embedded_frames/` directory.
3. `extractor.extract_frame_at_timestamp(video_path, float(timestamp))` returns `bytes | None`. **If `None`, write error JSON `"Failed to extract frame at timestamp X"` and return early.**
4. Build `EmbeddingConfig` from plugin settings, call `get_embedding_provider(config).embed_image(frame_bytes)` returns `ImageEmbeddingResult`
5. Convert embedding to numpy: `query_embedding = np.array(result["embedding"], dtype=np.float32)` — required because `FrameSearchIndex.search()` expects `NDArray[np.float32]`, not `list[float]`. The embedding provider returns L2-normalized vectors when `config.normalize=True` (default), which is required for correct cosine similarity with the `IndexFlatIP` index.
6. Load `FrameSearchIndex` for the model key. If index file doesn't exist, write error JSON and return early.
7. `index.search(query_embedding, top_k=limit * 3)` returns `list[FrameMatch]` (over-fetch to allow for filtering)
8. Filter out frames belonging to the query `scene_id`
9. `aggregate_to_scenes()` to deduplicate to best match per scene, preserving matched timestamp
10. Truncate to `limit` results
11. Batch-fetch scene details via `_get_scene_details_batch()`
12. Write result JSON to `assets/frame_search_{request_id}.json` via a new `_write_frame_search_result(request_id, result_json)` helper (follows the `_write_search_result` pattern but with `frame_search_` prefix)

**Result JSON shape:**

```json
{
  "status": "complete",
  "query_scene_id": 42,
  "query_timestamp": 123.5,
  "model_key": "openclip:ViT-bigG-14",
  "results": [
    {
      "scene_id": 99,
      "similarity": 0.87,
      "matched_timestamp": 45.0,
      "matched_frame_index": 45,
      "scene": { "title": "...", "performers": [...], "duration": 300, ... }
    }
  ],
  "limit": 20,
  "request_id": "abc123"
}
```

**Error JSON shape:**

```json
{
  "status": "error",
  "error": "Frame search index not found. Run 'Build Frame Index' first.",
  "request_id": "abc123"
}
```

### Plugin Manifest

New task entry in `stash-copilot.yml`:

```yaml
- name: Find Similar by Frame
  description: Extract current frame, embed it, and find similar scenes via frame index
  defaultArgs:
    mode: find_similar_by_frame
    scene_id: ""
    timestamp: ""
    limit: "20"
    request_id: ""
```

### JavaScript Frontend

#### Similar Tab Modifications

**Button injection** in `renderSidebarSimilarContent()`:
- Add a "Search by Current Frame" button at the top of the Similar tab, above the All/Diff. Performers sub-tabs
- Button styled with plugin's sidebar button patterns (gradient, subtle glow)

**Click handler:**
1. Read `document.querySelector('video').currentTime`
2. Guard: if no video element, show "No video player found". If `video.readyState < 2` (HAVE_CURRENT_DATA), show "Play the video first to capture a frame" — this correctly allows `currentTime === 0` (valid first frame) while catching unloaded videos
3. Generate `request_id` (timestamp + random suffix)
4. Set `frameSearchState.active = true`
5. Replace results area with loading indicator: `"Searching by frame at M:SS..."`
6. Call `runPluginTask('Find Similar by Frame', { mode: 'find_similar_by_frame', scene_id, timestamp: currentTime.toString(), limit: '20', request_id })`

**Poll loop** (`pollFrameSearchResults`):
- Poll `/plugin/stash-copilot/assets/frame_search_{requestId}.json` every 150ms
- Validate `request_id` matches (stale result guard)
- On `status === "complete"`: render results via `buildSceneCard()` with `theme: "frame-search"` and `matchTimestamp` for each result
- On `status === "error"`: show error message inline
- Timeout at 200 attempts (30s): show timeout message suggesting model may still be loading

**State management:**
- New `frameSearchState` object tracks: `active` (bool), `results` (array), `requestId` (string)
- When `active`, the results area shows frame search results instead of normal similar results
- "Back to Similar" button sets `frameSearchState.active = false` and restores cached `similarState` results
- Normal `similarState.contentLoaded` flag remains valid

**Result rendering:**
- Use `buildSceneCard()` with existing `matchTimestamp` parameter
- Theme: `"frame-search"` with rose/pink accent `#f43f5e`
- Each card shows the matched timestamp as a badge
- Standard card interactions (hover preview, tooltip, click-to-navigate)

### CSS

New theme definition:

```css
[data-theme="frame-search"] {
  --card-accent: #f43f5e;
  --card-accent-rgb: 244, 63, 94;
}
```

Button styling for "Search by Current Frame" following existing sidebar button patterns.

## Error Handling

| Scenario | Where | User Feedback |
|---|---|---|
| No video element on page | JS pre-check | Inline: "No video player found" |
| Video not loaded (`readyState < 2`) | JS pre-check | Inline: "Play the video first to capture a frame" |
| Scene has no video file in DB | Python step 1 | Error JSON: "Could not find video file for this scene" |
| FFmpeg extraction fails | Python step 2 | Error JSON: "Failed to extract frame at timestamp X" |
| Embedding provider not configured | Python step 3 | Error JSON: "No image embedding provider configured" |
| FAISS frame index not built | Python step 4 | Error JSON: "Frame search index not found. Run 'Build Frame Index' first." |
| No results above threshold | Python step 6-7 | Success JSON with empty results: "No similar frames found" |
| Poll timeout (30s) | JS | "Search timed out. The embedding model may still be loading. Try again." |

## Performance Expectations

| Step | Expected Duration | Notes |
|---|---|---|
| FFmpeg single-frame extract | < 1s | Fast seek with `-ss` before `-i` |
| Model loading (first call) | 5-10s | OpenCLIP ViT-bigG-14, one-time per process |
| Frame embedding | < 1s | Single forward pass |
| FAISS index search | < 1s | mmap-loaded, inner product |
| Scene details batch fetch | < 1s | SQLite batched query |
| **Total (first call)** | **~10-15s** | Model load dominates |
| **Total (subsequent)** | **< 3s** | Model already in memory |

## Files to Create/Modify

| File | Change |
|---|---|
| `stash-copilot.py` | Add `find_similar_by_frame` to dispatch, implement `run_find_similar_by_frame()` |
| `stash-copilot.yml` | Add "Find Similar by Frame" task declaration |
| `stash-copilot.js` | Add button to Similar tab, click handler, poll loop, state management, result rendering |
| `stash-copilot.css` | Add `[data-theme="frame-search"]` theme |

## Non-Goals

- No client-side canvas frame capture (cross-origin restrictions risk)
- No frame caching to disk (ephemeral by design)
- No fallback to scene-level composite search (frame index required)
- No new sidebar tab (reuses existing Similar tab)
