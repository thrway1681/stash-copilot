# Find Similar by Frame Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Search by Current Frame" button to the Similar sidebar tab that extracts the current video frame, embeds it with OpenCLIP, and searches the FAISS frame index for visually similar frames across the library.

**Architecture:** Single new Python task mode `find_similar_by_frame` chains three existing primitives (FFmpeg extract → OpenCLIP embed → FAISS search). JS adds a button to the Similar tab that triggers this task and renders results in-place using the existing card system.

**Tech Stack:** Python (FFmpeg, OpenCLIP, FAISS), JavaScript (Stash plugin UI injection), CSS

**Spec:** `docs/superpowers/specs/2026-03-17-find-similar-by-frame-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `stash-copilot.py` | Modify | Add `find_similar_by_frame` dispatch + `run_find_similar_by_frame()` + `_write_frame_search_result()` |
| `stash-copilot.yml` | Modify | Add "Find Similar by Frame" task declaration |
| `stash-copilot.js` | Modify | Add button, click handler, poll loop, state, result rendering |
| `stash-copilot.css` | Modify | Add `[data-theme="frame-search"]` theme + button styles |

---

## Task 1: Python Backend — `run_find_similar_by_frame()` and result helper

**Files:**
- Modify: `stash-copilot.py` (dispatch chain — add new elif after `elif task_name == "find_similar":`)
- Modify: `stash-copilot.py` (add new method after `run_search_by_text()` — both deal with frame-level FAISS search)
- Modify: `stash-copilot.py` (add `_write_frame_search_result` helper after `_write_search_result()`)

**Note:** Line numbers below are approximate and will shift as earlier steps add code. Use the text anchors (function names) to find insertion points.

- [ ] **Step 1: Add `_write_frame_search_result()` helper**

Add after `_write_search_result()` (~line 3012). Follow the same pattern:

```python
def _write_frame_search_result(self, request_id: str, data: dict[str, Any]) -> None:
    """Write frame search results to JSON file for frontend polling."""
    import json as json_module
    import os

    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(plugin_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    filename = f"frame_search_{request_id or 'latest'}.json"
    result_file = os.path.join(assets_dir, filename)

    try:
        with open(result_file, "w") as f:
            json_module.dump(data, f)
        self.log(f"Wrote frame search results to: {result_file}", "debug")
    except Exception as e:
        self.error(f"Failed to write frame search results: {e}")
```

- [ ] **Step 2: Add `run_find_similar_by_frame()` method**

Add after `run_search_by_text()` (ends ~line 2993, before `_write_search_result`). Both methods deal with frame-level FAISS search, so they logically belong together. This chains FrameExtractor → embedding provider → FrameSearchIndex:

**Note:** The `cache_dir` parameter is required by the `FrameExtractor` constructor but is NOT used for single-frame extraction — no frames are cached to disk. The extracted frame lives only in memory.

```python
def run_find_similar_by_frame(self, args: dict[str, Any]) -> None:
    """Find similar scenes by extracting and embedding the current video frame.

    Extracts a single frame at the given timestamp, embeds it with the
    configured image embedding provider, and searches the FAISS frame
    index for visually similar frames across the library.

    Args:
        args: Task arguments containing:
            - scene_id: Scene ID currently playing (required)
            - timestamp: Playback position in seconds (required)
            - limit: Maximum results (default 20)
            - request_id: Unique request ID for frontend polling (required)
    """
    try:
        import numpy as np

        from stash_ai.embeddings.config import EmbeddingConfig
        from stash_ai.embeddings.frame_search import FrameSearchIndex
        from stash_ai.embeddings.provider import get_embedding_provider
        from stash_ai.tasks.frame_extractor import FrameExtractionConfig, FrameExtractor
        from stash_ai.tools.database import get_readonly_connection, get_stash_db_path

        scene_id = args.get("scene_id", "")
        timestamp_str = args.get("timestamp", "0")
        limit = int(args.get("limit", 20))
        request_id = args.get("request_id", "")

        if not scene_id:
            self._write_frame_search_result(
                request_id, {"status": "error", "error": "Scene ID is required", "request_id": request_id}
            )
            return

        timestamp = float(timestamp_str)
        self.log(f"Frame search: scene={scene_id}, timestamp={timestamp:.1f}s, limit={limit}", "info")

        # Step 1: Resolve video file path from Stash SQLite
        db_path = get_stash_db_path()
        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT fo.path || '/' || f.basename as video_path
            FROM scenes s
            JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
            JOIN files f ON sf.file_id = f.id
            JOIN folders fo ON f.parent_folder_id = fo.id
            JOIN video_files vf ON f.id = vf.file_id
            WHERE s.id = ?
            """,
            (int(scene_id),),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            self._write_frame_search_result(
                request_id,
                {"status": "error", "error": f"Could not find video file for scene {scene_id}", "request_id": request_id},
            )
            return

        video_path = row["video_path"]

        # Step 2: Extract frame at timestamp
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(plugin_dir, "assets")
        extractor = FrameExtractor(
            config=FrameExtractionConfig(),
            cache_dir=os.path.join(assets_dir, "embedded_frames"),
            log_callback=self.log,
        )

        frame_bytes = extractor.extract_frame_at_timestamp(video_path, timestamp)
        if frame_bytes is None:
            self._write_frame_search_result(
                request_id,
                {"status": "error", "error": f"Failed to extract frame at {timestamp:.1f}s", "request_id": request_id},
            )
            return

        self.log(f"Extracted frame: {len(frame_bytes)} bytes", "debug")

        # Step 3: Embed the frame
        plugin_settings = self.get_plugin_settings("stash-copilot")
        image_provider = plugin_settings.get("image_embedding_provider")
        image_model = plugin_settings.get("image_embedding_model")
        image_device = plugin_settings.get("image_embedding_device") or "auto"

        if not image_provider or not image_model:
            self._write_frame_search_result(
                request_id,
                {"status": "error", "error": "No image embedding provider configured. Set up in Plugin Settings.", "request_id": request_id},
            )
            return

        embedding_config = EmbeddingConfig(
            provider=image_provider,
            model=image_model,
            device=image_device,
        )
        model_key = embedding_config.model_key

        embedder = get_embedding_provider(embedding_config)
        result = embedder.embed_image(frame_bytes)
        query_embedding = np.array(result["embedding"], dtype=np.float32)

        self.log(f"Embedded frame: {result['dimensions']} dims", "debug")

        # Step 4: Load frame search index
        frame_index = FrameSearchIndex(assets_dir=assets_dir, model_key=model_key)

        if not frame_index.exists:
            self._write_frame_search_result(
                request_id,
                {
                    "status": "error",
                    "error": f"Frame search index not found for model '{model_key}'. Run 'Build Frame Search Index' task first.",
                    "request_id": request_id,
                },
            )
            return

        # Step 5: Search for similar frames
        frame_matches = frame_index.search(query_embedding, top_k=limit * 3)

        # Step 6: Filter out query scene's own frames
        frame_matches = [m for m in frame_matches if m.scene_id != int(scene_id)]

        # Step 7: Aggregate to best match per scene
        scene_matches = frame_index.aggregate_to_scenes(frame_matches)

        # Step 8: Truncate to limit
        scene_matches = scene_matches[:limit]

        # Step 9: Fetch scene details
        scene_details = self._get_scene_details_batch([m.scene_id for m in scene_matches])

        # Step 10: Build result data
        result_data = []
        for m in scene_matches:
            scene = scene_details.get(m.scene_id, {})
            result_data.append(
                {
                    "scene_id": m.scene_id,
                    "similarity": m.similarity,
                    "matched_timestamp": m.best_timestamp,
                    "matched_frame_index": m.best_frame_index,
                    "scene": scene,
                }
            )

        # Step 11: Write results
        self._write_frame_search_result(
            request_id,
            {
                "status": "complete",
                "query_scene_id": int(scene_id),
                "query_timestamp": timestamp,
                "model_key": model_key,
                "results": result_data,
                "limit": limit,
                "request_id": request_id,
            },
        )

        self.log(f"Frame search complete: {len(result_data)} scenes found", "info")

    except ImportError as e:
        self.error(f"Failed to import modules: {e}")
        self._write_frame_search_result(
            args.get("request_id", ""),
            {"status": "error", "error": f"Failed to import modules: {e}", "request_id": args.get("request_id", "")},
        )
    except Exception as e:
        self.error(f"Frame search error: {e}")
        self._write_frame_search_result(
            args.get("request_id", ""),
            {"status": "error", "error": str(e), "request_id": args.get("request_id", "")},
        )
```

- [ ] **Step 3: Add dispatch entry in `run_task()`**

In `run_task()`, find the line `elif task_name == "find_similar":` and add immediately after its handler line:

```python
        elif task_name == "find_similar_by_frame":
            self.run_find_similar_by_frame(args)
```

- [ ] **Step 4: Verify Python syntax**

Run: `python -c "import ast; ast.parse(open('stash-copilot.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add stash-copilot.py
git commit -m "feat: add find_similar_by_frame Python task

Chains FrameExtractor → OpenCLIP embed_image → FAISS FrameSearchIndex
to find visually similar scenes from the current playback frame."
```

---

## Task 2: Plugin Manifest — Task Declaration

**Files:**
- Modify: `stash-copilot.yml:553` (add new task before hooks section)

- [ ] **Step 1: Add task declaration**

Add after the last task entry (line 553, before the blank line and `hooks:` section):

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

- [ ] **Step 2: Validate YAML syntax**

Run: `python -c "import yaml; yaml.safe_load(open('stash-copilot.yml')); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add stash-copilot.yml
git commit -m "feat: add 'Find Similar by Frame' task to plugin manifest"
```

---

## Task 3: CSS Theme — `frame-search` card theme

**Files:**
- Modify: `stash-copilot.css:6260` (add new theme after `tag-gaps` theme)

- [ ] **Step 1: Add card theme**

After the `tag-gaps` theme block (line 6260), add:

```css
/* Theme: Frame Search (Rose/Pink) */
.stash-copilot-card[data-theme="frame-search"] {
    --card-accent: #f43f5e;
    --card-accent-rgb: 244, 63, 94;
    --card-gradient: linear-gradient(135deg, rgba(244, 63, 94, 0.9), rgba(236, 72, 153, 0.85));
}
```

- [ ] **Step 2: Add tooltip theme**

Find the tooltip theme section (~line 6571-6602) and add after the last tooltip theme:

```css
.stash-copilot-card-tooltip[data-theme="frame-search"] {
    --tooltip-accent: #f43f5e;
    --tooltip-accent-rgb: 244, 63, 94;
}
```

- [ ] **Step 3: Add "Search by Current Frame" button styles**

Add near the existing sidebar button styles:

```css
/* Frame search button */
.stash-copilot-frame-search-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    margin-bottom: 8px;
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-radius: 6px;
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(236, 72, 153, 0.1));
    color: #f43f5e;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.2s ease;
    width: 100%;
    justify-content: center;
}

.stash-copilot-frame-search-btn:hover {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.25), rgba(236, 72, 153, 0.2));
    border-color: rgba(244, 63, 94, 0.5);
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.2);
}

.stash-copilot-frame-search-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Back to similar button */
.stash-copilot-back-to-similar {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    margin-bottom: 8px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 6px;
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
    cursor: pointer;
    font-size: 11px;
    transition: all 0.2s ease;
}

.stash-copilot-back-to-similar:hover {
    background: rgba(16, 185, 129, 0.2);
    border-color: rgba(16, 185, 129, 0.5);
}

/* Frame search results header */
.stash-copilot-sidebar-frame-search-header {
    font-size: 11px;
    color: rgba(244, 63, 94, 0.8);
    margin-bottom: 8px;
    padding: 4px 0;
    font-weight: 500;
}
```

- [ ] **Step 4: Commit**

```bash
git add stash-copilot.css
git commit -m "feat: add frame-search card theme and button styles"
```

---

## Task 4: JavaScript Frontend — Button, State, Handler, Poll, Render

**Files:**
- Modify: `stash-copilot.js:9036-9071` (add frameSearchState after similarState)
- Modify: `stash-copilot.js:10246-10306` (add button to renderSidebarSimilarContent)
- Modify: `stash-copilot.js` (add new functions after renderSidebarSimilarResultsUI)

- [ ] **Step 1: Add `frameSearchState` object**

After `similarState` definition (~line 9071), add:

```javascript
    // ===== Frame Search State =====
    const frameSearchState = {
        active: false,
        results: [],
        requestId: '',
        queryTimestamp: 0
    };
```

- [ ] **Step 2: Add button to `renderSidebarSimilarContent()`**

In `renderSidebarSimilarContent()` (line 10249), insert the button HTML right after the opening `<div class="stash-copilot-sidebar-similar">` and before the subtabs div. Replace the container.innerHTML template:

Find the line:
```javascript
            <div class="stash-copilot-sidebar-similar">
                <div class="stash-copilot-sidebar-subtabs">
```

Insert between them:
```javascript
            <div class="stash-copilot-sidebar-similar">
                <button class="stash-copilot-frame-search-btn" title="Search for similar scenes using the current video frame">
                    🎯 Search by Current Frame
                </button>
                <div class="stash-copilot-sidebar-subtabs">
```

- [ ] **Step 3: Add click handler wiring in `setupSidebarSimilarListeners()` or after `renderSidebarSimilarContent()`**

After the `setupSidebarSimilarListeners(container, sceneId)` call in `renderSidebarSimilarContent()` (line 10302), add the frame search button click handler:

```javascript
        // Frame search button handler
        const frameSearchBtn = container.querySelector('.stash-copilot-frame-search-btn');
        if (frameSearchBtn) {
            frameSearchBtn.addEventListener('click', () => {
                startFrameSearch(sceneId, container);
            });
        }
```

- [ ] **Step 4: Add `startFrameSearch()` function**

Add after `renderSidebarSimilarResultsUI()` function:

```javascript
    async function startFrameSearch(sceneId, container) {
        const video = document.querySelector('video');
        if (!video) {
            showSidebarError(container, 'No video player found');
            return;
        }
        if (video.readyState < 2) {
            showSidebarError(container, 'Play the video first to capture a frame');
            return;
        }

        const timestamp = video.currentTime;
        const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

        frameSearchState.active = true;
        frameSearchState.requestId = requestId;
        frameSearchState.queryTimestamp = timestamp;

        // Format timestamp for display
        const mins = Math.floor(timestamp / 60);
        const secs = Math.floor(timestamp % 60);
        const timeStr = `${mins}:${secs.toString().padStart(2, '0')}`;

        // Show loading state — replace results area
        const panel = document.getElementById('scene-copilot-similar-panel');
        if (!panel) return;

        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        const paginationDiv = panel.querySelector('.stash-copilot-sidebar-pagination');
        const subtabsDiv = panel.querySelector('.stash-copilot-sidebar-subtabs');
        const sliderDiv = panel.querySelector('.stash-copilot-sidebar-slider');
        const filtersDiv = panel.querySelector('.stash-copilot-sidebar-filters');

        // Hide normal similar controls
        if (subtabsDiv) subtabsDiv.style.display = 'none';
        if (sliderDiv) sliderDiv.style.display = 'none';
        if (filtersDiv) filtersDiv.style.display = 'none';
        if (paginationDiv) paginationDiv.style.display = 'none';

        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <button class="stash-copilot-back-to-similar">
                    ← Back to Similar
                </button>
                <div class="stash-copilot-sidebar-loading">
                    <div class="stash-copilot-spinner"></div>
                    <span>Searching by frame at ${timeStr}...</span>
                </div>
            `;

            // Wire up back button via event delegation
            const backBtn = resultsDiv.querySelector('.stash-copilot-back-to-similar');
            if (backBtn) {
                backBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    exitFrameSearch(sceneId, panel);
                });
            }
        }

        // Disable the frame search button during search
        const frameSearchBtn = panel.querySelector('.stash-copilot-frame-search-btn');
        if (frameSearchBtn) frameSearchBtn.disabled = true;

        // Trigger backend task
        try {
            await runPluginTask('Find Similar by Frame', {
                mode: 'find_similar_by_frame',
                scene_id: String(sceneId),
                timestamp: String(timestamp),
                limit: '20',
                request_id: requestId
            });

            pollFrameSearchResults(requestId, sceneId, panel);
        } catch (e) {
            log(`Frame search error: ${e.message}`, 'error');
            showSidebarError(panel, e.message);
            if (frameSearchBtn) frameSearchBtn.disabled = false;
        }
    }
```

- [ ] **Step 5: Add `pollFrameSearchResults()` function**

```javascript
    function pollFrameSearchResults(requestId, sceneId, panel) {
        const resultFile = `/plugin/stash-copilot/assets/frame_search_${requestId}.json`;
        let attempts = 0;
        const maxAttempts = 200;

        const pollInterval = setInterval(async () => {
            attempts++;
            if (attempts > maxAttempts) {
                clearInterval(pollInterval);
                const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
                if (resultsDiv) {
                    const loading = resultsDiv.querySelector('.stash-copilot-sidebar-loading');
                    if (loading) loading.innerHTML = `
                        <span>Search timed out. The embedding model may still be loading. Try again.</span>
                    `;
                }
                const btn = panel.querySelector('.stash-copilot-frame-search-btn');
                if (btn) btn.disabled = false;
                return;
            }

            try {
                const response = await fetch(resultFile + `?t=${Date.now()}`, { cache: 'no-store' });
                if (response.ok) {
                    const data = await response.json();

                    // Validate request_id to avoid stale results
                    if (data.request_id !== requestId) return;

                    if (data.status === 'complete' || data.results) {
                        clearInterval(pollInterval);
                        frameSearchState.results = data.results || [];
                        renderFrameSearchResults(data, panel);
                    } else if (data.status === 'error') {
                        clearInterval(pollInterval);
                        showSidebarError(panel, data.error || 'Frame search failed');
                        const btn = panel.querySelector('.stash-copilot-frame-search-btn');
                        if (btn) btn.disabled = false;
                    }
                }
            } catch (e) {
                log(`Frame search poll error: ${e.message}`);
            }
        }, 150);
        // Timeout is handled by the attempt counter above (200 * 150ms = 30s)
    }
```

- [ ] **Step 6: Add `renderFrameSearchResults()` function**

```javascript
    function renderFrameSearchResults(data, panel) {
        const resultsDiv = panel.querySelector('.stash-copilot-sidebar-results');
        if (!resultsDiv) return;

        const results = data.results || [];
        const btn = panel.querySelector('.stash-copilot-frame-search-btn');
        if (btn) btn.disabled = false;

        if (results.length === 0) {
            resultsDiv.innerHTML = `
                <button class="stash-copilot-back-to-similar">← Back to Similar</button>
                <div class="stash-copilot-sidebar-empty">No similar frames found</div>
            `;
            wireBackButton(resultsDiv, data.query_scene_id, panel);
            return;
        }

        // Format query timestamp for header
        const qts = data.query_timestamp || 0;
        const qMins = Math.floor(qts / 60);
        const qSecs = Math.floor(qts % 60);
        const qTimeStr = `${qMins}:${qSecs.toString().padStart(2, '0')}`;

        // Build cards using unified card system
        const cardsHtml = results.map((result, idx) => {
            const scene = result.scene || {};
            return buildSceneCard({
                scene: scene,
                score: result.similarity,
                cardIndex: idx,
                theme: 'frame-search',
                scoreLabel: 'match',
                matchTimestamp: result.matched_timestamp
            });
        }).join('');

        resultsDiv.innerHTML = `
            <button class="stash-copilot-back-to-similar">← Back to Similar</button>
            <div class="stash-copilot-sidebar-frame-search-header">
                Frame at ${qTimeStr} · ${results.length} match${results.length !== 1 ? 'es' : ''}
            </div>
            ${cardsHtml}
        `;

        wireBackButton(resultsDiv, data.query_scene_id, panel);

        // Setup card events
        setupSceneCardEvents(resultsDiv, { theme: 'frame-search', tooltipMode: 'cursor' });
    }

    function wireBackButton(container, sceneId, panel) {
        const backBtn = container.querySelector('.stash-copilot-back-to-similar');
        if (backBtn) {
            backBtn.addEventListener('click', (e) => {
                e.preventDefault();
                exitFrameSearch(sceneId, panel);
            });
        }
    }
```

- [ ] **Step 7: Add `exitFrameSearch()` function**

```javascript
    function exitFrameSearch(sceneId, panel) {
        frameSearchState.active = false;
        frameSearchState.results = [];

        // Re-show normal similar controls
        const subtabsDiv = panel.querySelector('.stash-copilot-sidebar-subtabs');
        const sliderDiv = panel.querySelector('.stash-copilot-sidebar-slider');
        const paginationDiv = panel.querySelector('.stash-copilot-sidebar-pagination');

        if (subtabsDiv) subtabsDiv.style.display = '';
        if (sliderDiv) sliderDiv.style.display = '';
        // paginationDiv visibility is managed by renderSidebarSimilarResultsUI, just reset to allow it
        if (paginationDiv) paginationDiv.style.display = '';

        // Re-render cached similar results if available
        const tabState = similarState.tabs[similarState.activeTab];
        if (tabState && tabState.loaded) {
            renderSidebarSimilarResultsUI(tabState, panel);
        } else {
            // Re-trigger search if no cached results
            startSidebarSimilarSearch(sceneId);
        }
    }
```

- [ ] **Step 8: Verify JS has no syntax errors**

Run: `node -c stash-copilot.js`

Expected: no output (success)

- [ ] **Step 9: Commit**

```bash
git add stash-copilot.js
git commit -m "feat: add 'Search by Current Frame' button to Similar tab

Adds frame search button, click handler with video readyState guard,
poll loop for frame_search_{requestId}.json, result rendering with
frame-search theme, and back-to-similar navigation."
```

---

## Task 5: Integration Testing

**Files:** None (manual testing via Playwright MCP)

- [ ] **Step 1: Verify plugin loads without errors**

Run: `uv run python -c "import stash_ai; print('Import OK')"`

Navigate to a scene page in Stash and check browser console for JS errors.

- [ ] **Step 2: Verify the "Search by Current Frame" button appears**

Navigate to a scene page → click the Similar tab → verify the rose/pink "Search by Current Frame" button appears above the sub-tabs.

- [ ] **Step 3: Test the button click flow**

Play a video → seek to a specific time → click "Search by Current Frame" → verify:
1. Loading state appears with formatted timestamp
2. Normal similar controls (subtabs, slider) are hidden
3. Results appear after search completes (or error message if index not built)
4. Results use the rose/pink `frame-search` theme
5. Each result card shows the matched timestamp badge

- [ ] **Step 4: Test back navigation**

Click "Back to Similar" → verify:
1. Normal similar results are restored from cache
2. Subtabs and slider reappear
3. Frame search button is re-enabled

- [ ] **Step 5: Test error cases**

1. Click button before video loads → should show "Play the video first to capture a frame"
2. If frame index not built → should show clear error message

- [ ] **Step 6: Check logs for errors**

Run: `strings ~/.stash/stash.log | grep -i "error\|exception\|traceback" | tail -20`

- [ ] **Step 7: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address integration test findings for frame search"
```
