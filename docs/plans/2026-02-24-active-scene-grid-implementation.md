# Active Scene Grid Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 16-slot fixed filmstrip grid on active scene cards with a full-frame minimap/thumbnail grid driven entirely by real-time WebSocket events.

**Architecture:** Backend sends all frame statuses (name + ok/error/pending) as one-time seed when a scene appears. Frontend renders a minimap of tiny colored cells. Clicking expands to actual thumbnails. All live updates come exclusively from `FRAME_DONE` WebSocket events — no polling fallback. Soft glow CSS transitions replace jerky scale animations.

**Tech Stack:** Python (caption_dashboard.py), vanilla HTML/CSS/JS (caption_dashboard.html)

**Design doc:** `docs/plans/2026-02-24-active-scene-grid-redesign.md`

---

## Task 1: Backend — Add `frame_statuses` to in-progress scene payload

**Files:**
- Modify: `tools/dataset/caption_dashboard.py:302-444` (`load_in_progress_scenes`)

**Step 1: Add frame status collection to `load_in_progress_scenes()`**

In the per-scene loop (line ~397–443), instead of only counting `captioned_count` and `error_count` and building a 16-frame `sample_frames` list, build a full `frame_statuses` list for all frames:

```python
# Replace the current frame scanning block (lines ~403-441) with:

        # Build complete frame status map for minimap
        frame_statuses: list[dict[str, str]] = []
        captioned_count = 0
        error_count = 0
        latest_caption = ""
        for fname in frames:
            txt = images_dir / (fname.rsplit(".", 1)[0] + ".txt")
            if txt.exists():
                try:
                    content = txt.read_text(encoding="utf-8")
                    if content.startswith("[ERROR"):
                        error_count += 1
                        frame_statuses.append({"frame": fname, "status": "error"})
                    else:
                        captioned_count += 1
                        frame_statuses.append({"frame": fname, "status": "ok"})
                        latest_caption = content  # Keep last good caption
                except OSError:
                    frame_statuses.append({"frame": fname, "status": "pending"})
            else:
                frame_statuses.append({"frame": fname, "status": "pending"})
```

**Step 2: Update the scene dict to use `frame_statuses` instead of `sample_frames`**

Replace the current `sample_frames` construction and scene dict (lines ~422-441) with:

```python
        is_retry = sid in retry_sids
        scenes.append({
            "scene_id": sid,
            "frame_count": len(frames),
            "captioned_count": captioned_count,
            "error_count": error_count,
            "in_progress": True,
            "retrying": is_retry,
            "frame_statuses": frame_statuses,
            "latest_caption": latest_caption,
            "selection": {},
            "captioned_at": "",
        })
```

Note: `sample_frames` is removed from in-progress scenes. Completed scenes (from `load_recent_scenes`) keep their `sample_frames` — they don't use the minimap.

**Step 3: Verify the backend serves the new payload**

Run the dashboard, start the runner, and check `/api/scenes` returns `frame_statuses` for active scenes:

```bash
curl -s http://localhost:8766/api/scenes | python3 -c "import sys,json; d=json.load(sys.stdin); [print(s.get('scene_id'), len(s.get('frame_statuses',[])), 'statuses') for s in d['scenes'] if s.get('in_progress')]"
```

Expected: Scene IDs with full frame counts (e.g., `583 394 statuses`).

**Step 4: Commit**

```bash
git add tools/dataset/caption_dashboard.py
git commit -m "feat(dashboard): send full frame_statuses for active scenes instead of 16-frame sample"
```

---

## Task 2: CSS — Minimap cells, expanded thumbnails, soft glow animations

**Files:**
- Modify: `tools/dataset/caption_dashboard.html:549-578` (grid-mode CSS), `580-663` (filmstrip frame CSS)

**Step 1: Replace grid-mode CSS with minimap + expanded-grid styles**

Replace the existing `.scene-filmstrip.grid-mode` block and the `gridFadeIn`/`empty-slot` rules (lines 549–578) with:

```css
/* ── Minimap (default for active scenes) ─────── */
.scene-minimap {
  display: grid;
  grid-template-columns: repeat(auto-fill, 12px);
  gap: 1px;
  padding: 0 0.85rem 0.5rem;
  max-height: 120px;
  overflow: hidden;
  transition: max-height 0.4s ease-out;
}

.scene-minimap.collapsed {
  max-height: 120px;
}

.minimap-cell {
  width: 12px; height: 7px;
  border-radius: 1.5px;
  background: rgba(255,255,255,0.06);
  transition: background-color 0.6s ease, box-shadow 1.5s ease;
}

.minimap-cell.ok {
  background: rgba(52,211,153,0.6);
}

.minimap-cell.error {
  background: rgba(248,113,113,0.6);
}

.minimap-cell.just-done {
  box-shadow: 0 0 4px rgba(52,211,153,0.5);
}

.minimap-cell.just-error {
  box-shadow: 0 0 4px rgba(248,113,113,0.5);
}

/* ── Expand toggle ───────────────────────────── */
.minimap-expand-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.3rem;
  padding: 0.25rem 0.85rem;
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-3);
  cursor: pointer;
  user-select: none;
  transition: color 0.2s;
}

.minimap-expand-btn:hover {
  color: var(--blue);
}

.minimap-expand-btn .expand-arrow {
  transition: transform 0.3s ease;
  font-size: 0.5rem;
}

.minimap-expand-btn.expanded .expand-arrow {
  transform: rotate(180deg);
}

/* ── Expanded thumbnail grid ─────────────────── */
.scene-thumb-grid {
  display: none;
  grid-template-columns: repeat(auto-fill, 56px);
  gap: 3px;
  padding: 0 0.85rem 0.5rem;
  max-height: 400px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: rgba(96,165,250,0.2) transparent;
}

.scene-thumb-grid.visible {
  display: grid;
  animation: fadeSlideUp 0.3s ease-out;
}

.scene-thumb-grid::-webkit-scrollbar { width: 4px; }
.scene-thumb-grid::-webkit-scrollbar-track { background: transparent; }
.scene-thumb-grid::-webkit-scrollbar-thumb { background: rgba(96,165,250,0.2); border-radius: 2px; }

.thumb-cell {
  position: relative;
  width: 56px; height: 32px;
  border-radius: 4px;
  overflow: hidden;
  background: var(--surface-2);
  border: 1px solid var(--border);
  cursor: pointer;
  /* Pending state: dim and desaturated */
  opacity: 0.3;
  filter: saturate(0.2) brightness(0.5);
  transition: opacity 0.6s ease, filter 0.6s ease, border-color 0.6s ease, box-shadow 1.5s ease;
}

.thumb-cell.ok {
  opacity: 1;
  filter: none;
}

.thumb-cell.error {
  opacity: 0.8;
  filter: saturate(0.6);
  border-color: rgba(248,113,113,0.3);
}

.thumb-cell.just-done {
  border-color: rgba(52,211,153,0.5);
  box-shadow: 0 0 8px rgba(52,211,153,0.25);
}

.thumb-cell:hover {
  border-color: var(--blue);
  transform: scale(1.08);
  box-shadow: 0 2px 8px rgba(96,165,250,0.2);
  z-index: 2;
  opacity: 1;
  filter: none;
}

.thumb-cell img {
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
}
```

**Step 2: Remove old grid-mode CSS that is no longer used**

Delete these blocks that are replaced by the minimap/thumb-grid:
- `.scene-filmstrip.grid-mode` (line 550-556)
- `.scene-filmstrip-frame.empty-slot` and `:hover` (lines 558-568)
- `.scene-filmstrip-frame.grid-fill` and `@keyframes gridFadeIn` (lines 570-578)

**Step 3: Fix lightbox caption word-wrap**

In the `.lightbox-caption` rule (line ~818), add:

```css
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
```

**Step 4: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "feat(dashboard): minimap + expanded grid CSS with soft glow transitions"
```

---

## Task 3: HTML — Replace filmstrip markup in active card template

**Files:**
- Modify: `tools/dataset/caption_dashboard.html:2149-2171` (active card HTML in `buildSceneCardHTML`)

**Step 1: Replace the active card HTML template**

In `buildSceneCardHTML()` (line ~2149), replace the in-progress card return block with:

```javascript
    if (isInProgress) {
      // ── Active card with minimap ──
      const activeTag = s.retrying
        ? `<span class="tag tag-active" style="--tag-accent:var(--accent-amber,#f59e0b)">retrying\u2026</span>`
        : `<span class="tag tag-active">captioning\u2026</span>`;

      // Build minimap cells from frame_statuses
      const statuses = s.frame_statuses || [];
      const minimapCells = statuses.map(fs =>
        `<div class="minimap-cell ${fs.status}" data-frame="${escHTML(fs.frame)}"></div>`
      ).join('');

      // Build thumbnail grid cells (hidden by default)
      const thumbCells = statuses.map(fs => {
        const imgSrc = '/assets/lora_dataset/images/' + encodeURIComponent(fs.frame);
        const captionAttr = ''; // No caption text in initial payload
        return `<div class="thumb-cell ${fs.status}" data-frame="${escHTML(fs.frame)}"${captionAttr}>
          <img src="${imgSrc}" alt="${escHTML(fs.frame)}" loading="lazy"
               onclick="event.stopPropagation(); window.__lightbox('${imgSrc}', this.parentElement.dataset.caption || '')">
        </div>`;
      }).join('');

      const latestCaption = s.latest_caption || '';

      return `<div class="${cardClasses.join(' ')}" data-scene="${s.scene_id}" style="animation-delay:${Math.min(i * 30, 300)}ms">
        <div class="scene-card-head" onclick="window.__toggleScene(${s.scene_id})">
          <span class="scene-id">Scene ${s.scene_id}</span>
          <div class="scene-meta">
            <span class="tag tag-frames">${s.frame_count} frames</span>
            ${activeTag}
          </div>
          <span class="scene-chevron">&#9656;</span>
        </div>
        ${progressHTML}
        <div class="scene-minimap" id="minimap-${s.scene_id}">${minimapCells}</div>
        <div class="minimap-expand-btn" id="expand-btn-${s.scene_id}" onclick="event.stopPropagation(); window.__toggleGrid(${s.scene_id})">
          <span class="expand-arrow">&#9660;</span> Show thumbnails
        </div>
        <div class="scene-thumb-grid" id="thumbgrid-${s.scene_id}">${thumbCells}</div>
        <div class="scene-caption-preview" id="caption-${s.scene_id}">
          <span class="caption-label">Latest</span>
          <span class="caption-text">${escHTML(latestCaption)}</span>
        </div>
        <div class="scene-detail" id="detail-${s.scene_id}"></div>
      </div>`;
    }
```

**Step 2: Add the `__toggleGrid` function**

After the `__toggleScene` function (~line 2670), add:

```javascript
  window.__toggleGrid = function(sceneId) {
    const grid = document.getElementById('thumbgrid-' + sceneId);
    const btn = document.getElementById('expand-btn-' + sceneId);
    if (!grid || !btn) return;
    const isVisible = grid.classList.contains('visible');
    grid.classList.toggle('visible', !isVisible);
    btn.classList.toggle('expanded', !isVisible);
    btn.querySelector('.expand-arrow').textContent = isVisible ? '\u25BC' : '\u25B2';
    const label = btn.childNodes[btn.childNodes.length - 1];
    if (label) label.textContent = isVisible ? ' Show thumbnails' : ' Hide thumbnails';
  };
```

**Step 3: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "feat(dashboard): minimap + expandable thumbnail grid in active scene cards"
```

---

## Task 4: JS — Real-time frame event handler for minimap + thumb-grid

**Files:**
- Modify: `tools/dataset/caption_dashboard.html:2531-2649` (replace `animateFrameIn`)

**Step 1: Replace `animateFrameIn` with minimap+grid-aware version**

Replace the entire `animateFrameIn` function (lines ~2531-2649) with:

```javascript
  function animateFrameIn(sceneId, evt) {
    const frameName = evt.frame;
    if (!frameName) return;
    const status = (evt.status || 'OK').toLowerCase();
    const caption = evt.caption || '';
    const escapedFrame = CSS.escape(frameName);

    // ── Update minimap cell ──
    const minimap = document.getElementById('minimap-' + sceneId);
    if (minimap) {
      const cell = minimap.querySelector(`.minimap-cell[data-frame="${escapedFrame}"]`);
      if (cell) {
        cell.className = 'minimap-cell ' + status;
        // Add glow class, remove after transition
        const glowClass = status === 'error' ? 'just-error' : 'just-done';
        cell.classList.add(glowClass);
        setTimeout(() => cell.classList.remove(glowClass), 1500);
      }
    }

    // ── Update thumbnail grid cell ──
    const grid = document.getElementById('thumbgrid-' + sceneId);
    if (grid) {
      const thumb = grid.querySelector(`.thumb-cell[data-frame="${escapedFrame}"]`);
      if (thumb) {
        thumb.className = 'thumb-cell ' + status;
        if (caption) thumb.dataset.caption = caption;
        thumb.title = caption;
        if (status === 'ok') {
          thumb.classList.add('just-done');
          setTimeout(() => thumb.classList.remove('just-done'), 1500);
        }
      }
    }

    // ── Update old-style filmstrip if present (completed cards) ──
    const strip = document.getElementById('filmstrip-' + sceneId);
    if (strip) {
      const existingFrame = strip.querySelector(`.scene-filmstrip-frame[data-frame="${escapedFrame}"]`);
      if (existingFrame) {
        const statusEl = existingFrame.querySelector('.frame-status');
        if (statusEl) {
          statusEl.className = 'frame-status ' + status;
          statusEl.textContent = status === 'ok' ? '\u2713' : '\u2717';
        }
        if (caption) existingFrame.dataset.caption = caption;
        existingFrame.title = caption;
      }
    }

    // ── Update latest caption ──
    if (caption && !caption.startsWith('[ERROR')) {
      updateLatestCaption(sceneId, [caption]);
    }

    // ── Update progress bar from accumulated events ──
    const card = document.querySelector(`.scene-card[data-scene="${sceneId}"]`);
    if (card && minimap) {
      const okCount = minimap.querySelectorAll('.minimap-cell.ok').length;
      const errCount = minimap.querySelectorAll('.minimap-cell.error').length;
      const total = minimap.querySelectorAll('.minimap-cell').length;
      const pctVal = total ? (okCount + errCount) / total * 100 : 0;
      const hasErrors = errCount > 0;

      const fill = card.querySelector('.scene-progress-fill');
      if (fill) {
        fill.classList.toggle('has-errors', hasErrors);
        fill.style.width = pctVal.toFixed(1) + '%';
        if (hasErrors) {
          const okPct = (okCount + errCount) > 0 ? okCount / (okCount + errCount) * 100 : 100;
          fill.style.setProperty('--ok-pct', okPct.toFixed(1) + '%');
        } else {
          fill.style.removeProperty('--ok-pct');
        }
      }

      const label = card.querySelector('.scene-progress-label');
      if (label) {
        label.innerHTML = okCount + '/' + total + (errCount ? ' <span class="err">' + errCount + ' err</span>' : '');
      }
    }
  }
```

**Step 2: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "feat(dashboard): real-time minimap + thumb-grid updates from WS frame events"
```

---

## Task 5: JS — Remove `syncFilmstrip` fallback and update `renderActiveScenes`

**Files:**
- Modify: `tools/dataset/caption_dashboard.html:2201-2241` (`renderActiveScenes`), `2380-2477` (`syncFilmstrip`)

**Step 1: Remove the `syncFilmstrip` function entirely**

Delete the `syncFilmstrip` function (lines ~2382-2477). It is no longer called.

**Step 2: Remove the `autoScrollFilmstrip` function**

Delete `autoScrollFilmstrip` (line ~2376-2378). It only applied to the old horizontal filmstrip in active cards.

Note: keep `autoScrollFilmstrip` if completed cards still use it — check references. Actually completed cards use `populateCompletedFilmstrip` which doesn't call it. The only callers were `syncFilmstrip` and `animateFrameIn` (old version). Safe to remove.

**Step 3: Update `renderActiveScenes` to remove filmstrip sync calls**

In `renderActiveScenes()` (lines ~2201-2241), simplify the in-place update path. Replace the current else block (lines ~2229-2241):

```javascript
    } else {
      // In-place update: progress bars and captions (minimap updates come from frame_events)
      activeData.forEach(s => {
        const card = activeContainer.querySelector(`.scene-card[data-scene="${s.scene_id}"]`);
        if (!card) return;
        updateCardProgress(card, s);
      });
    }
```

This removes the `syncFilmstrip` and `updateLatestCaption` calls from the polling path — those are now handled exclusively by `animateFrameIn` via frame events.

**Step 4: In `buildFilmstripHTML`, remove the grid-mode branch**

The `buildFilmstripHTML` function (line ~2068) has an `if (gridMode)` branch that generates the old 16-slot grid. Since active cards no longer use filmstrips, remove the `gridMode` parameter and the grid branch entirely. Only keep the horizontal filmstrip mode (used by completed cards):

```javascript
  function buildFilmstripHTML(sampleFrames, frameCount, isNew) {
    const items = (sampleFrames || []).slice(0, 16);
    // Horizontal filmstrip mode (completed cards only)
    const frames = items.map((item, i) => {
      // ... existing horizontal code unchanged ...
    }).join('');
    const overflow = frameCount > 16
      ? `<span class="scene-filmstrip-overflow">+${frameCount - 16}</span>`
      : '';
    return frames + overflow;
  }
```

Update the one call site in `buildSceneCardHTML` for the completed card path (~line 2135):
```javascript
    const filmstripHTML = buildFilmstripHTML(s.sample_frames, s.frame_count, false);
```

**Step 5: Clean up `sceneFrameEvents` buffer limit**

In the `frame_events` handler (line ~1876), remove the 16-item cap since the minimap can hold all frames:

```javascript
        if (!sceneFrameEvents[sid]) sceneFrameEvents[sid] = [];
        sceneFrameEvents[sid].push(evt);
        // No length cap — minimap shows all frames
```

Do the same in the `init` handler (line ~1826).

**Step 6: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "refactor(dashboard): remove syncFilmstrip fallback, clean up grid-mode code"
```

---

## Task 6: Visual polish and testing

**Files:**
- Modify: `tools/dataset/caption_dashboard.html` (CSS tweaks)

**Step 1: Test with the live runner**

1. Start the dashboard: `uv run python tools/dataset/caption_dashboard.py`
2. Open in browser via Playwright MCP
3. Launch the caption runner with 10-15 workers
4. Verify:
   - Minimap renders all frames as tiny dark cells
   - Cells light up green as workers complete captions
   - Glow effect is visible and fades smoothly
   - "Latest" caption updates in real-time
   - Click "Show thumbnails" expands the thumbnail grid
   - Thumbnails transition from dim → bright as they're captioned
   - Click any thumbnail opens lightbox with word-wrapped caption
   - Progress bar updates from frame events (no 2s lag)
   - No orphan row / wrapping issues in the minimap
5. Take screenshots to `tests/screenshots/`

**Step 2: Tune CSS values if needed**

Likely adjustments:
- Minimap `max-height: 120px` — may need to increase for large scenes
- `gap: 1px` on minimap — may want 2px for visual clarity
- Glow duration (1.5s setTimeout) — tune based on feel
- Thumb-cell size (56x32) — may want to adjust

**Step 3: Verify completed cards still work**

Completed cards use the horizontal filmstrip (unchanged). Verify:
- Completed cards render normally
- Expand to show detail grid
- Lightbox works on completed card frames

**Step 4: Final commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "polish(dashboard): tune minimap visual feel after live testing"
```

---

## Task Summary

| Task | What | Files |
|---|---|---|
| 1 | Backend: `frame_statuses` payload | `caption_dashboard.py` |
| 2 | CSS: minimap, thumb-grid, glow transitions | `caption_dashboard.html` (styles) |
| 3 | HTML/JS: active card template with minimap | `caption_dashboard.html` (template + toggleGrid) |
| 4 | JS: real-time `animateFrameIn` for minimap+grid | `caption_dashboard.html` (event handler) |
| 5 | JS: remove `syncFilmstrip`, clean up old grid code | `caption_dashboard.html` (cleanup) |
| 6 | Visual polish + live testing | both files |
