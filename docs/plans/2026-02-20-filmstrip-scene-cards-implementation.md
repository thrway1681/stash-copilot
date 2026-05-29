# Filmstrip Scene Cards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign scene cards with filmstrip layout, frame-by-frame streaming animations, and collapsed completed cards.

**Architecture:** The runner emits `FRAME_DONE` log lines per-frame, which the dashboard's broadcaster parses from the existing log deque and pushes as a `frame_events` WS channel. The frontend tracks recent frame events per-scene, animating new frames sliding into a filmstrip. Completed cards collapse to a single header row.

**Tech Stack:** Python 3.12 (backend), vanilla JS/CSS (frontend), WebSocket (real-time push)

---

### Task 1: Add FRAME_DONE Log Lines to Caption Runner

**Files:**
- Modify: `tools/dataset/caption_runner.py:81-141` (`_caption_one_frame`)
- Modify: `tools/dataset/caption_runner.py:144-202` (`process_scene_frames`)
- Modify: `tools/dataset/caption_runner.py:407-433` (fill pass)
- Modify: `tools/dataset/caption_runner.py:778-802` (retry pass)

**Context:** The runner currently only logs ERROR frames during normal captioning. We need a structured log line for every frame completion (success AND error) so the dashboard can stream them.

**Step 1: Add FRAME_DONE to `process_scene_frames()`**

In `process_scene_frames()`, after each future completes (line 177-188), emit the log line. The callback signature needs to change to pass frame info.

Change the `on_frame_done` callback type from `Callable[[], None]` to `Callable[[str, str, bool], None]` and call it with `(image_name, caption, was_error)`:

```python
# In process_scene_frames(), line 174-188:
for future in as_completed(futures):
    fp = futures[future]
    try:
        results[fp] = future.result()
    except (BudgetExhausted, DailyLimitReached) as e:
        budget_stop = e
        for f in futures:
            f.cancel()
        break
    except Exception as e:
        img_name = dataset_image_name(scene_id, Path(fp))
        results[fp] = (img_name, f"[ERROR: {e}]", True)
    if on_frame_done:
        img_name, caption, was_error = results[fp]
        on_frame_done(img_name, caption, was_error)
```

Update the type annotation on line 154:

```python
on_frame_done: Callable[[str, str, bool], None] | None = None,
```

**Step 2: Update `_on_frame_done` in the main loop**

In the main loop (line 485-492), update the closure to accept the new args and emit the FRAME_DONE line:

```python
def _on_frame_done(img_name: str, caption: str, was_error: bool) -> None:
    nonlocal frames_done_in_scene
    frames_done_in_scene += 1
    checkpoint["total_frames_captioned"] = (
        base_frames_captioned + frames_done_in_scene
    )
    _save_checkpoint(checkpoint_path, checkpoint)
    # Structured frame event for dashboard streaming
    status = "ERROR" if was_error else "OK"
    # Truncate caption to 200 chars to keep log lines bounded
    cap_preview = caption[:200].replace("\n", " ")
    _log(f"FRAME_DONE {img_name} {status} {cap_preview}")
```

**Step 3: Add FRAME_DONE to fill pass (line 421, 433)**

After `FILLED` log line (line 421), add:

```python
_log(f"FRAME_DONE {jpg.name} OK {caption[:200].replace(chr(10), ' ')}")
```

After `FILL ERROR` log line (line 433), add:

```python
_log(f"FRAME_DONE {jpg.name} ERROR {e}")
```

**Step 4: Add FRAME_DONE to retry pass (line 789, 802)**

After `FIXED` log line (line 789), add:

```python
_log(f"FRAME_DONE {img_name} OK {caption[:200].replace(chr(10), ' ')}")
```

After `FAILED` log line (line 802), add:

```python
_log(f"FRAME_DONE {img_name} ERROR {e}")
```

**Step 5: Verify and commit**

Run: `uv run python -c "from tools.dataset.caption_runner import process_scene_frames; print('OK')"`

```bash
git add tools/dataset/caption_runner.py
git commit -m "feat(runner): emit FRAME_DONE log line per frame completion"
```

---

### Task 2: Add Frame Event Parsing and WS Channel to Dashboard Backend

**Files:**
- Modify: `tools/dataset/caption_dashboard.py` (new function + broadcaster change)

**Context:** The broadcaster needs to parse `FRAME_DONE` lines from the runner's log buffer and push them as a `frame_events` WS channel. The log buffer is a `deque(maxlen=500)` that already captures runner output.

**Step 1: Add `_extract_frame_events()` function**

Add after the `load_error_count` function (around line 475):

```python
# ── Frame event extraction ────────────────────────────────────────────

_frame_event_cursor: int = 0  # Index into log to avoid re-sending old events


def _extract_new_frame_events(max_events: int = 20) -> list[dict[str, str]]:
    """Parse FRAME_DONE lines from the runner log, returning only unseen events.

    Each event: {"frame": "s1042_f0029.jpg", "status": "OK"|"ERROR",
                 "caption": "A woman...", "scene_id": "1042"}
    """
    global _frame_event_cursor

    if not runner_manager.is_running:
        _frame_event_cursor = 0
        return []

    log_lines = runner_manager.get_log(200)
    events: list[dict[str, str]] = []

    # Find new FRAME_DONE lines we haven't sent yet
    for i, line in enumerate(log_lines):
        if i < _frame_event_cursor:
            continue
        if not line.startswith("FRAME_DONE "):
            continue
        # Format: FRAME_DONE s1042_f0029.jpg OK A woman in a red dress...
        parts = line.split(" ", 3)
        if len(parts) < 3:
            continue
        frame = parts[1]
        status = parts[2]
        caption = parts[3] if len(parts) > 3 else ""
        # Extract scene_id from frame name (s1042_f0029.jpg -> 1042)
        sid = ""
        if frame.startswith("s") and "_f" in frame:
            sid = frame[1:].split("_f")[0]
        events.append({
            "frame": frame,
            "status": status,
            "caption": caption,
            "scene_id": sid,
        })

    # Advance cursor to end of log
    _frame_event_cursor = len(log_lines)

    # Return only the last max_events
    return events[-max_events:]
```

**Step 2: Add `frame_events` to broadcaster `_tick()`**

In `_tick()`, add a new channel at 1s cadence (every tick). Add after the runner check (around line 1241):

```python
        # Every 1s: frame events (individual frame completions)
        frame_events = _extract_new_frame_events(max_events=20)
        if frame_events:
            # Don't use _diff_and_collect — frame events are always "new"
            messages.append(json.dumps({
                "type": "frame_events",
                "data": {"events": frame_events},
            }))
```

Note: We bypass `_diff_and_collect` here because frame events are inherently append-only — each batch of events is always new data.

**Step 3: Add frame events to init event**

In `_send_init_event()` (around line 940), add frame events to the init payload:

```python
    frame_events = _extract_new_frame_events(max_events=20)
```

And add to the payload dict:

```python
            "frame_events": {"events": frame_events},
```

**Step 4: Verify and commit**

Run: `uv run python -c "from tools.dataset.caption_dashboard import _extract_new_frame_events; print('OK')"`

```bash
git add tools/dataset/caption_dashboard.py
git commit -m "feat(dashboard): add frame_events WS channel for per-frame streaming"
```

---

### Task 3: Redesign Active Scene Cards — CSS

**Files:**
- Modify: `tools/dataset/caption_dashboard.html` (CSS section)

**Context:** Replace the current scene card CSS with a filmstrip layout. Active cards show an open filmstrip with frames sliding in. This task is CSS-only; JS changes are in Tasks 4-5.

**Step 1: Replace scene card CSS**

Replace the `/* ── Scene Cards */` section (lines 401-646) with the new filmstrip design. Key changes:

- `.scene-card` — keep as container, same border/radius
- `.scene-card-head` — more compact: scene ID, frame counter, status pill in a row
- `.scene-filmstrip` — new element: horizontal flex with `overflow-x: auto`, `gap: 6px`, smooth scroll
- `.scene-filmstrip-frame` — individual frame tile: `100x56px`, rounded, with status overlay
- `.scene-filmstrip-frame.new` — slide-in animation from right + glow
- `.scene-filmstrip-frame .frame-status` — small icon overlay (checkmark or spinner)
- `.scene-latest-caption` — single-line caption preview below filmstrip, monospace, fade-in
- `.scene-card.completed` — collapsed: no filmstrip, no caption, just the header row
- `.scene-card.completed.expanded` — reveals filmstrip on click

```css
/* ── Scene Cards ────────────────────────────── */
.scenes {
  display: flex; flex-direction: column; gap: 0.5rem;
}

.scene-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  transition: border-color 0.3s, transform 0.2s, box-shadow 0.3s;
  animation: cardSlideIn 0.35s ease-out both;
}

.scene-card:hover {
  border-color: var(--border-active);
  box-shadow: var(--glow-blue);
}

@keyframes cardSlideIn {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── Card Header ─────────────────────────────── */
.scene-card-head {
  display: flex; align-items: center; gap: 0.75rem;
  padding: 0.55rem 0.85rem;
  cursor: pointer;
  user-select: none;
}

.scene-id {
  font-family: var(--font-mono);
  font-size: 0.78rem; font-weight: 600;
  color: var(--blue);
  white-space: nowrap;
}

.scene-meta {
  display: flex; gap: 0.5rem; flex: 1;
  font-size: 0.68rem; color: var(--text-3);
  font-family: var(--font-mono);
  align-items: center;
}

.scene-meta .tag {
  padding: 0.1rem 0.45rem;
  background: rgba(255,255,255,0.04);
  border-radius: 4px;
  white-space: nowrap;
}

.scene-meta .tag-frames { color: var(--cyan); }
.scene-meta .tag-time { color: var(--text-3); }
.scene-meta .tag-active {
  color: var(--amber);
  animation: activePulse 1.5s ease-in-out infinite;
}

@keyframes activePulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.scene-chevron {
  font-size: 0.65rem;
  color: var(--text-3);
  transition: transform 0.2s;
}

.scene-card.expanded .scene-chevron { transform: rotate(90deg); }

/* ── Progress Bar ────────────────────────────── */
.scene-progress {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0 0.85rem 0.4rem;
}

.scene-progress-track {
  flex: 1;
  height: 4px;
  background: rgba(255,255,255,0.04);
  border-radius: 2px;
  overflow: hidden;
}

.scene-progress-fill {
  height: 100%;
  border-radius: 2px;
  background: var(--gradient-primary);
  transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}

.scene-progress-fill.has-errors {
  background: linear-gradient(90deg, var(--green) 0%, var(--green) var(--ok-pct), var(--red) var(--ok-pct), var(--red) 100%);
}

.scene-progress-label {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-3);
  white-space: nowrap;
  min-width: 5.5rem;
  text-align: right;
}

.scene-progress-label .err { color: var(--red); }

/* ── In-Progress Card ────────────────────────── */
.scene-card.in-progress {
  border-color: rgba(255,176,0,0.3);
  animation: cardSlideIn 0.35s ease-out both, borderGlowAmber 2s ease-in-out infinite;
}

.scene-card.in-progress .scene-id { color: var(--amber); }

.scene-card.in-progress .scene-progress-fill {
  position: relative;
}

.scene-card.in-progress .scene-progress-fill::after {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
  animation: progressShimmer 2s ease-in-out infinite;
}

@keyframes borderGlowAmber {
  0%, 100% { border-color: rgba(255,176,0,0.15); box-shadow: 0 0 0 rgba(251,191,36,0); }
  50% { border-color: rgba(255,176,0,0.35); box-shadow: 0 0 15px rgba(251,191,36,0.08); }
}

/* ── Filmstrip ───────────────────────────────── */
.scene-filmstrip {
  display: flex;
  gap: 6px;
  padding: 0 0.85rem 0.5rem;
  overflow-x: auto;
  scroll-behavior: smooth;
  scrollbar-width: thin;
  scrollbar-color: rgba(96,165,250,0.2) transparent;
}

.scene-filmstrip::-webkit-scrollbar { height: 4px; }
.scene-filmstrip::-webkit-scrollbar-track { background: transparent; }
.scene-filmstrip::-webkit-scrollbar-thumb { background: rgba(96,165,250,0.2); border-radius: 2px; }

.scene-filmstrip-frame {
  position: relative;
  flex-shrink: 0;
  width: 100px; height: 56px;
  border-radius: 6px;
  overflow: hidden;
  background: var(--surface-2);
  border: 1px solid var(--border);
  cursor: pointer;
  transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}

.scene-filmstrip-frame:hover {
  border-color: var(--blue);
  transform: scale(1.06) translateY(-2px);
  box-shadow: 0 4px 16px rgba(96,165,250,0.2);
  z-index: 2;
}

.scene-filmstrip-frame img {
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
}

/* Frame status overlay */
.frame-status {
  position: absolute;
  top: 3px; right: 3px;
  width: 16px; height: 16px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 9px;
  backdrop-filter: blur(4px);
}

.frame-status.ok {
  background: rgba(52,211,153,0.85);
  color: white;
}

.frame-status.pending {
  background: rgba(251,191,36,0.85);
  color: white;
  animation: spinnerPulse 1s ease-in-out infinite;
}

.frame-status.error {
  background: rgba(248,113,113,0.85);
  color: white;
}

@keyframes spinnerPulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.85); }
}

/* Slide-in animation for new frames */
.scene-filmstrip-frame.new {
  animation: frameSlideIn 0.4s cubic-bezier(0.22, 1, 0.36, 1) both;
}

@keyframes frameSlideIn {
  from {
    opacity: 0;
    transform: translateX(30px) scale(0.9);
    box-shadow: 0 0 20px rgba(251,191,36,0.3);
  }
  to {
    opacity: 1;
    transform: translateX(0) scale(1);
    box-shadow: none;
  }
}

/* Glow on frame that just got captioned */
.scene-filmstrip-frame.just-captioned {
  animation: frameCaptionedGlow 1.5s ease-out forwards;
}

@keyframes frameCaptionedGlow {
  0% { border-color: rgba(52,211,153,0.5); box-shadow: 0 0 12px rgba(52,211,153,0.25); }
  100% { border-color: var(--border); box-shadow: none; }
}

.scene-filmstrip-overflow {
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  min-width: 40px;
  font-family: var(--font-mono);
  font-size: 0.62rem; font-weight: 600;
  color: var(--text-3);
}

/* ── Latest Caption Preview ──────────────────── */
.scene-latest-caption {
  padding: 0.35rem 0.85rem 0.55rem;
  font-size: 0.7rem;
  line-height: 1.4;
  color: var(--text-2);
  font-family: var(--font-mono);
  display: flex; align-items: baseline; gap: 0.4rem;
  overflow: hidden;
}

.scene-latest-caption .caption-label {
  color: var(--text-3);
  font-size: 0.6rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  flex-shrink: 0;
}

.scene-latest-caption .caption-text {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: opacity 0.3s;
}

.scene-latest-caption .caption-text.fade-in {
  animation: captionFadeIn 0.5s ease-out;
}

@keyframes captionFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── Completed Card (Collapsed) ──────────────── */
.scene-card.completed .scene-filmstrip,
.scene-card.completed .scene-latest-caption,
.scene-card.completed .scene-progress {
  display: none;
}

.scene-card.completed.expanded .scene-filmstrip,
.scene-card.completed.expanded .scene-progress {
  display: flex;
}

.scene-card.completed .scene-card-head {
  padding: 0.45rem 0.85rem;
}

/* Completed status pill in header */
.scene-status-pill {
  display: inline-flex; align-items: center; gap: 0.25rem;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 0.62rem; font-weight: 600;
}

.scene-status-pill.ok {
  background: var(--green-dim);
  color: var(--green);
}

.scene-status-pill.has-errors {
  background: var(--red-dim);
  color: var(--red);
}

/* ── Expanded Detail Grid ────────────────────── */
.scene-detail {
  display: none;
  padding: 0 0.85rem 0.75rem;
  border-top: 1px solid var(--border);
  margin-top: 0.4rem;
}

.scene-card.expanded .scene-detail { display: block; }

.scene-detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.5rem;
  margin-top: 0.65rem;
}

.frame-item {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
  animation: scaleIn 0.3s ease-out both;
}

.frame-item:hover {
  border-color: var(--border-active);
  transform: translateY(-3px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.3), var(--glow-blue);
}

.frame-item img {
  width: 100%; aspect-ratio: 16/9;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}

.frame-item:hover img { transform: scale(1.03); }

.frame-item .frame-cap {
  padding: 0.4rem 0.5rem;
  font-size: 0.68rem; line-height: 1.4;
  color: var(--text-2);
}

.frame-item .frame-cap.error { color: var(--red); }

.frame-item .frame-name {
  padding: 0.15rem 0.5rem 0.35rem;
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-3);
}
```

**Step 2: Keep the active section and just-completed CSS from prior commit** (already in file, no changes needed)

**Step 3: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "style(dashboard): filmstrip layout CSS for scene cards"
```

---

### Task 4: Rebuild Scene Card HTML Generation (JS)

**Files:**
- Modify: `tools/dataset/caption_dashboard.html` (JS `buildSceneCardHTML` function)

**Context:** Replace `buildSceneCardHTML()` to generate the new filmstrip structure for active cards and compact header for completed cards.

**Step 1: Replace `buildSceneCardHTML()`**

The function needs to output different HTML based on whether the scene is in-progress or completed:

**Active card structure:**
```html
<div class="scene-card in-progress" data-scene="1042">
  <div class="scene-card-head">
    <span class="scene-id">Scene 1042</span>
    <div class="scene-meta">
      <span class="tag tag-frames">48 frames</span>
      <span class="tag tag-active">captioning...</span>
    </div>
    <span class="scene-chevron">&#9656;</span>
  </div>
  <div class="scene-progress">...</div>
  <div class="scene-filmstrip" id="filmstrip-1042">
    <!-- frames injected by renderActiveScenes / frame events -->
  </div>
  <div class="scene-latest-caption" id="caption-1042">
    <span class="caption-label">Latest</span>
    <span class="caption-text"></span>
  </div>
  <div class="scene-detail" id="detail-1042"></div>
</div>
```

**Completed card structure:**
```html
<div class="scene-card completed" data-scene="1042">
  <div class="scene-card-head">
    <span class="scene-id">Scene 1042</span>
    <div class="scene-meta">
      <span class="tag tag-frames">48 frames</span>
      <span class="scene-status-pill ok">48 ok</span>
      <span class="tag tag-time">2m ago</span>
    </div>
    <span class="scene-chevron">&#9656;</span>
  </div>
  <div class="scene-progress">...</div>
  <div class="scene-filmstrip" id="filmstrip-1042"></div>
  <div class="scene-detail" id="detail-1042"></div>
</div>
```

Replace the function entirely. Key differences from current:
- Add `completed` class for non-in-progress cards
- Replace `.scene-thumbs` with `.scene-filmstrip`
- Replace `.scene-captions` with `.scene-latest-caption`
- For completed cards: add `.scene-status-pill` showing "N ok" or "N ok M err"
- Filmstrip frames are generated in a helper `buildFilmstripHTML(frames, isNew)`

**Step 2: Add `buildFilmstripHTML()` helper**

```javascript
function buildFilmstripHTML(sampleFrames, frameCount, isNew) {
  const frames = (sampleFrames || []).slice(0, 8).map((f, i) =>
    `<div class="scene-filmstrip-frame${isNew ? ' new' : ''}" style="${isNew ? 'animation-delay:' + (i * 80) + 'ms' : ''}">
      <img src="/assets/lora_dataset/images/${encodeURIComponent(f)}" alt="${f}" loading="lazy"
           onclick="event.stopPropagation(); window.__lightbox('/assets/lora_dataset/images/${encodeURIComponent(f)}')">
      <span class="frame-status ok">\u2713</span>
    </div>`
  ).join('');
  const overflow = frameCount > 8
    ? `<span class="scene-filmstrip-overflow">+${frameCount - 8}</span>`
    : '';
  return frames + overflow;
}
```

**Step 3: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "feat(dashboard): rebuild scene card HTML with filmstrip layout"
```

---

### Task 5: Update Render Functions and Frame Event Handling (JS)

**Files:**
- Modify: `tools/dataset/caption_dashboard.html` (JS render functions + WS dispatch)

**Context:** Update `renderActiveScenes()` and `renderCompletedScenes()` for the new card structure. Add frame event handling that animates new frames into the filmstrip.

**Step 1: Add frame event state**

In the state section, add:

```javascript
let sceneFrameEvents = {};   // scene_id -> [{frame, status, caption}, ...] (last 8 per scene)
```

**Step 2: Update `renderActiveScenes()`**

The active renderer should populate filmstrips and latest caption:

```javascript
function renderActiveScenes() {
  if (!activeData.length) {
    activeSection.style.display = 'none';
    return;
  }
  activeSection.style.display = '';
  activeCountEl.textContent = activeData.length + ' scene' + (activeData.length !== 1 ? 's' : '');

  // Check if the scene list changed
  const activeIds = activeData.map(s => String(s.scene_id));
  const existingCards = activeContainer.querySelectorAll('.scene-card[data-scene]');
  const existingIds = Array.from(existingCards).map(c => c.dataset.scene);
  const listChanged = activeIds.length !== existingIds.length
    || activeIds.some((id, i) => id !== existingIds[i]);

  if (listChanged) {
    activeContainer.innerHTML = activeData.map((s, i) => buildSceneCardHTML(s, i)).join('');
    // Populate filmstrips from current sample frames
    activeData.forEach(s => {
      const strip = document.getElementById('filmstrip-' + s.scene_id);
      if (strip) {
        strip.innerHTML = buildFilmstripHTML(s.sample_frames, s.frame_count, false);
        autoScrollFilmstrip(strip);
      }
      updateLatestCaption(s.scene_id, s.sample_captions);
    });
  } else {
    // In-place update: progress bars, frame counts, captions
    activeData.forEach(s => {
      const card = activeContainer.querySelector(`[data-scene="${s.scene_id}"]`);
      if (!card) return;
      updateCardProgress(card, s);
      updateLatestCaption(s.scene_id, s.sample_captions);
    });
  }
}
```

**Step 3: Update `renderCompletedScenes()`**

For completed scenes, the render is simpler since cards are collapsed:

```javascript
function renderCompletedScenes() {
  if (!completedData.length) {
    scenesContainer.innerHTML = `<div class="empty-state">
      <div class="empty-icon">&#9678;</div>
      <div>No scenes captioned yet</div>
    </div>`;
    return;
  }

  const incomingIds = completedData.map(s => String(s.scene_id));
  const existingCards = scenesContainer.querySelectorAll('.scene-card[data-scene]');
  const existingMap = new Map();
  existingCards.forEach(card => existingMap.set(card.dataset.scene, card));
  const existingIds = Array.from(existingMap.keys());
  const listChanged = incomingIds.length !== existingIds.length
    || incomingIds.some((id, i) => id !== existingIds[i]);

  if (listChanged) {
    scenesContainer.innerHTML = completedData.map((s, i) => {
      let html = buildSceneCardHTML(s, i);
      if (justCompletedIds.has(s.scene_id)) {
        html = html.replace('class="scene-card', 'class="scene-card just-completed');
      }
      return html;
    }).join('');
    if (justCompletedIds.size > 0) {
      const idsToRemove = new Set(justCompletedIds);
      setTimeout(() => {
        idsToRemove.forEach(id => {
          justCompletedIds.delete(id);
          const card = scenesContainer.querySelector(`.scene-card[data-scene="${id}"]`);
          if (card) card.classList.remove('just-completed');
        });
      }, 3000);
    }
  } else {
    completedData.forEach(s => {
      const card = existingMap.get(String(s.scene_id));
      if (!card) return;
      const timeTag = card.querySelector('.tag-time');
      if (timeTag) timeTag.textContent = timeAgo(s.captioned_at);
    });
  }
}
```

**Step 4: Add frame event handler in `dispatchWS()`**

Add a new case in the dispatch:

```javascript
} else if (t === 'frame_events') {
  const events = (d && d.events) || [];
  events.forEach(evt => {
    const sid = evt.scene_id;
    if (!sid) return;
    // Accumulate per-scene, capped at 8
    if (!sceneFrameEvents[sid]) sceneFrameEvents[sid] = [];
    sceneFrameEvents[sid].push(evt);
    if (sceneFrameEvents[sid].length > 8) {
      sceneFrameEvents[sid] = sceneFrameEvents[sid].slice(-8);
    }
    // Animate frame into filmstrip
    animateFrameIn(sid, evt);
  });
```

Also update `init` handler to process initial frame events:

```javascript
// In init handler, after setting activeData/completedData:
const initFrameEvents = (d.frame_events && d.frame_events.events) || [];
initFrameEvents.forEach(evt => {
  const sid = evt.scene_id;
  if (!sid) return;
  if (!sceneFrameEvents[sid]) sceneFrameEvents[sid] = [];
  sceneFrameEvents[sid].push(evt);
  if (sceneFrameEvents[sid].length > 8) {
    sceneFrameEvents[sid] = sceneFrameEvents[sid].slice(-8);
  }
});
```

**Step 5: Add `animateFrameIn()` function**

```javascript
function animateFrameIn(sceneId, evt) {
  const strip = document.getElementById('filmstrip-' + sceneId);
  if (!strip) return;  // Scene not in active view

  // Check if this frame is already in the filmstrip
  const existing = strip.querySelector(`[data-frame="${evt.frame}"]`);
  if (existing) {
    // Frame already there — just update status
    const status = existing.querySelector('.frame-status');
    if (status) {
      status.className = 'frame-status ' + (evt.status === 'OK' ? 'ok' : 'error');
      status.textContent = evt.status === 'OK' ? '\u2713' : '\u2717';
    }
    existing.classList.add('just-captioned');
    setTimeout(() => existing.classList.remove('just-captioned'), 1500);
    return;
  }

  // Create new frame element
  const frameEl = document.createElement('div');
  frameEl.className = 'scene-filmstrip-frame new';
  frameEl.dataset.frame = evt.frame;
  frameEl.innerHTML = `
    <img src="/assets/lora_dataset/images/${encodeURIComponent(evt.frame)}" alt="${evt.frame}" loading="lazy"
         onclick="event.stopPropagation(); window.__lightbox('/assets/lora_dataset/images/${encodeURIComponent(evt.frame)}')">
    <span class="frame-status ${evt.status === 'OK' ? 'ok' : 'error'}">${evt.status === 'OK' ? '\u2713' : '\u2717'}</span>
  `;

  // Insert before overflow counter, or at end
  const overflow = strip.querySelector('.scene-filmstrip-overflow');
  if (overflow) {
    strip.insertBefore(frameEl, overflow);
    // Update overflow count
    const total = parseInt(overflow.textContent.replace('+', '')) || 0;
    // Don't update — the count comes from frame_count which updates via active_scenes
  } else {
    strip.appendChild(frameEl);
  }

  // Remove .new class after animation completes
  frameEl.addEventListener('animationend', () => frameEl.classList.remove('new'), { once: true });

  // Auto-scroll to show newest frame
  autoScrollFilmstrip(strip);

  // Update latest caption
  if (evt.status === 'OK' && evt.caption) {
    updateLatestCaption(sceneId, [evt.caption]);
  }
}

function autoScrollFilmstrip(strip) {
  requestAnimationFrame(() => {
    strip.scrollLeft = strip.scrollWidth;
  });
}

function updateLatestCaption(sceneId, captions) {
  const container = document.getElementById('caption-' + sceneId);
  if (!container) return;
  const textEl = container.querySelector('.caption-text');
  if (!textEl) return;
  const latest = (captions || []).filter(c => c && !c.startsWith('[ERROR')).pop();
  if (latest && textEl.textContent !== latest) {
    textEl.textContent = latest;
    textEl.classList.remove('fade-in');
    void textEl.offsetWidth;
    textEl.classList.add('fade-in');
  }
}

function updateCardProgress(card, s) {
  const errCount = s.error_count || 0;
  const captionedCount = s.captioned_count || 0;
  const total = s.frame_count || 1;
  const pctVal = (captionedCount + errCount) / total * 100;
  const hasErrors = errCount > 0;

  const fill = card.querySelector('.scene-progress-fill');
  if (fill) {
    fill.classList.toggle('has-errors', hasErrors);
    fill.style.cssText = hasErrors
      ? `width:${pctVal.toFixed(1)}%; --ok-pct:${(total ? captionedCount / (captionedCount + errCount) * 100 : 100).toFixed(1)}%`
      : `width:${pctVal.toFixed(1)}%`;
  }

  const label = card.querySelector('.scene-progress-label');
  if (label) {
    const parts = [`${captionedCount}/${total}`];
    if (errCount) parts.push(`<span class="err">${errCount} err</span>`);
    label.innerHTML = parts.join(' ');
  }

  const framesTag = card.querySelector('.tag-frames');
  if (framesTag) framesTag.textContent = s.frame_count + ' frames';
}
```

**Step 6: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "feat(dashboard): frame-by-frame streaming with filmstrip animations"
```

---

### Task 6: Update Completed Card Expand to Load Filmstrip

**Files:**
- Modify: `tools/dataset/caption_dashboard.html` (JS `__toggleScene` and `loadSceneDetail`)

**Context:** When a completed card is expanded, load frames via `get_scene_frames` API and populate the filmstrip (not just the detail grid). Keep the detail grid for the expanded view but also fill the filmstrip.

**Step 1: Update `loadSceneDetail()`**

After frames load, also populate the filmstrip:

```javascript
async function loadSceneDetail(sceneId) {
  const container = document.getElementById('detail-' + sceneId);
  const strip = document.getElementById('filmstrip-' + sceneId);

  if (sceneFramesCache[sceneId]) {
    if (container) renderDetail(container, sceneFramesCache[sceneId]);
    if (strip && !strip.children.length) {
      populateCompletedFilmstrip(strip, sceneFramesCache[sceneId]);
    }
    return;
  }

  if (container) container.innerHTML = '<div style="padding:0.5rem;color:var(--text-3);font-size:0.72rem">Loading frames&hellip;</div>';

  if (wsSend({type: 'get_scene_frames', scene_id: sceneId})) return;

  try {
    const data = await fetchJSON('/api/scene/' + sceneId + '/frames');
    sceneFramesCache[sceneId] = data.frames || [];
    if (container) renderDetail(container, data.frames || []);
    if (strip) populateCompletedFilmstrip(strip, data.frames || []);
  } catch (e) {
    if (container) container.innerHTML = '<div style="padding:0.5rem;color:var(--red);font-size:0.72rem">Failed to load: ' + escHTML(e.message) + '</div>';
  }
}
```

**Step 2: Add `populateCompletedFilmstrip()` helper**

```javascript
function populateCompletedFilmstrip(strip, frames) {
  // Show up to 8 evenly-spaced frames from the full set
  let sample;
  if (frames.length <= 8) {
    sample = frames;
  } else {
    const step = frames.length / 8;
    sample = Array.from({length: 8}, (_, i) => frames[Math.floor(i * step)]);
  }
  strip.innerHTML = sample.map(f => {
    const isErr = f.caption && f.caption.startsWith('[ERROR');
    return `<div class="scene-filmstrip-frame" data-frame="${escHTML(f.image_name)}">
      <img src="/assets/lora_dataset/images/${encodeURIComponent(f.image_name)}" alt="${escHTML(f.image_name)}" loading="lazy"
           onclick="event.stopPropagation(); window.__lightbox('/assets/lora_dataset/images/${encodeURIComponent(f.image_name)}')">
      <span class="frame-status ${isErr ? 'error' : 'ok'}">${isErr ? '\u2717' : '\u2713'}</span>
    </div>`;
  }).join('') + (frames.length > 8 ? `<span class="scene-filmstrip-overflow">+${frames.length - 8}</span>` : '');
}
```

**Step 3: Update `scene_frames` WS handler to also populate filmstrip**

In the `dispatchWS` `scene_frames` handler, add:

```javascript
const strip = document.getElementById('filmstrip-' + sid);
if (strip && !strip.children.length) {
  populateCompletedFilmstrip(strip, d.frames || []);
}
```

**Step 4: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "feat(dashboard): populate filmstrip on completed card expand"
```

---

### Task 7: Clean Up and Final Verification

**Files:**
- All modified files

**Step 1: Verify Python syntax**

```bash
uv run python -c "import py_compile; py_compile.compile('tools/dataset/caption_runner.py', doraise=True)"
uv run python -c "import py_compile; py_compile.compile('tools/dataset/caption_dashboard.py', doraise=True)"
```

**Step 2: Verify JS syntax**

```bash
node -e "const fs=require('fs'); const h=fs.readFileSync('tools/dataset/caption_dashboard.html','utf8'); const m=h.match(/<script>([\s\S]*?)<\/script>/); new Function(m[1]); console.log('JS OK')"
```

**Step 3: Run tests**

```bash
uv run pytest tests/tools/test_caption_dashboard.py -v
```

**Step 4: Manual verification checklist**

- [ ] Start dashboard with no runner → "Now Captioning" section hidden, completed cards collapsed
- [ ] Start a caption run → "Now Captioning" appears, filmstrip populates, frames slide in
- [ ] Watch frame events arrive in DevTools WS messages
- [ ] Wait for scene to complete → slides to "Recently Captioned" with green glow, collapses
- [ ] Click completed card → expands, filmstrip + detail grid load
- [ ] Stop runner → "Now Captioning" disappears, retrying tags clear
- [ ] Kill server → HTTP fallback still works

**Step 5: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore(dashboard): cleanup filmstrip scene card implementation"
```
