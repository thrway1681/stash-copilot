# Active Scene Grid Redesign

**Date:** 2026-02-24
**Status:** Approved
**Scope:** `caption_dashboard.html`, `caption_dashboard.py`

## Problem

The active (in-progress) scene card uses a 16-slot fixed grid that wraps poorly — orphan thumbnails land on a second row. Animations are jerky (`scale(0.92)→1` pop-in). The 2s `syncFilmstrip` polling fallback is unnecessary overhead now that WebSocket frame events use set-based dedup.

## Design

### Minimap (Default/Collapsed State)

- Compact grid of tiny cells (~12×7px, 1px gap) representing **every frame** in the scene.
- Cell color by status:
  - **Pending:** `rgba(255,255,255,0.06)` — barely visible dark squares.
  - **OK:** `var(--green)` at ~60% opacity.
  - **Error:** `var(--red)` at ~60% opacity.
  - **Just captioned (last ~2s):** brief green glow pulse, then settles to OK color.
- Below the minimap: "Latest" caption preview (unchanged).

### Expanded State (Click to Expand)

- Full grid of actual thumbnail images (~56×32px each).
- Uncaptioned: `opacity: 0.3; filter: saturate(0.2) brightness(0.5)` — visible but dim.
- Captioned: full brightness, normal saturation.
- Scrollable at `max-height: ~400px`.
- Click any thumbnail → lightbox modal with full image + word-wrapped caption.

### Animation: Soft Glow Transition

- **Minimap cells:** `background-color` CSS transition over 0.6s + `box-shadow` glow that fades over 1.5s. No scale/transform.
- **Expanded thumbnails:** `opacity` + `filter` transition over 0.6s, green border glow fading over 1.5s.
- **Expand/collapse:** `max-height` transition with `overflow: hidden`.

### Lightbox

- Existing modal. Fix caption to use `white-space: normal; word-wrap: break-word` for proper wrapping.

### Data Flow (Real-time Only)

| Mechanism | Role |
|---|---|
| `frame_statuses` in init / `active_scenes` push | One-time seed on scene appear or WS reconnect |
| `FRAME_DONE` WebSocket events | All real-time updates, sole mechanism |
| ~~`syncFilmstrip` 2s fallback~~ | **Removed** |
| ~~`sample_frames` for active cards~~ | **Removed** (kept only for completed cards) |

#### Backend

- `load_in_progress_scenes()` returns new `frame_statuses: [{frame, status}]` for all frames (no caption text — too large).
- Caption text is lazy-loaded on lightbox click via existing endpoint.

#### Frontend

- `sceneFrameStatuses[sceneId]` — Map of `frameName → status` per active scene.
- Seeded from `frame_statuses` on init.
- Updated exclusively by `FRAME_DONE` events.
- On WS reconnect: `init` event re-seeds full state.

### Summary of Changes

| Component | Current | New |
|---|---|---|
| Grid CSS | `repeat(auto-fill, 72px)`, 16 fixed slots | Two modes: minimap (tiny cells) + expanded (real thumbnails) |
| Frame count | Hardcoded 16 | All frames in scene |
| Uncaptioned frames | Empty dashed slots | Visible but dim |
| Fill animation | `scale(0.92)→1` (jerky) | Color/opacity 0.6s + glow fade 1.5s |
| Backend payload | `sample_frames` (16 items) | `frame_statuses` (all items, status only) |
| Lightbox caption | Truncated | Proper word-wrap |
| Sync mechanism | WS events + 2s polling fallback | WS events only |
