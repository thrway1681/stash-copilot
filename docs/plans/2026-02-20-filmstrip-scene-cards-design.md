# Filmstrip Scene Cards with Frame-by-Frame Streaming

**Date:** 2026-02-20
**Status:** Approved

## Problem

1. Active and completed scenes share the same card layout — no visual distinction
2. No frame-by-frame animation — thumbnails are replaced wholesale every 2s
3. Completed scenes take too much vertical space when there are 50+

## Design

### Backend: Frame Completion Events

**Runner** (`caption_runner.py`): Emit a structured log line after each frame:

```
FRAME_DONE s1042_f0029.jpg OK A woman in a red dress...
FRAME_DONE s1042_f0030.jpg ERROR Rate limit exceeded
```

Emitted from the `on_frame_done` path (and retry/fill paths). Written to stdout (already captured by dashboard's `_read_output()` into the existing `deque(maxlen=500)`).

**Dashboard** (`caption_dashboard.py`): New broadcaster channel `"frame_events"`:
- Parse last N log lines for `FRAME_DONE` pattern
- Track last-seen index to emit only new events per tick (every 1s)
- Cap at 20 most recent events per broadcast
- Zero memory growth (fixed-size deque + integer counter)

### Frontend: Active Scene Cards (Filmstrip)

Open filmstrip layout showing last 8 frames per scene:
- New frames slide in from the right (`@keyframes frameSlideIn`)
- Each frame has a status overlay (checkmark for captioned, spinner for pending)
- Latest caption text fades in below the filmstrip
- Progress bar with live counter (`29/48 captioned`)
- Filmstrip auto-scrolls right to keep newest frame visible

### Frontend: Completed Scene Cards (Collapsed)

Single header row by default:
```
> Scene 1042    48 frames    48 ok    2m ago
```

Click to expand reveals full detail grid (existing `get_scene_frames` API).
Just-completed cards retain green glow animation from prior work.

### Resource Safety

| Component | Bound | Growth |
|-----------|-------|--------|
| Runner log buffer | 500 lines | Fixed (deque) |
| Frame event tracker | 1 int + 20 events | Constant |
| Frontend frame store | ~8 frames x active scenes | ~24 max |
| Completed cards DOM | 1 div each (collapsed) | No images until expanded |
| WS broadcast | MD5 diff dedup | Unchanged |

## Files to Modify

- `tools/dataset/caption_runner.py` — Add `FRAME_DONE` log lines
- `tools/dataset/caption_dashboard.py` — Add frame event parsing + WS channel
- `tools/dataset/caption_dashboard.html` — New card CSS/HTML/JS
