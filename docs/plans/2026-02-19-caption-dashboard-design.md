# Caption Pipeline Dashboard Design

**Date:** 2026-02-19
**Goal:** Live-updating web dashboard to monitor caption pipeline progress, API cost, rate limits, and recently processed frames.

## Decisions

| Setting | Value | Rationale |
|---|---|---|
| Architecture | Separate server process | Decoupled from runner — start/stop independently, no IPC |
| Update mechanism | Polling (2s status, 10s scenes) | Data changes every 2-30s per scene; polling is indistinguishable from push at this cadence |
| Framework | Vanilla JS SPA | Matches caption workbench and Stash Copilot patterns — no build step, no dependencies |
| Server | `ThreadingMixIn + HTTPServer` | Same pattern as `caption_workbench.py` |
| Design language | Stash Copilot AI Insights modal | Dark gradients, blue/purple/cyan accents, stat cards, glow effects, animated AI orb |
| Scene history | Configurable window (10-500, default 50) | Summary stats always cover everything; detail cards for recent N |

## Architecture

```
caption_runner.py                    caption_dashboard.py
┌──────────────────┐                ┌──────────────────────┐
│ Processes scenes  │                │ Python HTTP server    │
│ Writes:           │                │ Reads:                │
│  • progress.json  │───files───────▶│  • progress.json      │
│  • budget.json    │                │  • budget.json        │
│  • metadata.jsonl │                │  • metadata.jsonl     │
│  • images/*.jpg   │                │ Serves:               │
│  • images/*.txt   │                │  • /  (HTML dashboard)│
└──────────────────┘                │  • /api/status        │
                                    │  • /api/scenes        │
                                    │  • /api/scene/<id>    │
                                    │  • /assets/...        │
                                    └───────────┬──────────┘
                                                │
                                    Browser polls /api/status
                                    every 2 seconds
                                                │
                                    ┌───────────▼──────────┐
                                    │ caption_dashboard.html│
                                    │ Vanilla JS SPA        │
                                    └──────────────────────┘
```

**Default port:** 8766

## API Endpoints

| Endpoint | Method | Poll Interval | Returns |
|---|---|---|---|
| `/` | GET | — | Dashboard HTML |
| `/api/status` | GET | 2s | Budget state + progress + rate limits |
| `/api/scenes?n=50` | GET | 10s | Last N completed scenes from metadata.jsonl |
| `/api/scene/<id>/frames` | GET | on-demand | Frame thumbnails + captions for expanded card |
| `/assets/lora_dataset/images/<file>` | GET | — | Captioned frame JPEGs |

### `/api/status` Response

```json
{
  "budget": {
    "total_calls": 1234,
    "total_input_tokens": 1851000,
    "total_output_tokens": 98640,
    "total_cost": 1.30,
    "total_errors": 3,
    "max_cost": 50.0,
    "rpd_count": 1234,
    "rpd_date": "2026-02-19"
  },
  "progress": {
    "completed_scenes": 47,
    "total_scenes": 12762,
    "total_frames_captioned": 8930,
    "estimated_total_frames": 2419013,
    "errors": 3,
    "last_updated": "2026-02-19T14:32:00Z"
  },
  "rate_limits": {
    "rpm_limit": 900,
    "tpm_limit": 900000,
    "rpd_limit": 9500
  },
  "pricing": {
    "model": "gemini-3.0-flash-preview",
    "input_per_m": 0.50,
    "output_per_m": 3.00
  },
  "runner_active": true
}
```

`runner_active` is true if `budget_state.json` was modified within the last 60 seconds.

### `/api/scenes?n=50` Response

```json
{
  "scenes": [
    {
      "scene_id": 10065,
      "frame_count": 84,
      "cost": 0.088,
      "captioned_at": "2026-02-19T14:30:00Z",
      "selection": { "novelty_count": 42, "temporal_count": 42 },
      "sample_captions": ["first caption...", "second caption..."],
      "sample_frames": ["s10065_f0001.jpg", "s10065_f0042.jpg", "s10065_f0084.jpg"]
    }
  ]
}
```

## UI Layout

### Design System (matches AI Insights modal)

- **Background:** `linear-gradient(180deg, #1e242d, #1a2030, #181c23)`
- **Accents:** Blue `#60a5fa`, purple `#8b5cf6`, cyan `#06b6d4`
- **Cards:** `rgba(0,0,0,0.3)` bg, `1px solid rgba(255,255,255,0.08)` border, `8px` radius
- **Stat items:** `linear-gradient(135deg, rgba(96,165,250,0.08), rgba(139,92,246,0.05))`
- **Typography:** `0.6rem` uppercase labels in `#6b7280`, `1.15rem` bold values in `#60a5fa`, `0.8rem` body in `#d1d5db`
- **Scrollbar:** `8px` webkit track with blue-tinted thumb `rgba(96,165,250,0.3)`
- **Hover effects:** `translateY(-3px)` + glow `box-shadow` with accent color
- **Animations:** AI pulse orb, `unifiedCardFadeIn` for scene cards

### Sections

**1. Header bar** — fixed top, modal header gradient
```
[AI orb] Caption Pipeline Dashboard          [● active]  [gemini-3.0-flash]
```
- Animated AI pulse orb (same `aiPulse` keyframes)
- Runner status: green dot when active, gray when idle
- Model name pill badge

**2. Metrics row** — 6 stat items in flex row
```
┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│   $1.30     │   47       │   8,930    │   847      │   892K     │   1,234    │
│   COST      │   SCENES   │   FRAMES   │   RPM      │   TPM      │   RPD      │
│  of $50.00  │  of 12,762 │  of 2.42M  │  lim 900   │  lim 900K  │  lim 9,500 │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```
Each has a progress bar underneath using `linear-gradient(90deg, #60a5fa, #8b5cf6)`.

**3. Progress bar** — full-width below metrics
```
[████████░░░░░░░░░░░░░░░░░░░░░░░░░] 0.37%  ·  ETA: 48h 12m  ·  3 errors
```
Uses primary button gradient `linear-gradient(135deg, #3b82f6, #6366f1)` with blue glow.

**4. Scene cards** — scrollable, configurable window
```
Window: [─────●─────] 50 scenes

┌─ Scene 10065 ──────────── 2m ago ── 84 frames ── $0.088 ─┐
│ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ +79   │
│ │ thumb │ │ thumb │ │ thumb │ │ thumb │ │ thumb │        │
│ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘        │
│ "Doggy style, fit brunette with tattoos..."               │
│ "Close-up POV blowjob, brown eyes..."                     │
│                                      ▼ Show all captions  │
└───────────────────────────────────────────────────────────┘
```
- Cards animate in with `unifiedCardFadeIn`
- Hover lifts card with glow
- 5 sample thumbnails (evenly spaced from scene's frames)
- 2 sample captions shown, expandable to show all
- Cost badge with accent gradient pill
- Newest scenes at top

## Data Sources

| Dashboard needs | Source file | Updated |
|---|---|---|
| Cost, tokens, errors, RPD | `budget_state.json` | After each scene |
| Completed scenes, frame count | `caption_progress.json` | After each scene |
| Per-scene detail, captions | `metadata.jsonl` (tail N lines) | After each scene |
| Frame images | `assets/lora_dataset/images/*.jpg` | As captioned |
| Total scene count | `frame_search_*_info.json` | Static |

## Files

- `tools/dataset/caption_dashboard.py` — HTTP server
- `tools/dataset/caption_dashboard.html` — SPA (HTML + CSS + JS in one file)
