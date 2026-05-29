# Caption Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a live-updating web dashboard that monitors the caption pipeline's progress, API cost, rate limits, and recently processed frames.

**Architecture:** Separate Python HTTP server (`caption_dashboard.py`) polls the same output files that `caption_runner.py` writes (`budget_state.json`, `caption_progress.json`, `metadata.jsonl`). Serves a vanilla JS single-page app (`caption_dashboard.html`) that polls `/api/status` every 2s and `/api/scenes` every 10s. Uses the Stash Copilot AI Insights design language (dark gradients, blue/purple/cyan accents, stat cards, glow effects).

**Tech Stack:** Python 3.12, `http.server` (stdlib), vanilla JavaScript, CSS (no build step, no dependencies).

**Design doc:** `docs/plans/2026-02-19-caption-dashboard-design.md`

**Key data files the dashboard reads (written by caption_runner):**
- `assets/lora_dataset/budget_state.json` — cost, tokens, RPD, errors
- `assets/lora_dataset/caption_progress.json` — completed scenes, frame count
- `assets/lora_dataset/metadata.jsonl` — per-scene detail (captions, selection stats)
- `assets/lora_dataset/images/*.jpg` — captioned frame images
- `assets/lora_dataset/images/*.txt` — caption text files
- `assets/frame_search_openclip-ViT-H-14_info.json` — total scene/frame counts (static)

---

### Task 1: Dashboard Server — `caption_dashboard.py`

**Files:**
- Create: `tools/dataset/caption_dashboard.py`
- Test: `tests/tools/test_caption_dashboard.py`

The HTTP server reads runner output files and exposes them as JSON API endpoints. It also serves the HTML dashboard and frame images.

**Step 1: Write the failing tests**

Create `tests/tools/test_caption_dashboard.py`:

```python
"""Tests for tools.dataset.caption_dashboard — HTTP server for pipeline monitoring."""
from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

import pytest

from tools.dataset.caption_dashboard import (
    DashboardState,
    load_status,
    load_recent_scenes,
    load_scene_frames,
)


# ── DashboardState unit tests ────────────────────────────────────────────


def _write_budget(path: Path, **overrides: Any) -> None:
    data = {
        "total_calls": 100,
        "total_input_tokens": 150_000,
        "total_output_tokens": 8_000,
        "total_cost": 0.105,
        "total_errors": 2,
        "rpd_count": 100,
        "rpd_date": "2026-02-19",
        "saved_at": "2026-02-19T14:00:00Z",
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_progress(path: Path, **overrides: Any) -> None:
    data = {
        "completed_scenes": [1, 2, 3],
        "total_frames_captioned": 250,
        "errors": 2,
        "last_updated": "2026-02-19T14:00:00Z",
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_info(path: Path) -> None:
    data = {
        "model_key": "openclip:ViT-H-14",
        "frame_count": 4_074_898,
        "scene_count": 12_762,
        "dimensions": 1024,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_metadata(path: Path, scenes: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for scene in scenes:
            f.write(json.dumps(scene) + "\n")


def _sample_metadata_record(scene_id: int, n_frames: int = 5) -> dict[str, Any]:
    image_names = [f"s{scene_id}_f{i:04d}.jpg" for i in range(n_frames)]
    captions = {name: f"Caption for {name}" for name in image_names}
    return {
        "scene_id": scene_id,
        "image_names": image_names,
        "captions": captions,
        "selection": {"novelty_count": 3, "temporal_count": 2},
        "captioned_at": "2026-02-19T14:00:00Z",
        "method": "gemini-vlm+smart-select",
    }


def test_load_status_combines_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    _write_budget(output_dir / "budget_state.json")
    _write_progress(output_dir / "caption_progress.json")
    _write_info(assets_dir / "frame_search_openclip-ViT-H-14_info.json")

    status = load_status(output_dir, assets_dir)

    assert status["budget"]["total_calls"] == 100
    assert status["budget"]["total_cost"] == 0.105
    assert status["progress"]["completed_scenes"] == 3
    assert status["progress"]["total_frames_captioned"] == 250
    assert status["progress"]["total_scenes"] == 12_762
    assert status["progress"]["estimated_total_frames"] == 4_074_898
    assert "runner_active" in status


def test_load_status_handles_missing_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    status = load_status(output_dir, assets_dir)

    assert status["budget"]["total_calls"] == 0
    assert status["progress"]["completed_scenes"] == 0
    assert status["runner_active"] is False


def test_load_recent_scenes_returns_last_n(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()

    records = [_sample_metadata_record(i, n_frames=5) for i in range(20)]
    _write_metadata(output_dir / "metadata.jsonl", records)

    scenes = load_recent_scenes(output_dir, n=5)

    assert len(scenes) == 5
    # Newest first (last lines of file)
    assert scenes[0]["scene_id"] == 19
    assert scenes[4]["scene_id"] == 15
    # Each has sample_frames and sample_captions
    assert len(scenes[0]["sample_frames"]) <= 5
    assert len(scenes[0]["sample_captions"]) == 2


def test_load_recent_scenes_handles_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()

    scenes = load_recent_scenes(output_dir, n=10)
    assert scenes == []


def test_load_scene_frames(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()

    records = [_sample_metadata_record(42, n_frames=10)]
    _write_metadata(output_dir / "metadata.jsonl", records)

    frames = load_scene_frames(output_dir, scene_id=42)

    assert len(frames) == 10
    assert frames[0]["image_name"] == "s42_f0000.jpg"
    assert frames[0]["caption"] == "Caption for s42_f0000.jpg"


def test_load_scene_frames_unknown_scene(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()

    records = [_sample_metadata_record(1)]
    _write_metadata(output_dir / "metadata.jsonl", records)

    frames = load_scene_frames(output_dir, scene_id=999)
    assert frames == []


def test_runner_active_detects_recent_update(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    _write_budget(output_dir / "budget_state.json")
    _write_progress(output_dir / "caption_progress.json")
    _write_info(assets_dir / "frame_search_openclip-ViT-H-14_info.json")

    status = load_status(output_dir, assets_dir)
    # File was just written, should be "active"
    assert status["runner_active"] is True


def test_load_recent_scenes_includes_cost(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()
    images_dir = output_dir / "images"
    images_dir.mkdir()

    record = _sample_metadata_record(10, n_frames=3)
    _write_metadata(output_dir / "metadata.jsonl", [record])

    # Write caption .txt files so cost can be inferred from image count
    scenes = load_recent_scenes(output_dir, n=5)
    assert scenes[0]["frame_count"] == 3
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/.stash/plugins/stash-copilot && uv run pytest tests/tools/test_caption_dashboard.py -v`
Expected: `ModuleNotFoundError: No module named 'tools.dataset.caption_dashboard'`

**Step 3: Write the implementation**

Create `tools/dataset/caption_dashboard.py`:

```python
#!/usr/bin/env python3
"""Caption Pipeline Dashboard — live monitoring of caption generation.

Serves a web dashboard that polls the caption runner's output files
(budget_state.json, caption_progress.json, metadata.jsonl) and displays
real-time progress, cost, rate limits, and recently captioned frames.

Usage:
    uv run python tools/dataset/caption_dashboard.py
    uv run python tools/dataset/caption_dashboard.py --port 8766
    uv run python tools/dataset/caption_dashboard.py --output-dir /path/to/lora_dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlparse

# ── Defaults ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "lora_dataset"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets"
HTML_FILE = Path(__file__).resolve().parent / "caption_dashboard.html"
MODEL_KEY = "openclip-ViT-H-14"

# How recently budget_state.json must be modified to consider runner "active"
ACTIVE_THRESHOLD_SECONDS = 60


# ── Data loading ─────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning empty dict if missing or corrupt."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_status(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
) -> dict[str, Any]:
    """Load combined status from budget, progress, and info files.

    This is called on every /api/status poll (every 2s). The files are
    tiny (<10KB each), so re-reading them is negligible overhead.
    """
    budget_path = output_dir / "budget_state.json"
    progress_path = output_dir / "caption_progress.json"
    info_path = assets_dir / f"frame_search_{MODEL_KEY}_info.json"

    budget = _read_json(budget_path)
    progress = _read_json(progress_path)
    info = _read_json(info_path)

    # Determine runner liveness from file mtime
    runner_active = False
    if budget_path.exists():
        mtime = budget_path.stat().st_mtime
        runner_active = (time.time() - mtime) < ACTIVE_THRESHOLD_SECONDS

    # Normalize progress
    completed_list = progress.get("completed_scenes", [])

    return {
        "budget": {
            "total_calls": budget.get("total_calls", 0),
            "total_input_tokens": budget.get("total_input_tokens", 0),
            "total_output_tokens": budget.get("total_output_tokens", 0),
            "total_cost": budget.get("total_cost", 0.0),
            "total_errors": budget.get("total_errors", 0),
            "max_cost": budget.get("max_cost"),
            "rpd_count": budget.get("rpd_count", 0),
            "rpd_date": budget.get("rpd_date", ""),
        },
        "progress": {
            "completed_scenes": len(completed_list),
            "total_scenes": info.get("scene_count", 0),
            "total_frames_captioned": progress.get("total_frames_captioned", 0),
            "estimated_total_frames": info.get("frame_count", 0),
            "errors": progress.get("errors", 0),
            "last_updated": progress.get("last_updated", ""),
        },
        "rate_limits": {
            "rpm_limit": 900,
            "tpm_limit": 900_000,
            "rpd_limit": 9_500,
        },
        "pricing": {
            "model": "gemini-3.0-flash-preview",
            "input_per_m": 0.50,
            "output_per_m": 3.00,
        },
        "runner_active": runner_active,
    }


def load_recent_scenes(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n: int = 50,
) -> list[dict[str, Any]]:
    """Load the last N scenes from metadata.jsonl.

    Reads the file in reverse to get the most recent entries efficiently.
    Returns newest-first with sample frames and captions.
    """
    jsonl_path = output_dir / "metadata.jsonl"
    if not jsonl_path.exists():
        return []

    # Read all lines (metadata.jsonl grows to ~12K lines max — small enough)
    try:
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []

    if not lines:
        return []

    # Take last N, reverse for newest-first
    recent_lines = lines[-n:]
    recent_lines.reverse()

    scenes: list[dict[str, Any]] = []
    for line in recent_lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        image_names = record.get("image_names", [])
        captions = record.get("captions", {})
        frame_count = len(image_names)

        # Sample frames: up to 5 evenly spaced
        if frame_count <= 5:
            sample_frames = image_names
        else:
            step = frame_count / 5
            sample_frames = [image_names[int(i * step)] for i in range(5)]

        # Sample captions: first 2
        sample_captions = []
        for name in image_names[:2]:
            cap = captions.get(name, "")
            if cap and not cap.startswith("[ERROR"):
                sample_captions.append(cap)

        scenes.append({
            "scene_id": record.get("scene_id"),
            "frame_count": frame_count,
            "selection": record.get("selection", {}),
            "captioned_at": record.get("captioned_at", ""),
            "sample_frames": sample_frames,
            "sample_captions": sample_captions,
        })

    return scenes


def load_scene_frames(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    scene_id: int = 0,
) -> list[dict[str, str]]:
    """Load all frames and captions for a specific scene.

    Returns list of {"image_name": "...", "caption": "..."} dicts.
    """
    jsonl_path = output_dir / "metadata.jsonl"
    if not jsonl_path.exists():
        return []

    try:
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return []

    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if record.get("scene_id") == scene_id:
            image_names = record.get("image_names", [])
            captions = record.get("captions", {})
            return [
                {"image_name": name, "caption": captions.get(name, "")}
                for name in image_names
            ]

    return []


# ── HTTP Server ──────────────────────────────────────────────────────────


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    # Set by main() before server starts
    output_dir: Path = DEFAULT_OUTPUT_DIR
    assets_dir: Path = DEFAULT_ASSETS_DIR

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", ""):
            self._serve_file(HTML_FILE, "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/status":
            self._send_json(load_status(self.output_dir, self.assets_dir))
        elif path == "/api/scenes":
            n = int(query.get("n", ["50"])[0])
            n = max(1, min(n, 500))
            self._send_json({"scenes": load_recent_scenes(self.output_dir, n=n)})
        elif path.startswith("/api/scene/") and path.endswith("/frames"):
            # /api/scene/12345/frames
            scene_id_str = path.split("/")[3]
            try:
                scene_id = int(scene_id_str)
            except ValueError:
                self.send_error(400, "Invalid scene ID")
                return
            self._send_json({"frames": load_scene_frames(self.output_dir, scene_id)})
        elif path.startswith("/assets/lora_dataset/images/"):
            filename = path.split("/")[-1]
            full = self.output_dir / "images" / filename
            if full.is_file() and full.suffix in (".jpg", ".jpeg", ".png"):
                mime = "image/png" if full.suffix == ".png" else "image/jpeg"
                self._serve_file(full, mime)
            elif full.is_file() and full.suffix == ".txt":
                self._serve_file(full, "text/plain; charset=utf-8")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default request logging — too noisy with 2s polling
        pass


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live dashboard for caption pipeline monitoring.",
    )
    parser.add_argument("--port", type=int, default=8766,
                        help="HTTP server port (default: 8766)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Caption runner output directory")
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR,
                        help="Assets directory with embedding info")
    args = parser.parse_args()

    DashboardHandler.output_dir = args.output_dir
    DashboardHandler.assets_dir = args.assets_dir

    server = ThreadedServer(("", args.port), DashboardHandler)
    print(f"Caption Dashboard: http://localhost:{args.port}")
    print(f"Output dir: {args.output_dir}")
    print(f"Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/.stash/plugins/stash-copilot && uv run pytest tests/tools/test_caption_dashboard.py -v`
Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
git add tools/dataset/caption_dashboard.py tests/tools/test_caption_dashboard.py
git commit -m "feat(dataset): add caption_dashboard server with status/scenes API"
```

---

### Task 2: Dashboard UI — `caption_dashboard.html`

**Files:**
- Create: `tools/dataset/caption_dashboard.html`

Single-file vanilla JS SPA matching the Stash Copilot AI Insights design language. No tests (tested via browser). All CSS and JS inline.

**Reference files for design system:**
- `stash-copilot.css` — color palette, stat cards, card themes, animations
- `stash-copilot.js:3720` — `createInsightsModal()` for component structure
- `docs/plans/2026-02-19-caption-dashboard-design.md` — layout spec

**Step 1: Create the HTML file**

Create `tools/dataset/caption_dashboard.html`:

The file must implement these sections (full code to be written during implementation):

**HTML structure:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Caption Pipeline Dashboard</title>
    <style>/* All CSS inline */</style>
</head>
<body>
    <header class="dashboard-header">
        <!-- AI orb + title + runner status + model badge -->
    </header>

    <section class="metrics-row">
        <!-- 6 stat items: cost, scenes, frames, rpm, tpm, rpd -->
    </section>

    <section class="progress-section">
        <!-- Full-width progress bar + ETA + error count -->
    </section>

    <section class="scenes-section">
        <!-- Window slider + scene cards -->
    </section>

    <script>/* All JS inline */</script>
</body>
</html>
```

**CSS requirements (from design doc + AI Insights modal reference):**

```css
/* Background */
body {
    margin: 0;
    background: linear-gradient(180deg, #1e242d 0%, #1a2030 50%, #181c23 100%);
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #d1d5db;
}

/* Stat items — same pattern as .stash-copilot-stat-item */
.stat-item {
    padding: 0.875rem 0.75rem;
    background: linear-gradient(135deg, rgba(96,165,250,0.08) 0%, rgba(139,92,246,0.05) 100%);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.04);
    text-align: center;
}
.stat-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #60a5fa;
}
.stat-label {
    font-size: 0.6rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Scene cards — same pattern as .stash-copilot-card */
.scene-card {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
    animation: cardFadeIn 0.5s cubic-bezier(0.34,1.2,0.64,1) both;
}
.scene-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(96,165,250,0.15), 0 0 20px rgba(139,92,246,0.1);
}

/* Progress bar — primary button gradient */
.progress-fill {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    border-radius: 4px;
    height: 100%;
    transition: width 0.5s ease;
}

/* AI orb — same keyframes as .stash-copilot-insights-title::before */
@keyframes aiPulse {
    0%, 100% { box-shadow: 0 0 12px rgba(96,165,250,0.6); background-position: 0% 50%; }
    33% { box-shadow: 0 0 16px rgba(139,92,246,0.6); background-position: 50% 50%; }
    66% { box-shadow: 0 0 14px rgba(6,182,212,0.6); background-position: 100% 50%; }
}

/* Card fade-in — same as unifiedCardFadeIn */
@keyframes cardFadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Scrollbar — same as modal */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb {
    background: rgba(96,165,250,0.3);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(96,165,250,0.5); }
```

**JavaScript requirements:**

```javascript
// State
let status = null;       // from /api/status
let scenes = [];         // from /api/scenes
let windowSize = 50;     // configurable via slider
let expandedScene = null; // scene ID of expanded card

// Polling
async function pollStatus() {
    const resp = await fetch('/api/status');
    status = await resp.json();
    renderMetrics();
    renderProgress();
    renderHeader();
}

async function pollScenes() {
    const resp = await fetch(`/api/scenes?n=${windowSize}`);
    const data = await resp.json();
    scenes = data.scenes;
    renderScenes();
}

// Start polling
setInterval(pollStatus, 2000);
setInterval(pollScenes, 10000);
pollStatus();
pollScenes();

// Rendering functions
function renderHeader() { /* Update runner status dot, model badge */ }
function renderMetrics() { /* Update 6 stat items + progress bars */ }
function renderProgress() { /* Update main progress bar, ETA, errors */ }
function renderScenes() { /* Render scene cards, newest first */ }

// Scene card expand/collapse
async function expandScene(sceneId) {
    const resp = await fetch(`/api/scene/${sceneId}/frames`);
    const data = await resp.json();
    // Show all frames + captions in expanded card
}

// Utility: format numbers, relative time, currency
function formatNumber(n) { return n.toLocaleString(); }
function formatCost(n) { return '$' + n.toFixed(2); }
function timeAgo(isoString) { /* "2m ago", "1h ago" */ }
```

**Key rendering details:**

- **Stat items** update in-place (no DOM recreation) to avoid flicker on 2s poll
- **Scene cards** are diffed by scene_id — only new cards get the fade-in animation
- **Expanded card** fetches all frames on-demand from `/api/scene/<id>/frames`
- **Thumbnails** are `<img>` tags pointing to `/assets/lora_dataset/images/<name>`
- **Window slider** triggers a re-poll of `/api/scenes?n=<new_value>` immediately
- **Runner status dot** uses green (`#10b981`) when active, gray (`#6b7280`) when idle
- **Numbers animate** from old to new value using `requestAnimationFrame` counter

**Step 2: Verify in browser**

```bash
# Start the dashboard (no runner needed — works with static test files)
cd ~/.stash/plugins/stash-copilot && \
uv run python tools/dataset/caption_dashboard.py --port 8766
```

Open http://localhost:8766. Verify:
- Page loads with dark gradient background
- Metrics show zeros (no runner data yet)
- "Runner: idle" appears in header
- No JS console errors
- Scrollbar uses custom blue style

**Step 3: Create test fixture data**

```bash
# Create fake runner output for visual testing
mkdir -p assets/lora_dataset/images
echo '{"total_calls":1234,"total_input_tokens":1851000,"total_output_tokens":98640,"total_cost":1.30,"total_errors":3,"rpd_count":1234,"rpd_date":"2026-02-19","saved_at":"2026-02-19T14:00:00Z"}' > assets/lora_dataset/budget_state.json
echo '{"completed_scenes":[1,2,3,4,5],"total_frames_captioned":890,"errors":3,"last_updated":"2026-02-19T14:00:00Z"}' > assets/lora_dataset/caption_progress.json
```

Reload the dashboard. Verify:
- Cost shows $1.30
- Scenes shows 5
- Frames shows 890
- RPD shows 1,234

**Step 4: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "feat(dataset): add caption dashboard UI with live metrics and scene cards"
```

---

### Task 3: Integration — Live Test with Caption Runner

**Step 1: Start dashboard + runner simultaneously**

```bash
# Terminal 1: Start dashboard
cd ~/.stash/plugins/stash-copilot && \
uv run python tools/dataset/caption_dashboard.py

# Terminal 2: Start runner (1 scene, budget cap)
cd ~/.stash/plugins/stash-copilot && \
uv run python tools/dataset/caption_runner.py --limit 1 --max-frames 20 --max-cost 1.00
```

**Step 2: Verify live updates in browser**

- Runner status changes from idle (gray) to active (green)
- Cost, frames, RPM meters update every 2s
- Scene card appears when scene completes
- Thumbnails load in the scene card filmstrip
- Captions appear below thumbnails
- Runner status returns to idle after completion

**Step 3: Verify expanded scene card**

- Click a scene card
- All frames load with their individual captions
- Frame images are clickable/zoomable

**Step 4: Commit any fixes**

```bash
git add -A && git commit -m "fix(dataset): dashboard integration fixes from live test"
```
