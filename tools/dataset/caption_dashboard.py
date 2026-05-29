#!/usr/bin/env python3
"""Caption Pipeline Dashboard — live monitoring and control of caption generation.

Serves a web dashboard that polls the caption runner's output files
(budget_state.json, caption_progress.json, metadata.jsonl) and displays
real-time progress, cost, rate limits, and recently captioned frames.

Also provides launch/stop controls to manage the runner subprocess directly.

Usage:
    uv run python tools/dataset/caption_dashboard.py
    uv run python tools/dataset/caption_dashboard.py --port 8766
    uv run python tools/dataset/caption_dashboard.py --output-dir /path/to/lora_dataset
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import os
import random
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, TypedDict
from urllib.parse import parse_qs, urlparse

from tools.dataset.batch_api import (
    BatchJob,
    load_batch_state,
    save_batch_state,
    prepare_batch_chunks,
    upload_jsonl,
    submit_batch_job,
    poll_batch_jobs,
    collect_batch_results,
    check_scene_completion,
    update_caption_progress,
)

# ── .env loader ──────────────────────────────────────────────────────────

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


def _load_dotenv(env_path: Path = _ENV_FILE) -> dict[str, str]:
    """Load KEY=VALUE pairs from .env into os.environ. Returns loaded keys."""
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


# ── Defaults ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "lora_dataset"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets"
HTML_FILE = Path(__file__).resolve().parent / "caption_dashboard.html"
MODEL_KEY = "openclip-ViT-H-14"

# How recently budget_state.json must be modified to consider runner "active"
ACTIVE_THRESHOLD_SECONDS = 60

# Cache TTLs (seconds) — prevents re-scanning 600K+ files every poll cycle
_CACHE_TTL_ERRORS = 10.0
_CACHE_TTL_SCENES = 10.0
_CACHE_TTL_SHOWCASE = 30.0


class _CachedResult:
    """Simple TTL cache for expensive filesystem-scanning functions."""

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._data: Any = None
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

    def get(self) -> Any | None:
        with self._lock:
            if time.time() < self._expires_at:
                return self._data
        return None

    def set(self, data: Any) -> None:
        with self._lock:
            self._data = data
            self._expires_at = time.time() + self._ttl

    def invalidate(self) -> None:
        with self._lock:
            self._expires_at = 0.0


_errors_cache = _CachedResult(_CACHE_TTL_ERRORS)
_scenes_cache = _CachedResult(_CACHE_TTL_SCENES)
_completed_ids_cache = _CachedResult(30.0)  # Completed IDs change infrequently
_showcase_cache = _CachedResult(_CACHE_TTL_SHOWCASE)


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
    """Load combined status from budget, progress, and info files."""
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
            "model": budget.get("model", "gemini-3-flash-preview"),
            "input_per_m": 0.50,
            "output_per_m": 3.00,
        },
        "runner_active": runner_active,
    }


def _tail_lines(path: Path, n: int) -> list[str]:
    """Read the last N lines of a file efficiently without loading it all."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []

    # Read a chunk from the end — each metadata line is ~10-20KB, so
    # budget ~25KB per line to be safe
    chunk_size = min(size, n * 25_000)
    try:
        with open(path, "rb") as f:
            f.seek(max(0, size - chunk_size))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []

    lines = data.strip().splitlines()
    # If we didn't read from the start, the first line may be partial
    if chunk_size < size and lines:
        lines = lines[1:]
    return lines[-n:]


def load_recent_scenes(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n: int = 50,
) -> list[dict[str, Any]]:
    """Load the last N scenes from metadata.jsonl."""
    jsonl_path = output_dir / "metadata.jsonl"
    if not jsonl_path.exists():
        return []

    recent_lines = _tail_lines(jsonl_path, n)
    if not recent_lines:
        return []

    # Reverse for newest-first
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

        # Sample frames: up to 16 evenly spaced, with captions embedded
        if frame_count <= 16:
            sample_names = image_names
        else:
            step = frame_count / 16
            sample_names = [image_names[int(i * step)] for i in range(16)]

        sample_frames = [
            {"frame": name, "caption": captions.get(name, "")}
            for name in sample_names
        ]

        # Count errors for progress display
        error_count = sum(
            1 for cap in captions.values()
            if isinstance(cap, str) and cap.startswith("[ERROR")
        )

        scenes.append({
            "scene_id": record.get("scene_id"),
            "frame_count": frame_count,
            "error_count": error_count,
            "selection": record.get("selection", {}),
            "captioned_at": record.get("captioned_at", ""),
            "sample_frames": sample_frames,
        })

    return scenes


def _load_completed_scene_ids(output_dir: Path) -> set[int]:
    """Extract completed scene IDs from metadata.jsonl.

    Cached for 30s — completed IDs only grow, never shrink, so staleness
    just means we might briefly show a scene as in-progress after completion.
    """
    cached = _completed_ids_cache.get()
    if cached is not None:
        return cached

    jsonl_path = output_dir / "metadata.jsonl"
    if not jsonl_path.exists():
        _completed_ids_cache.set(set())
        return set()
    ids: set[int] = set()
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                # Fast extraction: grab "scene_id": NNN without full JSON parse
                # Format: {"scene_id": 1234, ...}
                idx = line.find('"scene_id"')
                if idx == -1:
                    continue
                # Extract the number after "scene_id":
                rest = line[idx + 12:idx + 25]  # enough chars for the number
                num_str = ""
                for ch in rest:
                    if ch.isdigit():
                        num_str += ch
                    elif num_str:
                        break
                if num_str:
                    ids.add(int(num_str))
    except OSError:
        pass
    _completed_ids_cache.set(ids)
    return ids


def load_in_progress_scenes(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    completed_scene_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Detect scenes being captioned right now.

    Uses a lightweight approach: checks runner log for the most recent scene
    being processed, then only scans that scene's files — NOT the entire
    images directory (which can have 300K+ files).
    """
    # When runner is stopped, clear in-progress state immediately so
    # "retrying…" tags don't persist from stale log buffer entries.
    if not runner_manager.is_running:
        _scenes_cache.set([])
        return []

    cached = _scenes_cache.get()
    if cached is not None:
        return cached

    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        _scenes_cache.set([])
        return []

    # Build completed IDs set
    if completed_scene_ids is None:
        completed_scene_ids = _load_completed_scene_ids(output_dir)

    # Instead of globbing ALL 300K+ jpgs, find in-progress scenes from
    # the runner log — the log tells us which scene is being worked on.
    # Fall back to scanning only the LAST few scenes by directory listing.
    active_sids: set[int] = set()
    fill_sids: set[int] = set()  # Fill-pass scenes (orphan frames, NOT retry)
    retry_sids_ordered: list[int] = []  # Only --retry-errors scenes

    # Strategy: get recent log lines and extract scene IDs being processed
    log_lines = runner_manager.get_log(50)
    for line in log_lines:
        stripped = line.strip()
        # Log format: "  [44/327201] FIXED s1003_f0027.jpg" or
        #             "  [44/327201] FAILED s1003_f0027.jpg: error" or
        #             "      FILLED s23_f0042.jpg" (fill pass) or
        #             "      FILL ERROR s23_f0042.jpg: ..." (fill pass)
        if "_f" in line and stripped.startswith("[") and ("FIXED" in line or "FAILED" in line):
            # Retry-mode lines (--retry-errors) — show as "retrying…"
            for part in line.split():
                if part.startswith("s") and "_f" in part:
                    prefix = part.split("_f")[0]
                    try:
                        sid = int(prefix[1:])
                        if sid not in retry_sids_ordered:
                            retry_sids_ordered.append(sid)
                    except (ValueError, IndexError):
                        pass
        elif "_f" in line and ("FILLED" in line or "FILL ERROR" in line):
            # Fill-pass lines — orphan frame repair during normal captioning.
            # These are NOT retries; show as "captioning…" not "retrying…".
            for part in line.split():
                if part.startswith("s") and "_f" in part:
                    prefix = part.split("_f")[0]
                    try:
                        sid = int(prefix[1:])
                        fill_sids.add(sid)
                    except (ValueError, IndexError):
                        pass
        elif "_f" in line and stripped.startswith("["):
            # Normal captioning in-progress lines
            for part in line.split():
                if part.startswith("s") and "_f" in part:
                    prefix = part.split("_f")[0]
                    try:
                        sid = int(prefix[1:])
                        if sid not in completed_scene_ids:
                            active_sids.add(sid)
                    except (ValueError, IndexError):
                        pass
        elif "Scene " in line and ":" in line and "selected" in line:
            try:
                sid = int(line.split("Scene")[1].split()[0])
                if sid not in completed_scene_ids:
                    active_sids.add(sid)
            except (ValueError, IndexError):
                pass

    # Combine: active (new) + fill (orphan repair) + retry (error recovery)
    # Only retry_sids get "retrying…" tag; fill_sids get "captioning…"
    retry_sids = set(retry_sids_ordered[-5:])
    all_sids = active_sids | fill_sids | retry_sids
    if not all_sids:
        _scenes_cache.set([])
        return []

    # Only scan files for the identified active scenes (typically 1-5 scenes)
    scenes: list[dict[str, Any]] = []
    for sid in sorted(all_sids, reverse=True):
        pattern = f"s{sid}_f*.jpg"
        frames = sorted(p.name for p in images_dir.glob(pattern))
        if not frames:
            continue

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

    _scenes_cache.set(scenes)
    return scenes


def load_scene_frames(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    scene_id: int = 0,
) -> list[dict[str, str]]:
    """Load all frames and captions for a specific scene."""
    jsonl_path = output_dir / "metadata.jsonl"
    if not jsonl_path.exists():
        return []

    # Stream line-by-line instead of loading entire 50MB+ file
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
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
    except OSError:
        return []

    return []


def load_showcase(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n: int = 20,
    max_scenes: int = 50,
) -> dict[str, Any]:
    """Sample N random captioned frames from the last *max_scenes* scenes.

    Reads only the tail of metadata.jsonl (last 50 scenes by default),
    collects valid (image, caption, scene_id) tuples, then randomly samples N.
    Results are cached for 30s.
    """
    cached = _showcase_cache.get()
    if cached is not None:
        pool = cached
        sample = random.sample(pool, min(n, len(pool)))
        return {"frames": sample}

    jsonl_path = output_dir / "metadata.jsonl"
    if not jsonl_path.exists():
        return {"frames": []}

    recent_lines = _tail_lines(jsonl_path, max_scenes)
    if not recent_lines:
        return {"frames": []}

    pool: list[dict[str, Any]] = []
    for raw_line in recent_lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        scene_id = record.get("scene_id")
        captions = record.get("captions", {})
        for img_name, caption in captions.items():
            if not caption or caption.startswith("[ERROR"):
                continue
            pool.append({
                "image": img_name,
                "caption": caption,
                "scene_id": scene_id,
            })

    if not pool:
        return {"frames": []}

    _showcase_cache.set(pool)
    sample = random.sample(pool, min(n, len(pool)))
    return {"frames": sample}




def load_error_count(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    """Fast error count — uses cached full scan if available, else progress file.

    Priority:
    1. If errors cache is warm (from /api/errors call), use its length — exact.
    2. Otherwise, return the count from caption_progress.json — slightly stale
       but instant (no filesystem scan needed).
    """
    # If the full errors list is cached, its length is authoritative
    cached_errors = _errors_cache.get()
    if cached_errors is not None:
        return len(cached_errors)

    # Fall back to progress file count (written by runner, always available)
    progress = _read_json(output_dir / "caption_progress.json")
    return progress.get("errors", 0)


# ── Frame event extraction ────────────────────────────────────────────


class FrameEvent(TypedDict):
    frame: str
    status: str  # "OK" or "ERROR"
    caption: str
    scene_id: str


# Set-based dedup: tracks which frame events (by "scene:frame:status" key)
# are currently in the log buffer.  New events = keys in buffer but not in set.
# This avoids the sliding-window cursor bug where a positional index breaks
# once get_log() returns a full-length result and old lines rotate out.
_seen_frame_keys: set[str] = set()
_frame_event_lock: threading.Lock = threading.Lock()


def _extract_scene_id(frame: str) -> str:
    """Extract scene ID from frame name (s1042_f0029.jpg -> '1042').

    Returns empty string if frame name doesn't match expected format.
    """
    if not frame.startswith("s") or "_f" not in frame:
        return ""
    try:
        sid = frame[1:].split("_f")[0]
        int(sid)  # Validate numeric
        return sid
    except (ValueError, IndexError):
        return ""


def _extract_new_frame_events(max_events: int = 20) -> list[FrameEvent]:
    """Parse FRAME_DONE lines from the runner log, returning only unseen events.

    Uses set-based deduplication instead of a positional cursor.  The old
    cursor broke once ``get_log(200)`` returned a full-length result — new
    lines rotated in at the same indices as old ones, so the cursor never
    advanced and ALL subsequent events were silently dropped.
    """
    global _seen_frame_keys

    if not runner_manager.is_running:
        with _frame_event_lock:
            _seen_frame_keys = set()
        return []

    log_lines = runner_manager.get_log(200)
    new_events: list[FrameEvent] = []
    current_keys: set[str] = set()

    with _frame_event_lock:
        for line in log_lines:
            if not line.startswith("FRAME_DONE "):
                continue
            # Format: FRAME_DONE s1042_f0029.jpg OK A woman in a red dress...
            parts = line.split(" ", 3)
            if len(parts) < 3:
                continue
            frame = parts[1]
            status = parts[2]
            if status not in ("OK", "ERROR"):
                continue
            caption = parts[3] if len(parts) > 3 else ""
            scene_id = _extract_scene_id(frame)

            # Dedup key includes status so retries (ERROR→OK) are re-sent
            key = f"{scene_id}:{frame}:{status}"
            current_keys.add(key)
            if key in _seen_frame_keys:
                continue

            new_events.append(FrameEvent(
                frame=frame,
                status=status,
                caption=caption,
                scene_id=scene_id,
            ))

        # Replace seen set with current buffer contents — keys that
        # rotated out of the deque are automatically forgotten.
        _seen_frame_keys = current_keys

    return new_events[-max_events:]


_errors_scan_lock = threading.Lock()
_errors_scan_running = False


def _run_errors_scan(images_dir: Path) -> None:
    """Background thread: scan all .txt files for errors."""
    global _errors_scan_running
    try:
        errors: list[dict[str, str]] = []
        for txt in sorted(images_dir.glob("*.txt")):
            try:
                with open(txt, "rb") as f:
                    head = f.read(7)
                if not head.startswith(b"[ERROR"):
                    continue
                content = txt.read_text(encoding="utf-8")
            except OSError:
                continue
            errors.append({
                "image_name": txt.stem + ".jpg",
                "error": content,
            })
        _errors_cache.set(errors)
    finally:
        with _errors_scan_lock:
            _errors_scan_running = False


def load_errors(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    """Return paginated error list: {total, errors: [...], scanning: bool}.

    On first call, kicks off a background scan and returns immediately
    with an empty list + scanning=true.  Subsequent calls during the scan
    return scanning=true.  Once the scan completes and the cache is warm,
    returns results from cache instantly.
    """
    global _errors_scan_running

    cached = _errors_cache.get()
    if cached is not None:
        total = len(cached)
        page = cached[offset:offset + limit]
        return {"total": total, "errors": page, "scanning": False}

    # Cache miss — kick off background scan if not already running
    images_dir = output_dir / "images"
    scanning = False
    with _errors_scan_lock:
        if not _errors_scan_running and images_dir.is_dir():
            _errors_scan_running = True
            scanning = True
            t = threading.Thread(target=_run_errors_scan, args=(images_dir,), daemon=True)
            t.start()
        elif _errors_scan_running:
            scanning = True

    # Return immediately — progress file gives approximate count
    progress = _read_json(output_dir / "caption_progress.json")
    approx_count = progress.get("errors", 0)
    return {"total": approx_count, "errors": [], "scanning": scanning}


# ── Runner process management ─────────────────────────────────────────────


RUNNER_MODULE = "tools.dataset.caption_runner"
LOG_BUFFER_SIZE = 500  # lines of stderr to keep


def _find_orphan_runner() -> int | None:
    """Find an already-running caption_runner process not managed by us."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", RUNNER_MODULE],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                pid = int(line.strip())
                # Verify it's actually alive
                try:
                    os.kill(pid, 0)
                    return pid
                except OSError:
                    continue
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


class RunnerManager:
    """Manages the caption runner subprocess lifecycle."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._orphan_pid: int | None = None
        self._log: collections.deque[str] = collections.deque(maxlen=LOG_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        # Check for orphaned runner on startup
        self._adopt_orphan()

    def _adopt_orphan(self) -> None:
        """Detect a runner process from a previous dashboard session."""
        orphan = _find_orphan_runner()
        if orphan is not None:
            with self._lock:
                self._orphan_pid = orphan
                self._log.append(
                    f"[dashboard] Adopted orphan runner PID {orphan}"
                )
                log_path = DEFAULT_OUTPUT_DIR / "runner.log"
                if log_path.exists():
                    self._log.append(
                        f"[dashboard] Tailing runner.log for output"
                    )
                else:
                    self._log.append(
                        f"[dashboard] No runner.log found — logs from this run are unavailable"
                    )
                    self._log.append(
                        f"[dashboard] Metrics are still updating live. Stop and re-launch to get full logs."
                    )

    def _orphan_alive(self) -> bool:
        """Check if the adopted orphan process is still running."""
        if self._orphan_pid is None:
            return False
        try:
            os.kill(self._orphan_pid, 0)
            return True
        except OSError:
            self._orphan_pid = None
            return False

    @property
    def is_running(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            if self._orphan_alive():
                return True
            # Re-check for a runner started after the dashboard
            orphan = _find_orphan_runner()
            if orphan is not None:
                self._orphan_pid = orphan
                self._log.append(f"[dashboard] Detected runner PID {orphan}")
                return True
            return False

    @property
    def pid(self) -> int | None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._proc.pid
            if self._orphan_alive():
                return self._orphan_pid
            return None

    @property
    def exit_code(self) -> int | None:
        with self._lock:
            if self._proc is not None:
                return self._proc.poll()
            return None

    def get_log(self, last_n: int = 200) -> list[str]:
        with self._lock:
            lines = list(self._log)
        # If we have an adopted orphan and no log output from pipe,
        # tail the persistent runner.log file instead.
        if len(lines) <= 1 and self._orphan_pid is not None:
            log_path = DEFAULT_OUTPUT_DIR / "runner.log"
            file_lines = _tail_lines(log_path, last_n)
            if file_lines:
                return file_lines
        return lines[-last_n:]

    def launch(
        self,
        *,
        api_key: str,
        openrouter_key: str = "",
        limit: int | None = None,
        max_cost: float | None = None,
        workers: int = 10,
        max_frames: int = 20,
        model: str = "gemini-3-flash-preview",
        temperature: float = 1.0,
        retry_errors: bool = False,
        skip_gemini_fallback: bool = False,
    ) -> dict[str, Any]:
        """Launch the caption runner as a subprocess.

        Returns dict with status info. Raises RuntimeError if already running.
        """
        if self.is_running:
            raise RuntimeError("Runner is already active")

        cmd = [
            sys.executable, "-m", RUNNER_MODULE,
            "--model", model,
            "--temperature", str(temperature),
        ]
        if api_key:
            cmd.extend(["--api-key", api_key])
        if openrouter_key:
            cmd.extend(["--openrouter-key", openrouter_key])
        if skip_gemini_fallback:
            cmd.append("--skip-gemini-fallback")

        # Workers apply to both main run and retry
        cmd.extend(["--workers", str(workers)])
        if retry_errors:
            cmd.append("--retry-errors")
        else:
            cmd.extend(["--max-frames", str(max_frames)])
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        if max_cost is not None:
            cmd.extend(["--max-cost", str(max_cost)])

        with self._lock:
            self._orphan_pid = None  # Clear orphan reference on new launch
            self._log.clear()
            self._log.append(f"[dashboard] Launching: {' '.join(cmd[2:])}")

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            self._reader_thread = threading.Thread(
                target=self._read_output,
                daemon=True,
            )
            self._reader_thread.start()

        return {"pid": self._proc.pid, "status": "launched"}

    def stop(self) -> dict[str, Any]:
        """Send SIGTERM to the runner. Returns status info."""
        with self._lock:
            # Try managed process first
            if self._proc is not None and self._proc.poll() is None:
                pid = self._proc.pid
                self._log.append(f"[dashboard] Sending SIGTERM to PID {pid}")
                self._proc.terminate()
            elif self._orphan_pid is not None and self._orphan_alive():
                # Stop adopted orphan process
                pid = self._orphan_pid
                self._log.append(f"[dashboard] Sending SIGTERM to orphan PID {pid}")
                os.kill(pid, signal.SIGTERM)
                self._orphan_pid = None
                return {"status": "stopped", "pid": pid}
            else:
                return {"status": "not_running"}

        # Wait briefly for graceful shutdown (managed process)
        try:
            self._proc.wait(timeout=5)
            with self._lock:
                self._log.append(f"[dashboard] Process {pid} exited (code {self._proc.returncode})")
        except subprocess.TimeoutExpired:
            with self._lock:
                self._log.append(f"[dashboard] Process {pid} did not exit, sending SIGKILL")
                self._proc.kill()
                self._proc.wait(timeout=3)
                self._log.append(f"[dashboard] Process {pid} killed")

        return {"status": "stopped", "pid": pid}

    def _read_output(self) -> None:
        """Background thread: reads process stdout/stderr into the log buffer."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
                with self._lock:
                    self._log.append(line)
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                if proc.poll() is not None:
                    self._log.append(
                        f"[dashboard] Process exited with code {proc.returncode}"
                    )


# Singleton — shared across all request handler instances
runner_manager = RunnerManager()


# ── Batch Manager ────────────────────────────────────────────────────────


class BatchManager:
    """Manages batch job lifecycle from the dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_file = DEFAULT_OUTPUT_DIR / "batch_state.json"
        self._jobs: list[BatchJob] = load_batch_state(self._state_file)

    @property
    def jobs(self) -> list[BatchJob]:
        with self._lock:
            return list(self._jobs)

    def submit_all(
        self, api_key: str, model: str = "gemini-3-flash-preview",
        max_frames: int = 20, temperature: float = 1.0,
        scene_limit: int = 0,
    ) -> list[BatchJob]:
        """Prepare chunks, upload, and submit batch jobs for pending frames.

        Args:
            scene_limit: Max scenes to include. 0 = all pending scenes.
        """
        from tools.dataset.frame_selector import load_embedding_index

        index = load_embedding_index(DEFAULT_ASSETS_DIR)
        progress = _read_json(DEFAULT_OUTPUT_DIR / "caption_progress.json")
        completed = set(progress.get("completed_scenes", []))

        chunk_paths = prepare_batch_chunks(
            index=index,
            completed_scenes=completed,
            images_dir=DEFAULT_OUTPUT_DIR / "images",
            frames_dir=DEFAULT_ASSETS_DIR / "embedded_frames",
            max_frames=max_frames,
            temperature=temperature,
            scene_limit=scene_limit,
        )

        new_jobs: list[BatchJob] = []
        for i, chunk_path in enumerate(chunk_paths):
            frame_count = sum(1 for _ in open(chunk_path))
            scene_ids: set[int] = set()
            with open(chunk_path) as f:
                for line in f:
                    key = json.loads(line).get("key", "")
                    try:
                        scene_ids.add(int(key.split("_f")[0][1:]))
                    except (ValueError, IndexError):
                        pass

            file_name = upload_jsonl(chunk_path, api_key)
            job = submit_batch_job(
                file_name=file_name,
                model=model,
                api_key=api_key,
                display_name=f"chunk-{i+1:03d}",
                frame_count=frame_count,
                scene_ids=sorted(scene_ids),
            )
            new_jobs.append(job)

        with self._lock:
            self._jobs.extend(new_jobs)
            save_batch_state(self._state_file, self._jobs)

        return new_jobs

    def poll(self, api_key: str) -> list[BatchJob]:
        """Poll all active jobs, auto-collect succeeded ones, update progress."""
        from tools.dataset.frame_selector import load_embedding_index

        images_dir = DEFAULT_OUTPUT_DIR / "images"
        progress_path = DEFAULT_OUTPUT_DIR / "caption_progress.json"
        jobs_collected: list[BatchJob] = []

        with self._lock:
            self._jobs = poll_batch_jobs(self._jobs, api_key)

            for job in self._jobs:
                if job.state == "BATCH_STATE_SUCCEEDED" and not job.collected:
                    try:
                        collect_batch_results(job, images_dir, api_key)
                        jobs_collected.append(job)
                    except Exception:
                        pass

            save_batch_state(self._state_file, self._jobs)

        # Outside the lock: check scene completion and update progress
        # (this calls SmartFrameSelector which can be slow for many scenes)
        if jobs_collected:
            try:
                index = load_embedding_index(DEFAULT_ASSETS_DIR)
                frames_dir = DEFAULT_ASSETS_DIR / "embedded_frames"

                for job in jobs_collected:
                    completed_sids = check_scene_completion(
                        job.scene_ids, images_dir, index, frames_dir,
                    )
                    if job.stats:
                        job.stats.scenes_completed = len(completed_sids)

                    update_caption_progress(
                        progress_path,
                        newly_completed=completed_sids,
                        captions_written=job.stats.captions_written if job.stats else 0,
                        errors=job.stats.errors if job.stats else 0,
                    )

                # Re-save state with updated scenes_completed counts
                with self._lock:
                    save_batch_state(self._state_file, self._jobs)
            except Exception:
                pass  # Progress update is best-effort; batch results are already written

        with self._lock:
            return list(self._jobs)

    def status(self) -> dict[str, Any]:
        """Return serializable status for the dashboard."""
        with self._lock:
            return {
                "jobs": [j.to_dict() for j in self._jobs],
                "total_submitted": sum(j.frame_count for j in self._jobs),
                "total_collected": sum(
                    j.stats.captions_written for j in self._jobs if j.stats
                ),
                "total_errors": sum(
                    j.stats.errors for j in self._jobs if j.stats
                ),
                "has_active": any(
                    j.state in ("BATCH_STATE_PENDING", "BATCH_STATE_RUNNING")
                    for j in self._jobs
                ),
            }


batch_manager = BatchManager()


# ── WebSocket ────────────────────────────────────────────────────────────

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Opcodes
_OP_TEXT = 0x1
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA


class _WebSocketClient:
    """Wraps a raw TCP socket for WebSocket text frame I/O."""

    def __init__(self, sock: socket.socket, address: tuple[str, int]) -> None:
        self._sock = sock
        self.address = address
        self._closed = False
        self._write_lock = threading.Lock()

    # ── Sending ────────────────────────────────

    @staticmethod
    def _build_frame(opcode: int, payload: bytes) -> bytes:
        """Build an unmasked WebSocket frame (server → client)."""
        header = bytes([0x80 | opcode])  # FIN + opcode
        length = len(payload)
        if length < 126:
            header += bytes([length])
        elif length < 65536:
            header += bytes([126]) + struct.pack("!H", length)
        else:
            header += bytes([127]) + struct.pack("!Q", length)
        return header + payload

    def _send_raw(self, data: bytes) -> bool:
        """Send raw bytes with write lock. Returns False on error."""
        if self._closed:
            return False
        with self._write_lock:
            try:
                self._sock.sendall(data)
                return True
            except OSError:
                self._closed = True
                return False

    def send_text(self, data: str) -> bool:
        """Send a text frame. Returns False if the connection is dead."""
        return self._send_raw(self._build_frame(_OP_TEXT, data.encode("utf-8")))

    def send_pong(self, payload: bytes) -> bool:
        return self._send_raw(self._build_frame(_OP_PONG, payload))

    def send_close(self, code: int = 1000) -> bool:
        return self._send_raw(self._build_frame(_OP_CLOSE, struct.pack("!H", code)))

    # ── Receiving ──────────────────────────────

    def _recv_exact(self, n: int) -> bytes | None:
        """Blocking read of exactly *n* bytes. Returns None on disconnect."""
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self._sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def read_frame(self) -> tuple[int, bytes] | None:
        """Read one WebSocket frame. Returns (opcode, payload) or None on disconnect."""
        head = self._recv_exact(2)
        if head is None:
            return None

        opcode = head[0] & 0x0F
        masked = bool(head[1] & 0x80)
        length = head[1] & 0x7F

        if length == 126:
            ext = self._recv_exact(2)
            if ext is None:
                return None
            length = struct.unpack("!H", ext)[0]
        elif length == 127:
            ext = self._recv_exact(8)
            if ext is None:
                return None
            length = struct.unpack("!Q", ext)[0]

        mask_key = b""
        if masked:
            mask_key = self._recv_exact(4)
            if mask_key is None:
                return None

        payload = self._recv_exact(length) if length else b""
        if payload is None:
            return None

        if masked and mask_key:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))

        return (opcode, payload)

    # ── Lifecycle ──────────────────────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    @property
    def closed(self) -> bool:
        return self._closed


# Global client registry
_ws_clients: set[_WebSocketClient] = set()
_ws_clients_lock = threading.Lock()


def _ws_broadcast(messages: list[str]) -> None:
    """Send messages to all connected WS clients, pruning dead ones."""
    if not messages:
        return
    with _ws_clients_lock:
        clients = list(_ws_clients)
    dead: list[_WebSocketClient] = []
    for client in clients:
        for msg in messages:
            if not client.send_text(msg):
                dead.append(client)
                break
    if dead:
        with _ws_clients_lock:
            for d in dead:
                _ws_clients.discard(d)
                d.close()


def _send_init_event(
    client: _WebSocketClient,
    output_dir: Path,
    assets_dir: Path,
) -> None:
    """Send a full state snapshot to a newly connected client."""
    status = load_status(output_dir, assets_dir)
    ec = load_error_count(output_dir)
    completed = load_recent_scenes(output_dir, n=50)
    in_progress = load_in_progress_scenes(output_dir)
    frame_events = _extract_new_frame_events(max_events=20)
    runner = {
        "running": runner_manager.is_running,
        "pid": runner_manager.pid,
        "exit_code": runner_manager.exit_code,
    }
    log_lines = runner_manager.get_log(200)

    payload = json.dumps({
        "type": "init",
        "data": {
            "status": status,
            "error_count": ec,
            "active_scenes": {"scenes": in_progress},
            "completed_scenes": {"scenes": completed},
            "frame_events": {"events": frame_events},
            "runner": runner,
            "log": {"lines": log_lines},
        },
    })
    client.send_text(payload)


def _handle_ws_message(
    client: _WebSocketClient,
    payload: str,
    output_dir: Path,
) -> None:
    """Handle a text message received from the client over WebSocket."""
    try:
        msg = json.loads(payload)
    except json.JSONDecodeError:
        return

    msg_type = msg.get("type", "")

    if msg_type == "get_errors":
        limit = int(msg.get("limit", 200))
        offset = int(msg.get("offset", 0))
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        data = load_errors(output_dir, limit=limit, offset=offset)
        client.send_text(json.dumps({"type": "errors", "data": data}))

    elif msg_type == "get_scene_frames":
        scene_id = int(msg.get("scene_id", 0))
        frames = load_scene_frames(output_dir, scene_id)
        client.send_text(json.dumps({
            "type": "scene_frames",
            "data": {"scene_id": scene_id, "frames": frames},
        }))


def _ws_read_loop(
    client: _WebSocketClient,
    output_dir: Path,
) -> None:
    """Blocking read loop — runs in the handler's thread until disconnect."""
    while not client.closed:
        frame = client.read_frame()
        if frame is None:
            break

        opcode, data = frame

        if opcode == _OP_TEXT:
            _handle_ws_message(client, data.decode("utf-8", errors="replace"), output_dir)
        elif opcode == _OP_PING:
            client.send_pong(data)
        elif opcode == _OP_CLOSE:
            client.send_close()
            break


# ── HTTP Server ──────────────────────────────────────────────────────────


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    # Set by main() before server starts
    output_dir: Path = DEFAULT_OUTPUT_DIR
    assets_dir: Path = DEFAULT_ASSETS_DIR

    def do_GET(self) -> None:
        # WebSocket upgrade detection
        if (self.headers.get("Upgrade", "").lower() == "websocket"
                and "Sec-WebSocket-Key" in self.headers):
            self._handle_ws_upgrade()
            return

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
            completed = load_recent_scenes(self.output_dir, n=n)
            in_progress = load_in_progress_scenes(self.output_dir)
            self._send_json({"scenes": in_progress + completed})
        elif path.startswith("/api/scene/") and path.endswith("/frames"):
            scene_id_str = path.split("/")[3]
            try:
                scene_id = int(scene_id_str)
            except ValueError:
                self.send_error(400, "Invalid scene ID")
                return
            self._send_json({"frames": load_scene_frames(self.output_dir, scene_id)})
        elif path == "/api/errors":
            limit = int(query.get("limit", ["200"])[0])
            offset = int(query.get("offset", ["0"])[0])
            limit = max(1, min(limit, 1000))
            offset = max(0, offset)
            self._send_json(load_errors(self.output_dir, limit=limit, offset=offset))
        elif path == "/api/error-count":
            self._send_json({"count": load_error_count(self.output_dir)})
        elif path == "/api/log":
            n = int(query.get("n", ["200"])[0])
            n = max(1, min(n, LOG_BUFFER_SIZE))
            self._send_json({"lines": runner_manager.get_log(n)})
        elif path == "/api/runner":
            self._send_json({
                "running": runner_manager.is_running,
                "pid": runner_manager.pid,
                "exit_code": runner_manager.exit_code,
            })
        elif path == "/api/env-keys":
            self._send_json({
                "gemini": bool(os.environ.get("GEMINI_API_KEY")),
                "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
            })
        elif path == "/api/showcase":
            n = int(query.get("n", ["20"])[0])
            n = max(1, min(n, 100))
            self._send_json(load_showcase(self.output_dir, n=n))
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
        elif path == "/api/batch/status":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                batch_manager.poll(api_key)
            self._send_json(batch_manager.status())
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # Read request body
        content_len = int(self.headers.get("Content-Length", 0))
        body: dict[str, Any] = {}
        if content_len > 0:
            raw = self.rfile.read(content_len)
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
                return

        if path == "/api/launch":
            api_key = body.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            openrouter_key = body.get("openrouter_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
            model = body.get("model", "gemini-3-flash-preview")
            is_openrouter = "/" in model
            if not is_openrouter and not api_key:
                self._send_json({"error": "Gemini API key is required (set in .env or enter above)"}, 400)
                return
            if is_openrouter and not openrouter_key:
                self._send_json({"error": "OpenRouter key is required (set in .env or enter above)"}, 400)
                return
            try:
                result = runner_manager.launch(
                    api_key=api_key,
                    openrouter_key=body.get("openrouter_key", ""),
                    limit=body.get("limit"),
                    max_cost=body.get("max_cost"),
                    workers=body.get("workers", 10),
                    max_frames=body.get("max_frames", 20),
                    model=body.get("model", "gemini-3-flash-preview"),
                    temperature=body.get("temperature", 1.0),
                    retry_errors=body.get("retry_errors", False),
                    skip_gemini_fallback=body.get("skip_gemini_fallback", False),
                )
                self._send_json(result)
            except RuntimeError as e:
                self._send_json({"error": str(e)}, 409)
        elif path == "/api/stop":
            result = runner_manager.stop()
            self._send_json(result)
        elif path == "/api/batch/submit":
            api_key = body.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                self._send_json({"error": "Gemini API key required"}, 400)
                return
            try:
                jobs = batch_manager.submit_all(
                    api_key=api_key,
                    model=body.get("model", "gemini-3-flash-preview"),
                    max_frames=body.get("max_frames", 20),
                    temperature=body.get("temperature", 1.0),
                    scene_limit=body.get("scene_limit", 0),
                )
                self._send_json({
                    "status": "submitted",
                    "jobs_created": len(jobs),
                    "total_frames": sum(j.frame_count for j in jobs),
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/batch/cancel":
            job_name = body.get("job_name", "")
            api_key = body.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            if not job_name:
                self._send_json({"error": "job_name required"}, 400)
                return
            try:
                import requests as _req
                _req.post(
                    f"https://generativelanguage.googleapis.com/v1beta/{job_name}:cancel",
                    headers={"x-goog-api-key": api_key},
                    timeout=30,
                )
                # Update local state to reflect cancellation
                with batch_manager._lock:
                    for job in batch_manager._jobs:
                        if job.name == job_name:
                            job.state = "BATCH_STATE_CANCELLED"
                            break
                    save_batch_state(batch_manager._state_file, batch_manager._jobs)
                self._send_json({"status": "cancelled", "job": job_name})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/batch/collect":
            job_name = body.get("job_name", "")
            api_key = body.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            if not job_name or not api_key:
                self._send_json({"error": "job_name and api_key required"}, 400)
                return
            try:
                with batch_manager._lock:
                    target = next((j for j in batch_manager._jobs if j.name == job_name), None)
                if not target:
                    self._send_json({"error": f"Job {job_name} not found"}, 404)
                    return
                collect_batch_results(target, DEFAULT_OUTPUT_DIR / "images", api_key)
                with batch_manager._lock:
                    save_batch_state(batch_manager._state_file, batch_manager._jobs)
                self._send_json({
                    "status": "collected",
                    "job": job_name,
                    "stats": target.stats.to_dict() if target.stats else None,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/batch/collect-all":
            api_key = body.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                self._send_json({"error": "api_key required"}, 400)
                return
            try:
                collected_count = 0
                with batch_manager._lock:
                    succeeded = [
                        j for j in batch_manager._jobs
                        if j.state == "BATCH_STATE_SUCCEEDED" and not j.collected
                    ]
                for job in succeeded:
                    collect_batch_results(job, DEFAULT_OUTPUT_DIR / "images", api_key)
                    collected_count += 1
                with batch_manager._lock:
                    save_batch_state(batch_manager._state_file, batch_manager._jobs)
                self._send_json({"status": "collected", "jobs_collected": collected_count})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
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

    def _handle_ws_upgrade(self) -> None:
        """Perform RFC 6455 handshake and enter WebSocket read loop."""
        key = self.headers["Sec-WebSocket-Key"].strip()
        accept = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode()).digest()
        ).decode()

        # Flush any buffered wfile data, then write 101 directly on the
        # raw socket to avoid BaseHTTPRequestHandler's BufferedIOBase
        # interfering with the upgrade handshake.
        self.wfile.flush()
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        self.request.sendall(response.encode())

        client = _WebSocketClient(self.request, self.client_address)
        with _ws_clients_lock:
            _ws_clients.add(client)

        try:
            _send_init_event(client, self.output_dir, self.assets_dir)
            _ws_read_loop(client, self.output_dir)
        finally:
            with _ws_clients_lock:
                _ws_clients.discard(client)
            client.close()

    def log_message(self, format: str, *args: object) -> None:
        pass


# ── Broadcaster ──────────────────────────────────────────────────────────


class _Broadcaster:
    """Background thread that polls data sources and pushes diffs over WS."""

    def __init__(self, output_dir: Path, assets_dir: Path) -> None:
        self._output_dir = output_dir
        self._assets_dir = assets_dir
        self._hashes: dict[str, str] = {}
        self._stop = threading.Event()

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True, name="ws-broadcaster")
        t.start()

    def _run(self) -> None:
        tick = 0
        while not self._stop.is_set():
            self._stop.wait(1.0)
            if self._stop.is_set():
                break
            tick += 1

            # Skip work if nobody is listening
            with _ws_clients_lock:
                if not _ws_clients:
                    continue

            self._tick(tick)

    def _tick(self, tick: int) -> None:
        messages: list[str] = []

        # Every 1s: status, error_count, runner
        status = load_status(self._output_dir, self._assets_dir)
        self._diff_and_collect(messages, "status", status)

        ec = load_error_count(self._output_dir)
        self._diff_and_collect(messages, "error_count", {"count": ec})

        runner = {
            "running": runner_manager.is_running,
            "pid": runner_manager.pid,
            "exit_code": runner_manager.exit_code,
        }
        self._diff_and_collect(messages, "runner", runner)

        # Every 3s: log
        if tick % 3 == 0:
            log_lines = runner_manager.get_log(200)
            self._diff_and_collect(messages, "log", {"lines": log_lines})

        # Every 2s: active (in-progress) scenes — fast cadence for live progress
        if tick % 2 == 0:
            in_progress = load_in_progress_scenes(self._output_dir)
            self._diff_and_collect(
                messages, "active_scenes", {"scenes": in_progress},
            )

        # Every 5s: completed scenes
        if tick % 5 == 0:
            completed = load_recent_scenes(self._output_dir, n=50)
            self._diff_and_collect(
                messages, "completed_scenes", {"scenes": completed},
            )

        # Every 1s: frame events (individual frame completions)
        frame_events = _extract_new_frame_events(max_events=20)
        if frame_events:
            # Don't use _diff_and_collect — frame events are always "new"
            messages.append(json.dumps({
                "type": "frame_events",
                "data": {"events": frame_events},
            }))

        # Every 30s: ping all clients to keep connections alive
        if tick % 30 == 0:
            with _ws_clients_lock:
                clients = list(_ws_clients)
            dead: list[_WebSocketClient] = []
            for client in clients:
                if not client._send_raw(_WebSocketClient._build_frame(_OP_PING, b"")):
                    dead.append(client)
            if dead:
                with _ws_clients_lock:
                    for d in dead:
                        _ws_clients.discard(d)
                        d.close()

        _ws_broadcast(messages)

    def _diff_and_collect(
        self,
        messages: list[str],
        channel: str,
        data: object,
    ) -> None:
        """Serialize → hash → collect message if data changed."""
        payload = json.dumps(data, separators=(",", ":"))
        h = hashlib.md5(payload.encode()).hexdigest()
        if self._hashes.get(channel) == h:
            return
        self._hashes[channel] = h
        messages.append(json.dumps({"type": channel, "data": data}))

    def stop(self) -> None:
        self._stop.set()


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    # Load .env before anything else so API keys are in os.environ
    loaded = _load_dotenv()
    if loaded:
        print(f".env loaded: {', '.join(loaded.keys())}")

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

    broadcaster = _Broadcaster(args.output_dir, args.assets_dir)
    broadcaster.start()

    server = ThreadedServer(("", args.port), DashboardHandler)
    print(f"Caption Dashboard: http://localhost:{args.port}")
    print(f"Output dir: {args.output_dir}")
    print(f"WebSocket push: enabled")
    print(f"Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        broadcaster.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
