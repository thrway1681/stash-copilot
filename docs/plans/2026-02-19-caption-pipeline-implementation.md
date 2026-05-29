# Caption Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `caption_runner.py` — an automated script that uses SmartFrameSelector to pick up to 512 visually diverse frames per scene from pre-computed CLIP embeddings, then captions them using Gemini 3 Flash Preview for OpenCLIP LoRA training.

**Architecture:** Two-phase pipeline. Phase 1: load pre-computed OpenCLIP ViT-H-14 embeddings from numpy/FAISS files, run SmartFrameSelector per scene to pick up to 512 diverse frames. Phase 2: caption selected frames via Gemini API with rate limiting (1,000 RPM / 10,000 RPD Tier 2). Uses `ThreadPoolExecutor` for concurrent API calls within a scene, sequential scene processing for clean checkpointing.

**Tech Stack:** Python 3.12, `requests` (HTTP), `numpy` (embeddings), `concurrent.futures` (parallelism), existing `SmartFrameSelector`, existing `tools.dataset` package.

**Design doc:** `docs/plans/2026-02-19-caption-pipeline-design.md`

**Key data files:**
- Embeddings: `assets/frame_vectors_openclip-ViT-H-14.npy` (16 GB, 4,074,898 frames × 1024 dims)
- Metadata: `assets/frame_search_openclip-ViT-H-14_meta.npz` (63 MB — `scene_ids`, `frame_indices`, `timestamps`)
- SQLite backup: `assets/stash_copilot.sqlite` → `frame_embeddings` table (12,762 scenes)

**Rate limits (Tier 2):** 1,000 RPM / 10,000 RPD / 1M TPM

**Gemini pricing (source: [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)):**

| | Gemini 3 Flash | Gemini 2.0 Flash |
|---|---|---|
| Input $/1M tokens | $0.50 | $0.10 |
| Output $/1M tokens | $3.00 | $0.40 |
| Batch input $/1M | $0.25 | $0.05 |
| Batch output $/1M | $1.50 | $0.20 |
| Per-call (1,500 in + 100 out) | $0.00105 | $0.00019 |
| Batch per-call | $0.000525 | $0.000095 |

**Performance & cost estimate:**

| Metric | Value |
|---|---|
| Scenes | 12,762 |
| Frames to caption (max 512/scene) | ~2,419,013 |
| RPM limit | 1,000 |
| RPD limit | 10,000 |
| TPM limit | 1,000,000 |
| Effective daily throughput | 10,000 frames/day |
| **Days to complete (Standard)** | **~242 days** |
| **Days with Batch API** | **~1-3 days** |
| **Standard cost (3 Flash)** | **~$2,540** |
| **Batch cost (3 Flash)** | **~$1,270** |
| **Batch cost (2.0 Flash)** | **~$230** |

> **Recommendation:** Use the Batch API. Standard API at 10K RPD would take 8 months.
> Task 1 builds frame selection. Task 2 builds the API budget manager (rate limiting,
> cost tracking, budget cap, dashboard). Task 3 builds the Gemini API + caption runner.
> Task 4 adds Batch API support. Both runners share frame selection and budget tracking.

---

### Task 1: Frame Selection Module — `frame_selector.py`

**Files:**
- Create: `tools/dataset/frame_selector.py`
- Test: `tests/tools/test_frame_selector.py`

This module loads the pre-computed numpy embeddings and runs SmartFrameSelector to pick up to 512 diverse frames per scene. It's separate from the Gemini API logic so both Standard and Batch runners can use it.

**Step 1: Write the failing tests**

Create `tests/tools/test_frame_selector.py`:

```python
"""Tests for tools.dataset.frame_selector — smart frame selection from embeddings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.dataset.frame_selector import (
    load_embedding_index,
    select_frames_for_scene,
    EmbeddingIndex,
)


def _make_fake_index(tmp_path: Path, n_scenes: int = 3, frames_per: int = 100) -> Path:
    """Create fake numpy embedding files matching the real format."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    dims = 1024
    total = n_scenes * frames_per
    # Random normalized embeddings
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((total, dims)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Metadata arrays
    scene_ids = np.repeat(np.arange(1, n_scenes + 1), frames_per).astype(np.int64)
    frame_indices = np.tile(np.arange(frames_per), n_scenes).astype(np.int32)
    timestamps = frame_indices.astype(np.float32)

    np.save(assets_dir / "frame_vectors_openclip-ViT-H-14.npy", vectors)
    np.savez(
        assets_dir / "frame_search_openclip-ViT-H-14_meta.npz",
        scene_ids=scene_ids,
        frame_indices=frame_indices,
        timestamps=timestamps,
    )
    info = {
        "model_key": "openclip:ViT-H-14",
        "frame_count": total,
        "scene_count": n_scenes,
        "dimensions": dims,
    }
    (assets_dir / "frame_search_openclip-ViT-H-14_info.json").write_text(json.dumps(info))

    return assets_dir


def test_load_embedding_index(tmp_path: Path) -> None:
    assets_dir = _make_fake_index(tmp_path, n_scenes=3, frames_per=100)
    index = load_embedding_index(assets_dir)

    assert index.total_frames == 300
    assert index.scene_count == 3
    assert index.dimensions == 1024


def test_select_frames_for_scene_returns_all_when_under_max(tmp_path: Path) -> None:
    assets_dir = _make_fake_index(tmp_path, n_scenes=2, frames_per=50)
    index = load_embedding_index(assets_dir)

    # max_frames=512 > 50 frames in scene, should return all
    selected = select_frames_for_scene(index, scene_id=1, max_frames=512)

    assert len(selected) == 50
    # All should be "temporal" since no selection needed
    assert all(s.selection_reason == "temporal" for s in selected)


def test_select_frames_for_scene_caps_at_max(tmp_path: Path) -> None:
    assets_dir = _make_fake_index(tmp_path, n_scenes=2, frames_per=200)
    index = load_embedding_index(assets_dir)

    selected = select_frames_for_scene(index, scene_id=1, max_frames=64)

    assert len(selected) <= 64
    # Should have mix of temporal and novelty
    reasons = {s.selection_reason for s in selected}
    assert "temporal" in reasons


def test_select_frames_for_unknown_scene(tmp_path: Path) -> None:
    assets_dir = _make_fake_index(tmp_path, n_scenes=2, frames_per=50)
    index = load_embedding_index(assets_dir)

    selected = select_frames_for_scene(index, scene_id=999, max_frames=64)
    assert selected == []


def test_select_frames_paths_match_scene(tmp_path: Path) -> None:
    """Selected frame paths should correspond to actual frame files."""
    assets_dir = _make_fake_index(tmp_path, n_scenes=2, frames_per=50)
    index = load_embedding_index(assets_dir)

    frames_dir = tmp_path / "embedded_frames" / "scene_1"
    frames_dir.mkdir(parents=True)
    for i in range(50):
        (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"fake")

    selected = select_frames_for_scene(
        index, scene_id=1, max_frames=512, frames_dir=tmp_path / "embedded_frames",
    )
    for s in selected:
        assert Path(s.path).exists()
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/.stash/plugins/stash-copilot && uv run pytest tests/tools/test_frame_selector.py -v`
Expected: `ModuleNotFoundError: No module named 'tools.dataset.frame_selector'`

**Step 3: Write the implementation**

Create `tools/dataset/frame_selector.py`:

```python
"""Smart frame selection using pre-computed CLIP embeddings.

Loads the numpy embedding index (frame_vectors + metadata) and uses
SmartFrameSelector to pick up to max_frames visually diverse frames
per scene for captioning.

The numpy path is much faster than SQLite for bulk processing:
array slicing vs. 12K SQL queries with blob unpacking.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stash_ai.tasks.smart_frame_selector import FrameSelection, SmartFrameSelector

# Default assets directory
DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
DEFAULT_FRAMES_DIR = DEFAULT_ASSETS_DIR / "embedded_frames"
MODEL_KEY = "openclip-ViT-H-14"


@dataclass
class EmbeddingIndex:
    """In-memory embedding index for fast per-scene lookups."""

    vectors: NDArray[np.float32]       # (N, 1024) memory-mapped
    scene_ids: NDArray[np.int64]       # (N,)
    frame_indices: NDArray[np.int32]   # (N,)
    timestamps: NDArray[np.float32]    # (N,)
    total_frames: int
    scene_count: int
    dimensions: int

    # Pre-built scene → row range lookup for fast slicing
    _scene_ranges: dict[int, tuple[int, int]]

    @property
    def scene_id_list(self) -> list[int]:
        """Unique scene IDs in order."""
        return sorted(self._scene_ranges.keys())


def load_embedding_index(
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    mmap: bool = True,
) -> EmbeddingIndex:
    """Load the numpy embedding index from disk.

    Args:
        assets_dir: Directory containing frame_vectors and meta files.
        mmap: Memory-map the vectors file (recommended for 16GB file).

    Returns:
        EmbeddingIndex ready for per-scene queries.
    """
    vectors_path = assets_dir / f"frame_vectors_{MODEL_KEY}.npy"
    meta_path = assets_dir / f"frame_search_{MODEL_KEY}_meta.npz"
    info_path = assets_dir / f"frame_search_{MODEL_KEY}_info.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors file not found: {vectors_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    _log(f"Loading embedding index from {assets_dir}...")

    # Memory-map vectors for fast access without loading 16GB into RAM
    mmap_mode = "r" if mmap else None
    vectors = np.load(vectors_path, mmap_mode=mmap_mode)

    # Load metadata arrays
    meta = np.load(meta_path)
    scene_ids = meta["scene_ids"]
    frame_indices = meta["frame_indices"]
    timestamps = meta["timestamps"]

    # Load info
    info: dict[str, Any] = {}
    if info_path.exists():
        info = json.loads(info_path.read_text())

    total_frames = len(scene_ids)
    unique_scenes = np.unique(scene_ids)
    scene_count = len(unique_scenes)
    dimensions = vectors.shape[1] if vectors.ndim == 2 else info.get("dimensions", 1024)

    # Pre-build scene → row range lookup.
    # Data is ordered by (scene_id, frame_index), so each scene is a contiguous block.
    _scene_ranges: dict[int, tuple[int, int]] = {}
    # Use np.searchsorted on the sorted scene_ids array
    for sid in unique_scenes:
        mask = scene_ids == sid
        indices = np.where(mask)[0]
        _scene_ranges[int(sid)] = (int(indices[0]), int(indices[-1]) + 1)

    _log(f"  Loaded: {total_frames:,} frames, {scene_count:,} scenes, {dimensions}d")

    return EmbeddingIndex(
        vectors=vectors,
        scene_ids=scene_ids,
        frame_indices=frame_indices,
        timestamps=timestamps,
        total_frames=total_frames,
        scene_count=scene_count,
        dimensions=dimensions,
        _scene_ranges=_scene_ranges,
    )


def select_frames_for_scene(
    index: EmbeddingIndex,
    scene_id: int,
    max_frames: int = 512,
    temporal_ratio: float = 0.5,
    dedup_threshold: float = 0.90,
    frames_dir: Path = DEFAULT_FRAMES_DIR,
) -> list[FrameSelection]:
    """Select diverse frames for a scene using SmartFrameSelector.

    Args:
        index: Pre-loaded embedding index.
        scene_id: Stash scene ID.
        max_frames: Maximum frames to select (default 512).
        temporal_ratio: Fraction of budget for temporal baseline.
        dedup_threshold: Cosine similarity threshold for dedup.
        frames_dir: Root directory containing scene_*/frame_*.jpg dirs.

    Returns:
        List of FrameSelection with paths, timestamps, novelty scores.
        Empty list if scene not found in index.
    """
    if scene_id not in index._scene_ranges:
        return []

    start, end = index._scene_ranges[scene_id]

    # Extract this scene's data (contiguous slice — fast for mmap)
    embeddings = np.array(index.vectors[start:end], dtype=np.float32)
    frame_idxs = index.frame_indices[start:end]
    scene_timestamps = index.timestamps[start:end].tolist()

    # Build frame paths from frame indices
    scene_dir = frames_dir / f"scene_{scene_id}"
    frame_paths: list[str] = []
    for fidx in frame_idxs:
        frame_paths.append(str(scene_dir / f"frame_{int(fidx):04d}.jpg"))

    # Normalize embeddings (SmartFrameSelector assumes unit vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    # Run SmartFrameSelector
    selector = SmartFrameSelector(
        windows=[2, 15, 60],
        weights=[0.0, 1.0, 1.0],
        dedup_threshold=dedup_threshold,
    )

    return selector.select_frames(
        frame_paths=frame_paths,
        embeddings=embeddings,
        timestamps=scene_timestamps,
        max_frames=max_frames,
        temporal_ratio=temporal_ratio,
    )


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/.stash/plugins/stash-copilot && uv run pytest tests/tools/test_frame_selector.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add tools/dataset/frame_selector.py tests/tools/test_frame_selector.py
git commit -m "feat(dataset): add frame_selector module using SmartFrameSelector + numpy embeddings"
```

---

### Task 2: API Budget Module — `api_budget.py`

**Files:**
- Create: `tools/dataset/api_budget.py`
- Test: `tests/tools/test_api_budget.py`

Thread-safe rate limiter, cost tracker, budget cap, and dashboard logger. All values are **measured, not estimated**:
- Prompt token count is measured once at startup via Gemini's `countTokens` API
- Per-call token usage is read from `usageMetadata` in each `generateContent` response
- Cost is computed from actual tokens × known pricing (from Google's pricing page)

**Step 1: Write the failing tests**

Create `tests/tools/test_api_budget.py`:

```python
"""Tests for tools.dataset.api_budget — rate limiting, cost tracking, dashboard."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.dataset.api_budget import (
    ApiBudget,
    BudgetExhausted,
    DailyLimitReached,
    GeminiPricing,
    PRICING,
    compute_cost,
)


# ── Pricing ──────────────────────────────────────────────────────────────


def test_pricing_table_has_known_models() -> None:
    assert "gemini-3.0-flash-preview" in PRICING
    assert "gemini-2.0-flash" in PRICING


def test_compute_cost_uses_actual_tokens() -> None:
    pricing = PRICING["gemini-3.0-flash-preview"]
    # 1,500 input tokens × $0.50/1M = $0.00075
    # 100 output tokens × $3.00/1M = $0.0003
    cost = compute_cost(pricing, input_tokens=1_500, output_tokens=100)
    assert abs(cost - 0.00105) < 0.00001


def test_compute_cost_batch_pricing() -> None:
    pricing = PRICING["gemini-3.0-flash-preview:batch"]
    cost = compute_cost(pricing, input_tokens=1_500, output_tokens=100)
    assert abs(cost - 0.000525) < 0.00001


# ── Budget cap ───────────────────────────────────────────────────────────


def test_budget_exhausted_raises() -> None:
    budget = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=0.001,
    )
    # Simulate recording usage that exceeds budget
    budget.record_usage(input_tokens=1_500, output_tokens=100)  # $0.00105
    with pytest.raises(BudgetExhausted):
        budget.acquire()


def test_no_budget_cap_allows_unlimited() -> None:
    budget = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=None,
    )
    budget.record_usage(input_tokens=1_500, output_tokens=100)
    budget.acquire()  # should not raise


# ── RPD tracking ─────────────────────────────────────────────────────────


def test_rpd_limit_raises() -> None:
    budget = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=2,
        max_cost=None,
    )
    budget.record_usage(input_tokens=100, output_tokens=10)
    budget.record_usage(input_tokens=100, output_tokens=10)
    with pytest.raises(DailyLimitReached):
        budget.acquire()


# ── RPM throttling ───────────────────────────────────────────────────────


def test_rpm_throttle_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """When RPM is at limit, acquire() should sleep until window clears."""
    budget = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=2,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=None,
    )
    # Fill RPM window
    now = time.monotonic()
    budget._rpm_timestamps.append(now)
    budget._rpm_timestamps.append(now)

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    # Advance clock slightly so the sleep duration is positive
    monkeypatch.setattr(time, "monotonic", lambda: now + 0.1)

    budget.acquire()
    assert len(sleep_calls) >= 1
    assert sleep_calls[0] > 0


# ── TPM throttling ───────────────────────────────────────────────────────


def test_tpm_throttle_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=1000,
        tpm_limit=100,  # very low for test
        rpd_limit=10_000,
        max_cost=None,
    )
    now = time.monotonic()
    # Record tokens that fill the TPM window
    budget._tpm_entries.append((now, 100))
    budget._tpm_tokens_in_window = 100

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(time, "monotonic", lambda: now + 0.1)

    budget.acquire()
    assert len(sleep_calls) >= 1


# ── State persistence ────────────────────────────────────────────────────


def test_state_persistence(tmp_path: Path) -> None:
    state_file = tmp_path / "budget_state.json"
    budget1 = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=100.0,
        state_file=state_file,
    )
    budget1.record_usage(input_tokens=1_500, output_tokens=100)
    budget1.record_usage(input_tokens=1_500, output_tokens=100)
    budget1.save_state()

    # New instance should restore state
    budget2 = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=1000,
        tpm_limit=1_000_000,
        rpd_limit=10_000,
        max_cost=100.0,
        state_file=state_file,
    )
    assert budget2.total_calls == 2
    assert budget2.total_cost > 0
    assert budget2.total_input_tokens == 3_000
    assert budget2.total_output_tokens == 200


# ── Thread safety ────────────────────────────────────────────────────────


def test_concurrent_record_usage() -> None:
    budget = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=10_000,
        tpm_limit=100_000_000,
        rpd_limit=100_000,
        max_cost=None,
    )
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(100):
                budget.record_usage(input_tokens=100, output_tokens=10)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert budget.total_calls == 1000


# ── Dashboard ────────────────────────────────────────────────────────────


def test_dashboard_format() -> None:
    budget = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=900,
        tpm_limit=900_000,
        rpd_limit=9_500,
        max_cost=50.0,
    )
    budget.record_usage(input_tokens=1_500, output_tokens=100)
    budget._total_frames = 2_419_013

    text = budget.dashboard()
    assert "Calls:" in text
    assert "Cost:" in text
    assert "$50.00" in text  # budget cap
    assert "RPD:" in text


# ── Cost estimation ──────────────────────────────────────────────────────


def test_estimate_total_cost_from_measured() -> None:
    budget = ApiBudget(
        model="gemini-3.0-flash-preview",
        rpm_limit=900,
        tpm_limit=900_000,
        rpd_limit=9_500,
        max_cost=None,
    )
    # Simulate measuring prompt tokens via countTokens API
    budget.measured_input_tokens_per_call = 1_500
    budget.measured_output_tokens_per_call = 100

    estimated = budget.estimate_total_cost(n_frames=2_419_013)
    # 2,419,013 × $0.00105 = ~$2,540
    assert 2_400 < estimated < 2_700
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/.stash/plugins/stash-copilot && uv run pytest tests/tools/test_api_budget.py -v`
Expected: `ModuleNotFoundError: No module named 'tools.dataset.api_budget'`

**Step 3: Write the implementation**

Create `tools/dataset/api_budget.py`:

```python
"""Thread-safe API budget manager for Gemini caption pipeline.

Provides:
- Rate limiting: RPM (requests/min), TPM (tokens/min), RPD (requests/day)
- Cost tracking: actual cost from real token counts, not estimates
- Budget cap: hard stop when spending exceeds --max-cost
- Dashboard: periodic status with real numbers
- State persistence: RPD + cost survive restarts

All token counts come from real API responses (usageMetadata), not estimates.
The only "estimate" is the pre-run cost projection, which uses measured
prompt tokens from the countTokens API + average output tokens from early calls.

Pricing source: https://ai.google.dev/gemini-api/docs/pricing
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


# ── Exceptions ───────────────────────────────────────────────────────────


class BudgetExhausted(Exception):
    """Raised when spending exceeds the configured max_cost."""


class DailyLimitReached(Exception):
    """Raised when RPD limit for today is reached."""


# ── Pricing ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeminiPricing:
    """Pricing per 1M tokens for a Gemini model."""
    input_per_m: float   # $/1M input tokens (text + image + video)
    output_per_m: float  # $/1M output tokens


# Source: https://ai.google.dev/gemini-api/docs/pricing (2026-02-19)
PRICING: dict[str, GeminiPricing] = {
    "gemini-3.0-flash-preview": GeminiPricing(input_per_m=0.50, output_per_m=3.00),
    "gemini-3.0-flash-preview:batch": GeminiPricing(input_per_m=0.25, output_per_m=1.50),
    "gemini-2.0-flash": GeminiPricing(input_per_m=0.10, output_per_m=0.40),
    "gemini-2.0-flash:batch": GeminiPricing(input_per_m=0.05, output_per_m=0.20),
}


def compute_cost(pricing: GeminiPricing, input_tokens: int, output_tokens: int) -> float:
    """Compute actual cost from real token counts."""
    return (input_tokens * pricing.input_per_m / 1_000_000) + \
           (output_tokens * pricing.output_per_m / 1_000_000)


# ── countTokens API ─────────────────────────────────────────────────────


def count_tokens(
    model: str,
    api_key: str,
    prompt: str,
    frame_b64: str,
) -> int:
    """Call Gemini's countTokens API to measure exact input token count.

    This is FREE (no billing) and has a separate 3,000 RPM quota.
    Should be called once at startup with a sample frame to measure
    the real prompt + image token count.

    Returns:
        Total input token count (prompt text + image tokens).
    """
    url = f"{GEMINI_API_BASE}/models/{model}:countTokens"
    payload = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
            {"text": prompt},
        ]}],
    }
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("totalTokens", 0)


# ── ApiBudget ────────────────────────────────────────────────────────────


class ApiBudget:
    """Thread-safe API budget manager.

    Usage:
        budget = ApiBudget(model="gemini-3.0-flash-preview", ...)

        # At startup, measure real token count:
        budget.measured_input_tokens_per_call = count_tokens(model, key, prompt, sample_b64)
        print(budget.estimate_total_cost(n_frames=2_419_013))

        # In each worker thread:
        budget.acquire()             # blocks until rate limits allow
        response = call_gemini(...)  # your API call
        usage = response["usageMetadata"]
        budget.record_usage(
            input_tokens=usage["promptTokenCount"],
            output_tokens=usage["candidatesTokenCount"],
        )
    """

    def __init__(
        self,
        model: str,
        rpm_limit: int = 900,
        tpm_limit: int = 900_000,
        rpd_limit: int = 9_500,
        max_cost: float | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.model = model
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.max_cost = max_cost
        self.state_file = state_file

        # Pricing lookup
        self.pricing = PRICING.get(model, PRICING.get("gemini-2.0-flash"))

        # Measured values (set after calling countTokens)
        self.measured_input_tokens_per_call: int = 0
        self.measured_output_tokens_per_call: int = 0  # updated as rolling avg

        # Accumulators
        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost: float = 0.0
        self.total_errors: int = 0
        self._total_frames: int = 0  # set externally for dashboard %

        # RPD tracking (date-scoped)
        self._rpd_count: int = 0
        self._rpd_date: str = date.today().isoformat()

        # Sliding windows (thread-safe via lock)
        self._lock = threading.Lock()
        self._rpm_timestamps: deque[float] = deque()
        self._tpm_entries: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._tpm_tokens_in_window: int = 0

        # Start time for dashboard
        self._start_time: float = time.monotonic()

        # Load persisted state if available
        if state_file and state_file.exists():
            self._load_state()

    # ── Rate limiting ────────────────────────────────────────────────────

    def acquire(self) -> None:
        """Block until rate limits allow another API call.

        Raises:
            BudgetExhausted: if total_cost >= max_cost
            DailyLimitReached: if RPD count >= rpd_limit
        """
        with self._lock:
            # 1. Check budget cap
            if self.max_cost is not None and self.total_cost >= self.max_cost:
                raise BudgetExhausted(
                    f"Budget exhausted: ${self.total_cost:.2f} >= ${self.max_cost:.2f} cap"
                )

            # 2. Check RPD
            today = date.today().isoformat()
            if today != self._rpd_date:
                self._rpd_count = 0
                self._rpd_date = today
            if self._rpd_count >= self.rpd_limit:
                raise DailyLimitReached(
                    f"Daily limit reached: {self._rpd_count} >= {self.rpd_limit} RPD"
                )

        # 3. RPM throttle (outside lock to avoid blocking other threads on sleep)
        self._wait_for_rpm()

        # 4. TPM throttle
        self._wait_for_tpm()

    def _wait_for_rpm(self) -> None:
        """Sleep until the RPM sliding window has room."""
        while True:
            now = time.monotonic()
            with self._lock:
                # Evict timestamps older than 60s
                while self._rpm_timestamps and self._rpm_timestamps[0] < now - 60:
                    self._rpm_timestamps.popleft()

                if len(self._rpm_timestamps) < self.rpm_limit:
                    self._rpm_timestamps.append(now)
                    return

                # Calculate sleep time until oldest entry expires
                sleep_until = self._rpm_timestamps[0] + 60
                wait = sleep_until - now

            if wait > 0:
                time.sleep(wait)

    def _wait_for_tpm(self) -> None:
        """Sleep until the TPM sliding window has room."""
        # Estimate tokens for next call
        est_tokens = self.measured_input_tokens_per_call + self.measured_output_tokens_per_call
        if est_tokens <= 0:
            est_tokens = 1_600  # fallback if not measured yet

        while True:
            now = time.monotonic()
            with self._lock:
                # Evict entries older than 60s
                while self._tpm_entries and self._tpm_entries[0][0] < now - 60:
                    _, tokens = self._tpm_entries.popleft()
                    self._tpm_tokens_in_window -= tokens

                if self._tpm_tokens_in_window + est_tokens <= self.tpm_limit:
                    return

                # Wait until oldest entry expires
                if self._tpm_entries:
                    sleep_until = self._tpm_entries[0][0] + 60
                    wait = sleep_until - now
                else:
                    wait = 1.0

            if wait > 0:
                time.sleep(wait)

    # ── Recording ────────────────────────────────────────────────────────

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record actual token usage from a completed API call.

        Args:
            input_tokens: From usageMetadata.promptTokenCount
            output_tokens: From usageMetadata.candidatesTokenCount
        """
        cost = compute_cost(self.pricing, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self._rpd_count += 1

            # Update TPM window
            now = time.monotonic()
            self._tpm_entries.append((now, total_tokens))
            self._tpm_tokens_in_window += total_tokens

            # Update rolling average of output tokens (for cost estimation)
            if self.total_calls > 0:
                self.measured_output_tokens_per_call = (
                    self.total_output_tokens // self.total_calls
                )

    def record_error(self) -> None:
        """Record a failed API call (no tokens consumed)."""
        with self._lock:
            self.total_errors += 1

    # ── Cost estimation ──────────────────────────────────────────────────

    def estimate_total_cost(self, n_frames: int) -> float:
        """Pre-run cost estimate using measured token counts.

        Call this after setting measured_input_tokens_per_call (from countTokens API)
        and optionally measured_output_tokens_per_call (from early runs or default).

        Returns estimated total cost in USD.
        """
        input_per_call = self.measured_input_tokens_per_call
        output_per_call = self.measured_output_tokens_per_call or 100  # output default
        return n_frames * compute_cost(self.pricing, input_per_call, output_per_call)

    # ── Dashboard ────────────────────────────────────────────────────────

    def dashboard(self) -> str:
        """Formatted dashboard string with real numbers."""
        elapsed = time.monotonic() - self._start_time
        elapsed_str = _format_duration(elapsed)

        with self._lock:
            calls = self.total_calls
            cost = self.total_cost
            errors = self.total_errors
            rpd = self._rpd_count
            input_tok = self.total_input_tokens
            output_tok = self.total_output_tokens

            # Current RPM (calls in last 60s)
            now = time.monotonic()
            while self._rpm_timestamps and self._rpm_timestamps[0] < now - 60:
                self._rpm_timestamps.popleft()
            current_rpm = len(self._rpm_timestamps)

            # Current TPM
            while self._tpm_entries and self._tpm_entries[0][0] < now - 60:
                _, t = self._tpm_entries.popleft()
                self._tpm_tokens_in_window -= t
            current_tpm = self._tpm_tokens_in_window

        # Progress
        total = self._total_frames or 1
        pct = calls / total * 100 if total else 0

        # ETA
        if calls > 0 and elapsed > 0:
            rate = calls / elapsed  # calls/sec
            remaining = total - calls
            eta_secs = remaining / rate if rate > 0 else 0
            eta_str = _format_duration(eta_secs)
        else:
            eta_str = "calculating..."

        # Cost per call (actual average)
        avg_cost = cost / calls if calls else 0

        # Budget line
        if self.max_cost is not None:
            budget_str = f"${cost:.2f} / ${self.max_cost:.2f} budget ({cost / self.max_cost * 100:.1f}%)"
        else:
            budget_str = f"${cost:.2f} (no cap)"

        lines = [
            "── Dashboard ──────────────────────────────────",
            f"  Calls:       {calls:,} / {total:,} ({pct:.2f}%)",
            f"  Cost:        {budget_str}",
            f"  Avg cost:    ${avg_cost:.6f}/call",
            f"  Input tok:   {input_tok:,}  Output tok: {output_tok:,}",
            f"  RPM:         {current_rpm} / {self.rpm_limit}",
            f"  TPM:         {current_tpm:,} / {self.tpm_limit:,}",
            f"  RPD:         {rpd:,} / {self.rpd_limit:,}",
            f"  Errors:      {errors}",
            f"  Elapsed:     {elapsed_str}",
            f"  ETA:         {eta_str}",
            "───────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist budget state to disk for resume across restarts."""
        if not self.state_file:
            return
        state = {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "total_errors": self.total_errors,
            "rpd_count": self._rpd_count,
            "rpd_date": self._rpd_date,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_state(self) -> None:
        """Restore budget state from disk."""
        if not self.state_file or not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.total_calls = state.get("total_calls", 0)
            self.total_input_tokens = state.get("total_input_tokens", 0)
            self.total_output_tokens = state.get("total_output_tokens", 0)
            self.total_cost = state.get("total_cost", 0.0)
            self.total_errors = state.get("total_errors", 0)

            # RPD: only restore if same day
            saved_rpd_date = state.get("rpd_date", "")
            if saved_rpd_date == date.today().isoformat():
                self._rpd_count = state.get("rpd_count", 0)
                self._rpd_date = saved_rpd_date

            # Restore rolling average
            if self.total_calls > 0:
                self.measured_output_tokens_per_call = (
                    self.total_output_tokens // self.total_calls
                )
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt state file, start fresh


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m {seconds % 60:.0f}s"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.0f}h {minutes % 60:.0f}m"
    days = hours / 24
    return f"{days:.1f}d {hours % 24:.0f}h"
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/.stash/plugins/stash-copilot && uv run pytest tests/tools/test_api_budget.py -v`
Expected: All 11 tests PASS.

**Step 5: Commit**

```bash
git add tools/dataset/api_budget.py tests/tools/test_api_budget.py
git commit -m "feat(dataset): add api_budget module — rate limiting, cost tracking, dashboard"
```

---

### Task 3: Gemini API Module + Caption Runner (Standard API)

**Files:**
- Create: `tools/dataset/gemini_api.py`
- Create: `tools/dataset/caption_runner.py`
- Test: `tests/tools/test_gemini_api.py`
- Test: `tests/tools/test_caption_runner.py`

The Gemini API module wraps `generateContent` and returns both the caption and full `usageMetadata` for the budget tracker. The caption runner integrates `frame_selector.py` + `api_budget.py`.

**Step 1: Write `gemini_api.py` tests**

Create `tests/tools/test_gemini_api.py`:

```python
"""Tests for tools.dataset.gemini_api — Gemini generateContent wrapper."""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from tools.dataset.gemini_api import (
    caption_frame,
    parse_response,
    CaptionResult,
)


def _make_response(caption: str, input_tok: int = 1500, output_tok: int = 100) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": caption}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": input_tok,
            "candidatesTokenCount": output_tok,
            "totalTokenCount": input_tok + output_tok,
        },
    }


def test_parse_response_extracts_caption_and_usage() -> None:
    resp = _make_response("A doggy style scene.", input_tok=1500, output_tok=80)
    result = parse_response(resp)
    assert result.caption == "A doggy style scene."
    assert result.input_tokens == 1500
    assert result.output_tokens == 80


def test_parse_response_raises_on_blocked() -> None:
    resp = {
        "candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}],
        "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
    }
    with pytest.raises(RuntimeError, match="blocked"):
        parse_response(resp)


def test_parse_response_raises_on_empty() -> None:
    resp = {"candidates": [], "usageMetadata": {}}
    with pytest.raises(RuntimeError):
        parse_response(resp)


@patch("tools.dataset.gemini_api.requests.post")
def test_caption_frame_returns_result(mock_post: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response("Cowgirl POV.", 1500, 60)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = caption_frame("base64data", "prompt", "gemini-3.0-flash-preview", "key", 1.0)
    assert result.caption == "Cowgirl POV."
    assert result.input_tokens == 1500
    assert result.output_tokens == 60
```

**Step 2: Write `gemini_api.py` implementation**

Create `tools/dataset/gemini_api.py`:

```python
"""Gemini generateContent wrapper that returns caption + real token usage.

Every API call returns a CaptionResult with:
- caption: the text output
- input_tokens: from usageMetadata.promptTokenCount (measured, not estimated)
- output_tokens: from usageMetadata.candidatesTokenCount (measured)

These feed directly into ApiBudget.record_usage() for accurate cost tracking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class CaptionResult:
    """Result from a single generateContent call."""
    caption: str
    input_tokens: int
    output_tokens: int


def parse_response(response: dict) -> CaptionResult:
    """Extract caption + token usage from a Gemini generateContent response.

    Raises RuntimeError if the response is blocked or empty.
    """
    # Check for prompt-level blocking
    if "promptFeedback" in response:
        reason = response["promptFeedback"].get("blockReason")
        if reason:
            raise RuntimeError(f"Prompt blocked: {reason}")

    candidates = response.get("candidates", [])
    if not candidates:
        raise RuntimeError("No candidates in Gemini response")

    candidate = candidates[0]
    finish = candidate.get("finishReason", "STOP")
    if finish not in ("STOP", "MAX_TOKENS"):
        raise RuntimeError(f"Generation blocked: {finish}")

    parts = candidate.get("content", {}).get("parts", [])
    if not parts or "text" not in parts[0]:
        raise RuntimeError(f"No text in response (finishReason={finish})")

    caption = parts[0]["text"].strip()

    # Extract actual token counts from usageMetadata
    usage = response.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)

    return CaptionResult(
        caption=caption,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def caption_frame(
    frame_b64: str,
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_retries: int = 3,
) -> CaptionResult:
    """Caption a single frame via Gemini generateContent.

    Retries on 429 (rate limit) and 5xx (server error) with exponential backoff.

    Returns CaptionResult with caption text and actual token usage.
    """
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=120)
            resp.raise_for_status()
            return parse_response(resp.json())
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else 0
            if status == 429 or status >= 500:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.ConnectionError as e:
            last_error = e
            wait = 2 ** attempt
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")
```

**Step 3: Write the caption runner tests**

Create `tests/tools/test_caption_runner.py`:

```python
"""Tests for tools.dataset.caption_runner — scene processing and orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from tools.dataset.caption_runner import (
    process_scene_frames,
    PROMPT,
)
from tools.dataset.gemini_api import CaptionResult
from tools.dataset.io_utils import dataset_image_name


def _make_scene_dir(tmp_path: Path, scene_id: str, n_frames: int = 3) -> Path:
    """Create a fake scene directory with JPEG frames."""
    scene_dir = tmp_path / "embedded_frames" / f"scene_{scene_id}"
    scene_dir.mkdir(parents=True)
    for i in range(n_frames):
        (scene_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8fake jpeg")
    return scene_dir


def test_prompt_contains_key_elements() -> None:
    assert "CLIP LoRA" in PROMPT
    assert "casual, informal, slang" in PROMPT
    assert "cowgirl" in PROMPT


@patch("tools.dataset.caption_runner.caption_frame")
def test_process_scene_frames_writes_files(
    mock_caption: MagicMock, tmp_path: Path,
) -> None:
    mock_caption.return_value = CaptionResult(
        caption="A cowgirl scene with a fit brunette.",
        input_tokens=1500, output_tokens=80,
    )

    scene_dir = _make_scene_dir(tmp_path, "99", n_frames=3)
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    frame_paths = [str(p) for p in sorted(scene_dir.glob("frame_*.jpg"))]

    # Create a mock budget that does nothing
    mock_budget = MagicMock()

    result = process_scene_frames(
        scene_id="99",
        frame_paths=frame_paths,
        prompt="test prompt",
        model="gemini-3.0-flash-preview",
        api_key="fake-key",
        temperature=1.0,
        images_dir=images_dir,
        budget=mock_budget,
        workers=2,
    )

    assert len(result.image_names) == 3
    assert len(result.captions) == 3

    for name in result.image_names:
        assert (images_dir / name).exists()
        txt = images_dir / name.replace(".jpg", ".txt")
        assert txt.exists()
        assert txt.read_text() == "A cowgirl scene with a fit brunette."

    assert mock_caption.call_count == 3
    # Budget should have been called for each frame
    assert mock_budget.acquire.call_count == 3
    assert mock_budget.record_usage.call_count == 3


@patch("tools.dataset.caption_runner.caption_frame")
def test_process_scene_frames_skips_existing(
    mock_caption: MagicMock, tmp_path: Path,
) -> None:
    mock_caption.return_value = CaptionResult(
        caption="New caption.", input_tokens=1500, output_tokens=50,
    )

    scene_dir = _make_scene_dir(tmp_path, "50", n_frames=2)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frame_paths = [str(p) for p in sorted(scene_dir.glob("frame_*.jpg"))]

    # Pre-create one caption file
    name0 = dataset_image_name("50", Path(frame_paths[0]))
    (images_dir / name0).write_bytes(b"img")
    (images_dir / name0.replace(".jpg", ".txt")).write_text("Existing.")

    mock_budget = MagicMock()

    result = process_scene_frames(
        scene_id="50",
        frame_paths=frame_paths,
        prompt="test",
        model="m",
        api_key="k",
        temperature=1.0,
        images_dir=images_dir,
        budget=mock_budget,
        workers=1,
    )

    # Only 1 API call (skipped existing)
    assert mock_caption.call_count == 1
    # Budget acquire only called for the 1 real API call
    assert mock_budget.acquire.call_count == 1
    assert (images_dir / name0.replace(".jpg", ".txt")).read_text() == "Existing."


@patch("tools.dataset.caption_runner.caption_frame")
def test_process_scene_frames_handles_api_error(
    mock_caption: MagicMock, tmp_path: Path,
) -> None:
    mock_caption.side_effect = [
        RuntimeError("blocked"),
        CaptionResult(caption="Good caption.", input_tokens=1500, output_tokens=60),
    ]

    scene_dir = _make_scene_dir(tmp_path, "77", n_frames=2)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frame_paths = [str(p) for p in sorted(scene_dir.glob("frame_*.jpg"))]

    mock_budget = MagicMock()

    result = process_scene_frames(
        scene_id="77",
        frame_paths=frame_paths,
        prompt="test",
        model="m",
        api_key="k",
        temperature=1.0,
        images_dir=images_dir,
        budget=mock_budget,
        workers=1,
    )

    assert len(result.image_names) == 2
    error_txt = (images_dir / result.image_names[0].replace(".jpg", ".txt")).read_text()
    assert "[ERROR" in error_txt
    ok_txt = (images_dir / result.image_names[1].replace(".jpg", ".txt")).read_text()
    assert ok_txt == "Good caption."
    # Budget should record error for the failed call
    assert mock_budget.record_error.call_count == 1
```

**Step 4: Write the caption runner implementation**

Create `tools/dataset/caption_runner.py`:

```python
#!/usr/bin/env python3
"""Automated caption runner — selects diverse frames, captions via Gemini, tracks budget.

Phase 1: Load pre-computed CLIP embeddings, run SmartFrameSelector per scene.
Phase 2: Caption selected frames via Gemini API (single frame per call).
         Every call goes through ApiBudget for rate limiting + cost tracking.
Phase 3: Dashboard + checkpoint after each scene.

Usage:
    uv run python tools/dataset/caption_runner.py
    uv run python tools/dataset/caption_runner.py --limit 10 --max-cost 5.00
    uv run python tools/dataset/caption_runner.py --max-frames 256 --workers 5
    uv run python tools/dataset/caption_runner.py --dry-run
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.dataset.gemini_api import caption_frame, CaptionResult
from tools.dataset.api_budget import ApiBudget, BudgetExhausted, DailyLimitReached, count_tokens
from tools.dataset.frame_selector import (
    DEFAULT_ASSETS_DIR,
    DEFAULT_FRAMES_DIR,
    EmbeddingIndex,
    load_embedding_index,
    select_frames_for_scene,
)
from tools.dataset.io_utils import dataset_image_name

# ── Constants ───────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "lora_dataset"
CHECKPOINT_FILE = "caption_progress.json"
DASHBOARD_INTERVAL = 10  # scenes between dashboard prints

# ── Prompt ──────────────────────────────────────────────────────────────

PROMPT = """\
You are captioning a single video frame for a CLIP LoRA training dataset.
Describe ONLY what is visible in this image. You have NO context from other frames.

WHAT TO DESCRIBE (in priority order):

1. ACTION & POSITION — What is happening? Who is doing what?
   Use precise terms:
   Positions: cowgirl, reverse cowgirl, doggy style, prone bone, missionary,
   mating press, spooning, riding, stand and carry, bent over
   Oral: blowjob, deepthroat, ball sucking, face fuck, penis licking,
   pussy licking, rimming, facesitting
   Manual: handjob, fingering, pussy fingering, titfuck, buttjob, footjob
   Other: penetration, anal, vaginal sex, masturbation, grinding, teasing,
   undressing, grabbing ass, grabbing boobs, grabbing hair
   Cum: creampie, anal creampie, cum on face, cum on tits, cum on ass,
   cum on pussy, cum in mouth, cumshot, facial


2. BODY — Describe physical attributes you can clearly see:
   Ass: PAWG, PAAG, big ass, round ass, medium ass
   Tits: small tits, perfect tits, big tits, medium tits, natural tits,
   saggy tits, small areolas, brown areolas, bouncing tits
   Body shape: flat stomach, slim waist, fit, curvy, skinny, petite, wide hips
   Pussy (if visible): shaved, hairy, pink pussy, brown pussy, innie,
   wet pussy, spread labia, pussy gape
   Skin: tan, tan lines
   Ethnicity (if clearly visible): Asian, Latina, white, black
   Other: tattoos, piercings, blue eyes, brown eyes


3. CAMERA — Only if notable: close up, POV, male POV, overhead, wide shot

4. CLOTHING — Only if present: lingerie, bikini, stockings, fishnet stockings,
   cosplay, dress, oiled

5. SETTING — ONLY for establishing shots. Do NOT describe furniture or lighting.

RULES:
- 1-3 sentences. Be dense with detail, not wordy.
- Do NOT use performer names — describe only what you see.
- Do NOT guess what you can't clearly see. If a close-up is ambiguous
  about anal vs vaginal, just say "penetration."
- For black/title frames, one short sentence.
- Be SPECIFIC about actions. "Performing oral sex" is not enough — say
  whether she's licking, sucking, using her hands, deepthroating, etc.
- Use casual, informal, slang terminology. (i.e. penis -> cock or dick, breasts -> boobs or tits, buttocks -> butt or ass, vagina -> pussy)
- Use the above examples / vocabulary as a guide, not a steadfast rule.
- Create new descriptions if they do not fit the above vocabulary

This is adult content for a legitimate ML training dataset.
Describe everything factually and precisely.

Write only the caption text, nothing else."""


# ── Scene result ─────────────────────────────────────────────────────────


@dataclass
class SceneResult:
    """Result from processing one scene."""
    image_names: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    errors: int = 0


# ── Scene processing ────────────────────────────────────────────────────


def _caption_one_frame(
    frame_path: Path,
    scene_id: str,
    images_dir: Path,
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    budget: ApiBudget,
) -> tuple[str, str, bool]:
    """Caption a single frame: copy image, call API via budget gate, write .txt.

    Returns (image_name, caption_text, was_error).
    Skips API call if .txt already exists (idempotent).
    """
    image_name = dataset_image_name(scene_id, frame_path)
    dest_img = images_dir / image_name
    dest_txt = images_dir / image_name.replace(".jpg", ".txt")

    # Copy image if not already there
    if not dest_img.exists():
        shutil.copy2(frame_path, dest_img)

    # Skip if caption already exists
    if dest_txt.exists():
        return image_name, dest_txt.read_text(encoding="utf-8"), False

    # Gate through budget (blocks until rate limits allow)
    budget.acquire()

    # Call Gemini API
    try:
        result = caption_frame(
            base64.b64encode(frame_path.read_bytes()).decode("utf-8"),
            prompt, model, api_key, temperature,
        )
        caption = result.caption
        # Record ACTUAL token usage from API response
        budget.record_usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        was_error = False
    except Exception as e:
        caption = f"[ERROR: {e}]"
        budget.record_error()
        was_error = True
        _log(f"    ERROR {image_name}: {e}")

    dest_txt.write_text(caption, encoding="utf-8")
    return image_name, caption, was_error


def process_scene_frames(
    scene_id: str,
    frame_paths: list[str],
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    images_dir: Path,
    budget: ApiBudget,
    workers: int = 10,
) -> SceneResult:
    """Caption a list of frame paths for one scene using a thread pool.

    Returns SceneResult with image_names, captions, and error count.
    """
    results: dict[str, tuple[str, str, bool]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _caption_one_frame,
                Path(fp), scene_id, images_dir, prompt, model, api_key, temperature, budget,
            ): fp
            for fp in frame_paths
        }
        for future in as_completed(futures):
            fp = futures[future]
            try:
                results[fp] = future.result()
            except (BudgetExhausted, DailyLimitReached):
                raise  # propagate budget stops
            except Exception as e:
                img_name = dataset_image_name(scene_id, Path(fp))
                results[fp] = (img_name, f"[ERROR: {e}]", True)

    # Return in original frame order
    scene_result = SceneResult()
    for fp in frame_paths:
        img_name, caption, was_error = results[fp]
        scene_result.image_names.append(img_name)
        scene_result.captions.append(caption)
        if was_error:
            scene_result.errors += 1

    return scene_result


# ── Checkpoint ──────────────────────────────────────────────────────────


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed_scenes": [], "total_frames_captioned": 0, "errors": 0}


def _save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    data["last_updated"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Metadata ────────────────────────────────────────────────────────────


def _append_metadata(
    jsonl_path: Path,
    scene_id: int,
    image_names: list[str],
    captions: list[str],
    selection_stats: dict[str, Any],
) -> None:
    record = {
        "scene_id": scene_id,
        "image_names": image_names,
        "captions": {name: cap for name, cap in zip(image_names, captions)},
        "selection": selection_stats,
        "captioned_at": datetime.now(UTC).isoformat(),
        "method": "gemini-vlm+smart-select",
    }
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Main loop ───────────────────────────────────────────────────────────


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    model: str = "gemini-3.0-flash-preview",
    temperature: float = 1.0,
    api_key: str | None = None,
    max_frames: int = 512,
    workers: int = 10,
    limit: int | None = None,
    max_cost: float | None = None,
    dry_run: bool = False,
) -> None:
    """Run the caption pipeline: select frames then caption via Gemini."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key and not dry_run:
        _log("ERROR: No API key. Set GEMINI_API_KEY env var or pass --api-key.")
        sys.exit(1)

    # Phase 1: Load embeddings
    index = load_embedding_index(assets_dir)
    all_scenes = index.scene_id_list

    # Load checkpoint to skip completed scenes
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILE
    checkpoint = _load_checkpoint(checkpoint_path)
    completed = set(checkpoint["completed_scenes"])

    # Filter to pending scenes
    pending = [s for s in all_scenes if s not in completed]
    if limit:
        pending = pending[:limit]

    # Count total frames to caption
    total_frames = 0
    for sid in pending:
        start, end = index._scene_ranges[sid]
        n = end - start
        total_frames += min(n, max_frames)

    # ── Measure real token cost (not estimate) ──────────────────────────

    budget = ApiBudget(
        model=model,
        rpm_limit=900,     # safety margin below 1,000
        tpm_limit=900_000, # safety margin below 1,000,000
        rpd_limit=9_500,   # safety margin below 10,000
        max_cost=max_cost,
        state_file=output_dir / "budget_state.json",
    )
    budget._total_frames = total_frames

    if not dry_run and api_key:
        # Find a sample frame to measure real prompt token count
        sample_scene = pending[0] if pending else all_scenes[0]
        sample_start, _ = index._scene_ranges[sample_scene]
        sample_fidx = index.frame_indices[sample_start]
        sample_frame = frames_dir / f"scene_{sample_scene}" / f"frame_{int(sample_fidx):04d}.jpg"

        if sample_frame.exists():
            _log("Measuring prompt token count via countTokens API...")
            sample_b64 = base64.b64encode(sample_frame.read_bytes()).decode("utf-8")
            measured = count_tokens(model, api_key, PROMPT, sample_b64)
            budget.measured_input_tokens_per_call = measured
            _log(f"  Measured input tokens: {measured:,} per call")

            # Show cost estimate before starting
            est_cost = budget.estimate_total_cost(total_frames)
            _log(f"  Estimated total cost:  ${est_cost:,.2f} "
                 f"(for {total_frames:,} frames)")
            if max_cost:
                _log(f"  Budget cap:            ${max_cost:,.2f}")
            _log("")

    _log(f"Caption Runner (Standard API)")
    _log(f"  Model:       {model}")
    _log(f"  Temperature: {temperature}")
    _log(f"  Max frames:  {max_frames}/scene")
    _log(f"  Workers:     {workers}")
    _log(f"  Scenes:      {len(pending)} pending ({len(completed)} done)")
    _log(f"  Total frames:{total_frames:,}")
    _log(f"  Dry run:     {dry_run}")
    _log("")

    jsonl_path = output_dir / "metadata.jsonl"
    total_captioned = 0
    total_errors = 0
    start_time = time.monotonic()

    try:
        for i, scene_id in enumerate(pending):
            # Select frames
            selections = select_frames_for_scene(
                index, scene_id, max_frames=max_frames, frames_dir=frames_dir,
            )
            if not selections:
                _log(f"  [{i+1}/{len(pending)}] Scene {scene_id}: no frames, skipping")
                continue

            frame_paths = [s.path for s in selections]
            n_frames = len(frame_paths)

            from stash_ai.tasks.smart_frame_selector import SmartFrameSelector
            stats = SmartFrameSelector().get_selection_stats(selections)

            _log(f"  [{i+1}/{len(pending)}] Scene {scene_id} "
                 f"({n_frames} selected, {stats['novelty_count']} novelty)")

            if dry_run:
                _log(f"    [dry-run] Would caption {n_frames} frames")
                continue

            scene_start = time.monotonic()
            scene_result = process_scene_frames(
                scene_id=str(scene_id),
                frame_paths=frame_paths,
                prompt=PROMPT,
                model=model,
                api_key=api_key,  # type: ignore[arg-type]
                temperature=temperature,
                images_dir=images_dir,
                budget=budget,
                workers=workers,
            )
            scene_time = time.monotonic() - scene_start

            total_captioned += len(scene_result.image_names)
            total_errors += scene_result.errors

            _append_metadata(
                jsonl_path, scene_id, scene_result.image_names,
                scene_result.captions, stats,
            )

            checkpoint["completed_scenes"].append(scene_id)
            checkpoint["total_frames_captioned"] = (
                checkpoint.get("total_frames_captioned", 0) + len(scene_result.image_names)
            )
            checkpoint["errors"] = checkpoint.get("errors", 0) + scene_result.errors
            _save_checkpoint(checkpoint_path, checkpoint)

            # Save budget state after each scene
            budget.save_state()

            _log(f"    {len(scene_result.image_names)} captions in {scene_time:.1f}s"
                 + (f" ({scene_result.errors} errors)" if scene_result.errors else ""))

            # Dashboard every N scenes
            if (i + 1) % DASHBOARD_INTERVAL == 0:
                _log(f"\n{budget.dashboard()}\n")

    except BudgetExhausted as e:
        _log(f"\n  STOPPED: {e}")
        _log(f"  Increase --max-cost to continue.\n")
    except DailyLimitReached as e:
        _log(f"\n  STOPPED: {e}")
        _log(f"  Resume tomorrow or switch to Batch API.\n")

    # Final dashboard
    _log(f"\n{budget.dashboard()}")

    elapsed = time.monotonic() - start_time
    _log(f"\nDone: {total_captioned:,} frames across {len(pending)} scenes in {elapsed:.0f}s")
    if total_errors:
        _log(f"  Errors: {total_errors}")

    budget.save_state()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select diverse frames then caption via Gemini VLM.",
    )
    parser.add_argument("--model", default="gemini-3.0-flash-preview",
                        help="Gemini model (default: gemini-3.0-flash-preview)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--api-key",
                        default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                        help="Gemini API key (default: GEMINI_API_KEY env var)")
    parser.add_argument("--max-frames", type=int, default=512,
                        help="Max frames per scene via SmartFrameSelector (default: 512)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent API calls per scene (default: 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max scenes to process (default: all)")
    parser.add_argument("--max-cost", type=float, default=None,
                        help="Budget cap in USD — stops when reached (default: no cap)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview frame selection without API calls")
    args = parser.parse_args()

    run(
        model=args.model,
        temperature=args.temperature,
        api_key=args.api_key,
        max_frames=args.max_frames,
        workers=args.workers,
        limit=args.limit,
        max_cost=args.max_cost,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
```

**Step 5: Run all tests**

Run: `cd ~/.stash/plugins/stash-copilot && uv run pytest tests/tools/test_gemini_api.py tests/tools/test_caption_runner.py -v`
Expected: All tests PASS.

**Step 6: Commit**

```bash
git add tools/dataset/gemini_api.py tools/dataset/caption_runner.py \
    tests/tools/test_gemini_api.py tests/tools/test_caption_runner.py
git commit -m "feat(dataset): add caption_runner with budget-gated API calls"
```

---

### Task 4: Batch API Support (Recommended Path)

**Files:**
- Create: `tools/dataset/caption_batch.py`
- Test: `tests/tools/test_caption_batch.py`

The Batch API eliminates rate limits and costs 50%. It generates JSONL request files,
uploads via Google's File API, submits batch jobs, and polls for results.

**Key difference from Standard API:** No rate limiting needed (Google handles it), but
cost estimation is critical before submitting ~$1,270 worth of batch jobs.

> **Note:** Each JSONL line includes base64-encoded JPEG data (~140KB per frame).
> At 2.42M frames, total JSONL size is ~340 GB. Split into 2 GB chunks = ~170 batch files.
> Each batch completes within 24h. Submit in parallel batches.

**Step 1: Write tests**

Create `tests/tools/test_caption_batch.py`:

```python
"""Tests for tools.dataset.caption_batch — Gemini Batch API captioning."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.dataset.caption_batch import (
    build_batch_request,
    write_batch_jsonl,
    parse_batch_results,
    estimate_batch_cost,
)
from tools.dataset.api_budget import PRICING


def test_build_batch_request() -> None:
    req = build_batch_request(
        key="scene_42_frame_0010",
        frame_b64="AAAA",
        prompt="describe this",
        model="gemini-3.0-flash-preview",
        temperature=1.0,
    )
    assert req["key"] == "scene_42_frame_0010"
    inner = req["request"]
    parts = inner["contents"][0]["parts"]
    assert parts[0]["inlineData"]["data"] == "AAAA"
    assert parts[1]["text"] == "describe this"
    assert inner["generationConfig"]["temperature"] == 1.0


def test_write_batch_jsonl_splits_by_size(tmp_path: Path) -> None:
    requests = [
        build_batch_request(f"key_{i}", "A" * 1000, "prompt", "model", 1.0)
        for i in range(100)
    ]
    # Use tiny max_bytes to force splitting
    files = write_batch_jsonl(requests, tmp_path / "batch", max_bytes=5000)
    assert len(files) > 1
    # Verify each file is valid JSONL
    for f in files:
        for line in f.read_text().strip().splitlines():
            parsed = json.loads(line)
            assert "key" in parsed
            assert "request" in parsed


def test_parse_batch_results() -> None:
    lines = [
        json.dumps({
            "key": "scene_1_frame_0001",
            "response": {
                "candidates": [{"content": {"parts": [{"text": "A blowjob scene."}]},
                                 "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 1500, "candidatesTokenCount": 80,
                    "totalTokenCount": 1580,
                },
            },
        }),
        json.dumps({
            "key": "scene_1_frame_0002",
            "response": {
                "candidates": [{"content": {"parts": [{"text": "Cowgirl position."}]},
                                 "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 1500, "candidatesTokenCount": 60,
                    "totalTokenCount": 1560,
                },
            },
        }),
    ]
    results, total_cost = parse_batch_results(
        "\n".join(lines), model="gemini-3.0-flash-preview",
    )
    assert len(results) == 2
    assert results["scene_1_frame_0001"] == "A blowjob scene."
    assert results["scene_1_frame_0002"] == "Cowgirl position."
    assert total_cost > 0  # actual cost from real token counts


def test_parse_batch_results_handles_errors() -> None:
    lines = [
        json.dumps({
            "key": "scene_1_frame_0001",
            "error": {"code": 400, "message": "blocked"},
        }),
    ]
    results, total_cost = parse_batch_results(
        "\n".join(lines), model="gemini-3.0-flash-preview",
    )
    assert "ERROR" in results["scene_1_frame_0001"]
    assert total_cost == 0  # no cost for errors


def test_estimate_batch_cost_uses_measured_tokens() -> None:
    cost = estimate_batch_cost(
        model="gemini-3.0-flash-preview",
        n_frames=100,
        measured_input_tokens=1500,
        avg_output_tokens=80,
    )
    # 100 × (1500 × $0.25/1M + 80 × $1.50/1M) = 100 × $0.000495 = $0.0495
    assert 0.04 < cost < 0.06
```

**Step 2: Write the implementation**

Create `tools/dataset/caption_batch.py`:

```python
"""Gemini Batch API captioning — generates JSONL, tracks cost from real tokens.

Cost tracking:
- Pre-submission: estimate_batch_cost() uses measured input tokens (from countTokens)
  + average output tokens to project cost before uploading.
- Post-completion: parse_batch_results() reads actual usageMetadata from each response
  line and computes exact cost. No estimates in the final tally.

Workflow:
    uv run python tools/dataset/caption_batch.py generate --limit 100
    uv run python tools/dataset/caption_batch.py submit
    uv run python tools/dataset/caption_batch.py collect
    uv run python tools/dataset/caption_batch.py apply
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tools.dataset.gemini_api import parse_response
from tools.dataset.api_budget import PRICING, GeminiPricing, compute_cost

# 2 GB max per batch file (Google limit)
MAX_BATCH_BYTES = 2 * 1024 * 1024 * 1024


def build_batch_request(
    key: str,
    frame_b64: str,
    prompt: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    """Build a single Batch API request line."""
    return {
        "key": key,
        "request": {
            "model": f"models/{model}",
            "contents": [{"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
                {"text": prompt},
            ]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 4096,
            },
        },
    }


def write_batch_jsonl(
    requests: list[dict[str, Any]],
    output_prefix: Path,
    max_bytes: int = MAX_BATCH_BYTES,
) -> list[Path]:
    """Write batch requests to JSONL files, splitting at max_bytes.

    Returns list of JSONL file paths created.
    """
    files: list[Path] = []
    current_size = 0
    current_lines: list[str] = []
    file_idx = 0

    for req in requests:
        line = json.dumps(req, separators=(",", ":"))
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline

        if current_size + line_bytes > max_bytes and current_lines:
            path = Path(f"{output_prefix}_{file_idx:04d}.jsonl")
            path.write_text("\n".join(current_lines) + "\n", encoding="utf-8")
            files.append(path)
            file_idx += 1
            current_lines = []
            current_size = 0

        current_lines.append(line)
        current_size += line_bytes

    if current_lines:
        path = Path(f"{output_prefix}_{file_idx:04d}.jsonl")
        path.write_text("\n".join(current_lines) + "\n", encoding="utf-8")
        files.append(path)

    return files


def estimate_batch_cost(
    model: str,
    n_frames: int,
    measured_input_tokens: int,
    avg_output_tokens: int = 100,
) -> float:
    """Pre-submission cost estimate using measured token counts.

    Args:
        model: Gemini model name (batch pricing used automatically).
        n_frames: Number of frames in the batch.
        measured_input_tokens: From countTokens API (exact).
        avg_output_tokens: Average output tokens (from prior runs or default).

    Returns:
        Estimated total cost in USD.
    """
    batch_key = f"{model}:batch"
    pricing = PRICING.get(batch_key, PRICING.get(model))
    if pricing is None:
        raise ValueError(f"No pricing for model: {model}")
    return n_frames * compute_cost(pricing, measured_input_tokens, avg_output_tokens)


def parse_batch_results(
    jsonl_text: str,
    model: str,
) -> tuple[dict[str, str], float]:
    """Parse batch API response JSONL into key→caption mapping + actual cost.

    Reads usageMetadata from each response line to compute exact cost.

    Returns:
        (results dict, total_cost in USD)
    """
    batch_key = f"{model}:batch"
    pricing = PRICING.get(batch_key, PRICING.get(model))
    if pricing is None:
        raise ValueError(f"No pricing for model: {model}")

    results: dict[str, str] = {}
    total_cost = 0.0

    for line in jsonl_text.strip().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        key = entry.get("key", "unknown")

        if "error" in entry:
            err = entry["error"]
            results[key] = f"[ERROR: {err.get('message', str(err))}]"
            continue

        response = entry.get("response", {})
        try:
            result = parse_response(response)
            results[key] = result.caption

            # Actual cost from real token counts in this response
            total_cost += compute_cost(
                pricing, result.input_tokens, result.output_tokens,
            )
        except RuntimeError as e:
            results[key] = f"[ERROR: {e}]"

    return results, total_cost


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
```

> **Note:** The `submit` and `collect` subcommands (which use Google's `genai` Python SDK)
> are left for the implementation phase since they require the `google-genai` package
> and real API interaction. The core logic (JSONL generation + result parsing) is testable now.

**Step 3: Run tests**

Run: `cd ~/.stash/plugins/stash-copilot && uv run pytest tests/tools/test_caption_batch.py -v`
Expected: All 5 tests PASS.

**Step 4: Commit**

```bash
git add tools/dataset/caption_batch.py tests/tools/test_caption_batch.py
git commit -m "feat(dataset): add caption_batch with measured cost tracking"
```

---

### Task 5: Integration Test — Dry Run + Live Test

**Step 1: Dry-run frame selection across all scenes**

```bash
cd ~/.stash/plugins/stash-copilot && \
uv run python -m tools.dataset.caption_runner --dry-run --limit 20 --max-frames 512
```

Expected: Shows 20 scenes with frame counts and novelty stats. No API calls.

**Step 2: Live test with 1 scene + budget cap (Standard API)**

```bash
cd ~/.stash/plugins/stash-copilot && \
uv run python -m tools.dataset.caption_runner --limit 1 --max-frames 512 --workers 5 --max-cost 1.00
```

Verify:
- "Measuring prompt token count via countTokens API..." appears with real number
- "Estimated total cost: $X.XX" appears before captioning starts
- Caption `.txt` files appear in `assets/lora_dataset/images/`
- `budget_state.json` contains actual token counts and cost
- `caption_progress.json` shows the scene in `completed_scenes`
- `metadata.jsonl` has a record with `selection` stats
- Dashboard shows real RPM/TPM/RPD/cost numbers

**Step 3: Verify budget cap stops execution**

```bash
cd ~/.stash/plugins/stash-copilot && \
uv run python -m tools.dataset.caption_runner --limit 100 --max-frames 512 --workers 5 --max-cost 0.01
```

Expected: Stops after a few frames with "STOPPED: Budget exhausted: $X.XX >= $0.01 cap"

**Step 4: Generate batch JSONL for 10 scenes (Batch API test)**

```bash
cd ~/.stash/plugins/stash-copilot && \
uv run python -m tools.dataset.caption_batch generate --limit 10 --max-frames 512
```

Verify: JSONL files created, each line is valid JSON. Estimated cost printed before generation.

**Step 5: Commit any fixes**

```bash
git add -A && git commit -m "fix(dataset): integration fixes from live caption test"
```

---

### Task 6: Production Run (Batch API)

**Not a code task — this is the production run.**

```bash
# 1. Generate all batch JSONL files
cd ~/.stash/plugins/stash-copilot && \
nohup uv run python -m tools.dataset.caption_batch generate --max-frames 512 \
  > /tmp/caption_batch_generate.log 2>&1 &

# 2. Submit batch jobs (after JSONL generation completes)
uv run python -m tools.dataset.caption_batch submit

# 3. Poll and collect results (run periodically or leave running)
uv run python -m tools.dataset.caption_batch collect

# 4. Apply results to .txt files and print actual cost
uv run python -m tools.dataset.caption_batch apply
```

The `apply` step reads `usageMetadata` from every batch response line and prints
the **actual total cost** — not an estimate.

Expected: ~1-3 days for all batches to complete. Actual cost will be printed from
real token counts after collection.
