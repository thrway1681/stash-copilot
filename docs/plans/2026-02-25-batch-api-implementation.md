# Gemini Batch API Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Gemini Batch API as an alternative to real-time captioning, with 50% cost reduction and dashboard integration for monitoring/auto-collection.

**Architecture:** New `tools/dataset/batch_api.py` module handles the full batch lifecycle (prepare JSONL chunks, upload via File API, submit jobs, poll status, collect results). Dashboard gets new `/api/batch/*` endpoints and a Batch Jobs UI section with auto-polling and auto-collection.

**Tech Stack:** Python 3.12+, `requests` (existing), Gemini REST API (`generativelanguage.googleapis.com/v1beta`), existing `frame_selector.py` + `constants.py` + `io_utils.py`

---

## Constants and Configuration

Throughout this plan, use these values:

```python
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "lora_dataset"
BATCH_JOBS_DIR = DEFAULT_OUTPUT_DIR / "batch_jobs"
BATCH_STATE_FILE = "batch_state.json"
CHUNK_SIZE = 25_000  # frames per JSONL chunk
```

Safety settings (same as `gemini_api.py:98-103`):
```python
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
```

---

## Task 1: Batch API Type Definitions and State Management

**Files:**
- Create: `tools/dataset/batch_api.py`
- Test: `tests/tools/test_batch_api.py`

**Step 1: Write the failing test**

```python
# tests/tools/test_batch_api.py
"""Tests for batch API module."""
import json
from pathlib import Path

from tools.dataset.batch_api import BatchJob, CollectStats, load_batch_state, save_batch_state


def test_batch_job_round_trip(tmp_path: Path) -> None:
    """BatchJob serializes to/from JSON via state file."""
    state_file = tmp_path / "batch_state.json"
    job = BatchJob(
        name="batches/abc123",
        display_name="chunk-001",
        state="JOB_STATE_PENDING",
        submitted_at="2026-02-25T10:00:00Z",
        frame_count=25000,
        scene_ids=[1, 2, 3],
        file_name="files/xyz",
        result_file=None,
        collected=False,
        stats=None,
    )
    save_batch_state(state_file, [job])
    loaded = load_batch_state(state_file)
    assert len(loaded) == 1
    assert loaded[0].name == "batches/abc123"
    assert loaded[0].frame_count == 25000
    assert loaded[0].collected is False


def test_batch_state_empty_file(tmp_path: Path) -> None:
    """load_batch_state returns empty list for missing file."""
    state_file = tmp_path / "batch_state.json"
    assert load_batch_state(state_file) == []


def test_collect_stats_serialization() -> None:
    """CollectStats round-trips through dict."""
    stats = CollectStats(captions_written=100, errors=5, scenes_completed=10)
    d = stats.to_dict()
    loaded = CollectStats.from_dict(d)
    assert loaded.captions_written == 100
    assert loaded.errors == 5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_batch_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.dataset.batch_api'`

**Step 3: Write minimal implementation**

```python
# tools/dataset/batch_api.py
"""Gemini Batch API integration for caption pipeline.

Provides 50% cost reduction over real-time captioning by submitting
frames as batch jobs. Each job processes a JSONL chunk of up to 25K
frames with a target turnaround of 24 hours.

Lifecycle: prepare_chunks → submit_job → poll_jobs → collect_results
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Types ────────────────────────────────────────────────────────────────


@dataclass
class CollectStats:
    """Statistics from collecting a completed batch job's results."""
    captions_written: int = 0
    errors: int = 0
    scenes_completed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "captions_written": self.captions_written,
            "errors": self.errors,
            "scenes_completed": self.scenes_completed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CollectStats:
        return cls(
            captions_written=d.get("captions_written", 0),
            errors=d.get("errors", 0),
            scenes_completed=d.get("scenes_completed", 0),
        )


@dataclass
class BatchJob:
    """Metadata for a single batch job (one JSONL chunk)."""
    name: str                          # e.g. "batches/abc123"
    display_name: str                  # e.g. "chunk-001"
    state: str                         # JOB_STATE_*
    submitted_at: str                  # ISO timestamp
    frame_count: int
    scene_ids: list[int] = field(default_factory=list)
    file_name: str | None = None       # uploaded JSONL file ref
    result_file: str | None = None     # result file ref (when succeeded)
    collected: bool = False
    stats: CollectStats | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "state": self.state,
            "submitted_at": self.submitted_at,
            "frame_count": self.frame_count,
            "scene_ids": self.scene_ids,
            "file_name": self.file_name,
            "result_file": self.result_file,
            "collected": self.collected,
            "stats": self.stats.to_dict() if self.stats else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BatchJob:
        stats = CollectStats.from_dict(d["stats"]) if d.get("stats") else None
        return cls(
            name=d["name"],
            display_name=d["display_name"],
            state=d["state"],
            submitted_at=d["submitted_at"],
            frame_count=d["frame_count"],
            scene_ids=d.get("scene_ids", []),
            file_name=d.get("file_name"),
            result_file=d.get("result_file"),
            collected=d.get("collected", False),
            stats=stats,
        )


# ── State persistence ────────────────────────────────────────────────────


def load_batch_state(state_file: Path) -> list[BatchJob]:
    """Load batch jobs from state file. Returns empty list if missing."""
    if not state_file.exists():
        return []
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return [BatchJob.from_dict(j) for j in data.get("jobs", [])]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def save_batch_state(state_file: Path, jobs: list[BatchJob]) -> None:
    """Persist batch jobs to state file."""
    data = {
        "jobs": [j.to_dict() for j in jobs],
        "total_submitted": sum(j.frame_count for j in jobs),
        "total_collected": sum(
            j.stats.captions_written for j in jobs if j.stats
        ),
        "total_errors": sum(j.stats.errors for j in jobs if j.stats),
    }
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_batch_api.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add tools/dataset/batch_api.py tests/tools/test_batch_api.py
git commit -m "feat(batch): add type definitions and state management"
```

---

## Task 2: JSONL Chunk Preparation

**Files:**
- Modify: `tools/dataset/batch_api.py`
- Test: `tests/tools/test_batch_api.py`

**Step 1: Write the failing test**

```python
# Add to tests/tools/test_batch_api.py
import base64
from unittest.mock import MagicMock, patch

from tools.dataset.batch_api import prepare_batch_chunks


def test_prepare_batch_chunks_creates_jsonl(tmp_path: Path) -> None:
    """prepare_batch_chunks creates JSONL files with correct structure."""
    # Setup: mock frame selection to return 2 frames for 1 scene
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    scene_dir = frames_dir / "scene_1"
    scene_dir.mkdir()
    batch_dir = tmp_path / "batch_jobs"

    # Create dummy frames
    for i in range(3):
        frame = scene_dir / f"frame_{i+1:04d}.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)  # minimal JPEG header

    # Mock index and frame selector
    mock_selection = MagicMock()
    mock_selection.path = str(scene_dir / "frame_0001.jpg")

    mock_selection2 = MagicMock()
    mock_selection2.path = str(scene_dir / "frame_0002.jpg")

    with patch("tools.dataset.batch_api.select_frames_for_scene") as mock_select, \
         patch("tools.dataset.batch_api.dataset_image_name") as mock_name:
        mock_select.return_value = [mock_selection, mock_selection2]
        mock_name.side_effect = lambda sid, fp: f"s{sid}_f{Path(fp).stem.replace('frame_', '')}.jpg"

        chunks = prepare_batch_chunks(
            index=MagicMock(scene_id_list=[1]),
            completed_scenes=set(),
            images_dir=images_dir,
            frames_dir=frames_dir,
            batch_dir=batch_dir,
            max_frames=20,
            chunk_size=100,
        )

    assert len(chunks) == 1
    assert chunks[0].exists()

    # Verify JSONL structure
    lines = chunks[0].read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "key" in record
        assert "request" in record
        assert "contents" in record["request"]
        parts = record["request"]["contents"][0]["parts"]
        assert parts[0]["inlineData"]["mimeType"] == "image/jpeg"
        assert "text" in parts[1]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_batch_api.py::test_prepare_batch_chunks_creates_jsonl -v`
Expected: FAIL with `ImportError: cannot import name 'prepare_batch_chunks'`

**Step 3: Write implementation**

Add to `tools/dataset/batch_api.py`:

```python
import base64
from datetime import UTC, datetime

from tools.dataset.constants import CAPTION_PROMPT
from tools.dataset.frame_selector import EmbeddingIndex, select_frames_for_scene
from tools.dataset.io_utils import dataset_image_name

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta"

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "lora_dataset"
CHUNK_SIZE = 25_000


def prepare_batch_chunks(
    index: EmbeddingIndex,
    completed_scenes: set[int],
    images_dir: Path,
    frames_dir: Path,
    batch_dir: Path | None = None,
    max_frames: int = 20,
    chunk_size: int = CHUNK_SIZE,
    temperature: float = 1.0,
) -> list[Path]:
    """Select frames for pending scenes and build JSONL chunks.

    Each JSONL line:
    {"key": "s{sid}_f{idx}", "request": {GenerateContentRequest with base64 image}}

    Returns list of .jsonl file paths, each containing up to chunk_size requests.
    """
    if batch_dir is None:
        batch_dir = DEFAULT_OUTPUT_DIR / "batch_jobs"
    batch_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = index.scene_id_list
    pending = [s for s in all_scenes if s not in completed_scenes]

    # Collect all (scene_id, frame_path) pairs, skipping already-captioned
    frame_pairs: list[tuple[int, Path]] = []
    for scene_id in pending:
        selections = select_frames_for_scene(
            index, scene_id, max_frames=max_frames, frames_dir=frames_dir,
        )
        for sel in selections:
            img_name = dataset_image_name(str(scene_id), Path(sel.path))
            txt_path = images_dir / img_name.replace(".jpg", ".txt")
            # Skip frames that already have a non-error caption
            if txt_path.exists():
                content = txt_path.read_text(encoding="utf-8")
                if not content.startswith("[ERROR"):
                    continue
            frame_pairs.append((scene_id, Path(sel.path)))

    if not frame_pairs:
        return []

    # Build JSONL chunks
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    chunk_paths: list[Path] = []

    for chunk_idx in range(0, len(frame_pairs), chunk_size):
        chunk = frame_pairs[chunk_idx : chunk_idx + chunk_size]
        chunk_path = batch_dir / f"chunk-{timestamp}-{chunk_idx // chunk_size + 1:03d}.jsonl"

        with open(chunk_path, "w", encoding="utf-8") as f:
            for scene_id, frame_path in chunk:
                img_name = dataset_image_name(str(scene_id), frame_path)
                key = img_name.replace(".jpg", "")  # e.g. "s123_f0015"

                # Read and encode frame
                frame_b64 = base64.b64encode(frame_path.read_bytes()).decode("utf-8")

                request = {
                    "contents": [{"parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}},
                        {"text": CAPTION_PROMPT},
                    ]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": 4096,
                    },
                    "safetySettings": SAFETY_SETTINGS,
                }
                line = json.dumps({"key": key, "request": request})
                f.write(line + "\n")

        chunk_paths.append(chunk_path)

    return chunk_paths
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tools/test_batch_api.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add tools/dataset/batch_api.py tests/tools/test_batch_api.py
git commit -m "feat(batch): add JSONL chunk preparation from frame selection"
```

---

## Task 3: File Upload and Job Submission

**Files:**
- Modify: `tools/dataset/batch_api.py`
- Test: `tests/tools/test_batch_api.py`

**Step 1: Write the failing test**

```python
# Add to tests/tools/test_batch_api.py
from unittest.mock import patch, MagicMock
from tools.dataset.batch_api import upload_jsonl, submit_batch_job


def test_upload_jsonl_calls_file_api(tmp_path: Path) -> None:
    """upload_jsonl uploads file via resumable upload protocol."""
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text('{"key":"r1","request":{}}\n')

    with patch("tools.dataset.batch_api.requests") as mock_req:
        # Mock initiate upload (returns upload URL in headers)
        init_resp = MagicMock()
        init_resp.status_code = 200
        init_resp.headers = {"X-Goog-Upload-URL": "https://upload.example.com/resume"}

        # Mock actual upload (returns file metadata)
        upload_resp = MagicMock()
        upload_resp.status_code = 200
        upload_resp.json.return_value = {"file": {"name": "files/abc123"}}

        mock_req.post.side_effect = [init_resp, upload_resp]

        file_name = upload_jsonl(jsonl, "test-api-key")

    assert file_name == "files/abc123"
    assert mock_req.post.call_count == 2


def test_submit_batch_job_calls_api(tmp_path: Path) -> None:
    """submit_batch_job POSTs to batchGenerateContent."""
    with patch("tools.dataset.batch_api.requests") as mock_req:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "name": "batches/xyz789",
            "metadata": {"state": "JOB_STATE_PENDING"},
        }
        mock_req.post.return_value = resp

        job = submit_batch_job(
            file_name="files/abc123",
            model="gemini-3-flash-preview",
            api_key="test-key",
            display_name="chunk-001",
            frame_count=25000,
            scene_ids=[1, 2, 3],
        )

    assert job.name == "batches/xyz789"
    assert job.state == "JOB_STATE_PENDING"
    assert job.frame_count == 25000
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_batch_api.py::test_upload_jsonl_calls_file_api -v`
Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Add to `tools/dataset/batch_api.py`:

```python
import requests as _requests  # avoid shadowing with local vars


def upload_jsonl(jsonl_path: Path, api_key: str) -> str:
    """Upload a JSONL file via Gemini File API (resumable upload).

    Returns the file name (e.g. "files/abc123") for use in batch submission.
    """
    file_size = jsonl_path.stat().st_size
    display_name = jsonl_path.stem

    # Step 1: Initiate resumable upload
    init_resp = _requests.post(
        f"{UPLOAD_BASE}/files",
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": display_name}},
        timeout=60,
    )
    init_resp.raise_for_status()
    upload_url = init_resp.headers["X-Goog-Upload-URL"]

    # Step 2: Upload file bytes
    with open(jsonl_path, "rb") as f:
        data = f.read()

    upload_resp = _requests.post(
        upload_url,
        headers={
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        data=data,
        timeout=300,
    )
    upload_resp.raise_for_status()

    result = upload_resp.json()
    return result["file"]["name"]


def submit_batch_job(
    file_name: str,
    model: str,
    api_key: str,
    display_name: str,
    frame_count: int,
    scene_ids: list[int],
) -> BatchJob:
    """Submit a batch job for a previously uploaded JSONL file.

    Returns BatchJob with initial state (typically JOB_STATE_PENDING).
    """
    url = f"{GEMINI_API_BASE}/models/{model}:batchGenerateContent"
    payload = {
        "batch": {
            "display_name": display_name,
            "input_config": {"file_name": file_name},
        }
    }

    resp = _requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    return BatchJob(
        name=data["name"],
        display_name=display_name,
        state=data.get("metadata", {}).get("state", "JOB_STATE_PENDING"),
        submitted_at=datetime.now(UTC).isoformat(),
        frame_count=frame_count,
        scene_ids=scene_ids,
        file_name=file_name,
        result_file=None,
        collected=False,
        stats=None,
    )
```

**Step 4: Run tests**

Run: `uv run pytest tests/tools/test_batch_api.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add tools/dataset/batch_api.py tests/tools/test_batch_api.py
git commit -m "feat(batch): add JSONL upload and job submission via Gemini File API"
```

---

## Task 4: Job Polling and Result Collection

**Files:**
- Modify: `tools/dataset/batch_api.py`
- Test: `tests/tools/test_batch_api.py`

**Step 1: Write the failing test**

```python
# Add to tests/tools/test_batch_api.py
from tools.dataset.batch_api import poll_batch_jobs, collect_batch_results


def test_poll_updates_job_state() -> None:
    """poll_batch_jobs updates state from API response."""
    job = BatchJob(
        name="batches/abc", display_name="chunk-001",
        state="JOB_STATE_PENDING", submitted_at="",
        frame_count=10, scene_ids=[1],
    )

    with patch("tools.dataset.batch_api._requests") as mock_req:
        resp = MagicMock()
        resp.json.return_value = {
            "name": "batches/abc",
            "metadata": {"state": "JOB_STATE_SUCCEEDED"},
            "response": {"responsesFile": "files/result123"},
        }
        mock_req.get.return_value = resp

        updated = poll_batch_jobs([job], "test-key")

    assert updated[0].state == "JOB_STATE_SUCCEEDED"
    assert updated[0].result_file == "files/result123"


def test_collect_writes_txt_files(tmp_path: Path) -> None:
    """collect_batch_results parses response JSONL and writes .txt files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create a fake result JSONL
    result_content = (
        '{"key": "s1_f0001", "response": {"candidates": [{"content": {"parts": [{"text": "A test caption."}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}}}\n'
        '{"key": "s1_f0002", "response": {"candidates": [{"content": {"parts": [{"text": "Another caption."}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}}}\n'
    )

    job = BatchJob(
        name="batches/abc", display_name="chunk-001",
        state="JOB_STATE_SUCCEEDED", submitted_at="",
        frame_count=2, scene_ids=[1],
        result_file="files/result123",
    )

    with patch("tools.dataset.batch_api._requests") as mock_req:
        resp = MagicMock()
        resp.content = result_content.encode("utf-8")
        mock_req.get.return_value = resp

        result = collect_batch_results(job, images_dir, "test-key")

    assert result.stats.captions_written == 2
    assert result.stats.errors == 0
    assert (images_dir / "s1_f0001.txt").read_text() == "A test caption."
    assert (images_dir / "s1_f0002.txt").read_text() == "Another caption."
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_batch_api.py::test_poll_updates_job_state -v`
Expected: FAIL with `ImportError`

**Step 3: Write implementation**

Add to `tools/dataset/batch_api.py`:

```python
def poll_batch_jobs(jobs: list[BatchJob], api_key: str) -> list[BatchJob]:
    """Check status of all active batch jobs.

    Updates state and result_file for each job. Does not modify
    already-collected jobs.
    """
    for job in jobs:
        if job.collected:
            continue
        if job.state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                         "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            # Already in terminal state but not yet collected
            continue

        try:
            resp = _requests.get(
                f"{GEMINI_API_BASE}/{job.name}",
                headers={"x-goog-api-key": api_key},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            job.state = data.get("metadata", {}).get("state", job.state)

            # Extract result file when succeeded
            response = data.get("response", {})
            if response.get("responsesFile"):
                job.result_file = response["responsesFile"]
            # Also check dest.file_name (SDK format)
            dest = data.get("dest", {})
            if dest.get("file_name"):
                job.result_file = dest["file_name"]

        except Exception:
            pass  # Keep existing state on poll failure

    return jobs


def collect_batch_results(
    job: BatchJob,
    images_dir: Path,
    api_key: str,
) -> BatchJob:
    """Download result JSONL, parse responses, write .txt caption files.

    Updates job.collected and job.stats. Returns the updated job.
    """
    if not job.result_file:
        raise ValueError(f"Job {job.name} has no result file")

    # Download result file
    download_url = (
        f"https://generativelanguage.googleapis.com/download/v1beta"
        f"/{job.result_file}:download"
    )
    resp = _requests.get(
        download_url,
        params={"alt": "media", "key": api_key},
        timeout=300,
    )
    resp.raise_for_status()
    content = resp.content.decode("utf-8")

    captions_written = 0
    errors = 0
    scenes_seen: set[int] = set()

    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue

        key = record.get("key", "")
        response = record.get("response")
        error = record.get("error")

        txt_path = images_dir / f"{key}.txt"

        # Parse scene_id from key (e.g. "s123_f0015" → 123)
        try:
            sid = int(key.split("_f")[0][1:])
            scenes_seen.add(sid)
        except (ValueError, IndexError):
            pass

        if error:
            txt_path.write_text(f"[ERROR: {error}]", encoding="utf-8")
            errors += 1
            continue

        if not response:
            txt_path.write_text("[ERROR: empty response]", encoding="utf-8")
            errors += 1
            continue

        # Parse GenerateContentResponse (same structure as real-time)
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                txt_path.write_text("[ERROR: no candidates]", encoding="utf-8")
                errors += 1
                continue

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts or "text" not in parts[0]:
                txt_path.write_text("[ERROR: no text in response]", encoding="utf-8")
                errors += 1
                continue

            caption = parts[0]["text"].strip()
            if not caption:
                txt_path.write_text("[ERROR: empty caption]", encoding="utf-8")
                errors += 1
                continue

            txt_path.write_text(caption, encoding="utf-8")
            captions_written += 1

        except Exception as e:
            txt_path.write_text(f"[ERROR: {e}]", encoding="utf-8")
            errors += 1

    job.collected = True
    job.stats = CollectStats(
        captions_written=captions_written,
        errors=errors,
        scenes_completed=len(scenes_seen),
    )

    return job
```

**Step 4: Run tests**

Run: `uv run pytest tests/tools/test_batch_api.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add tools/dataset/batch_api.py tests/tools/test_batch_api.py
git commit -m "feat(batch): add job polling and result collection with .txt writing"
```

---

## Task 5: Dashboard Backend — Batch API Endpoints

**Files:**
- Modify: `tools/dataset/caption_dashboard.py`

**Step 1: Add batch manager class and imports**

Add near the top of `caption_dashboard.py` (after existing imports):

```python
from tools.dataset.batch_api import (
    BatchJob,
    load_batch_state,
    save_batch_state,
    prepare_batch_chunks,
    upload_jsonl,
    submit_batch_job,
    poll_batch_jobs,
    collect_batch_results,
)
```

Add a batch manager singleton pattern (similar to `runner_manager`):

```python
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
    ) -> list[BatchJob]:
        """Prepare chunks, upload, and submit batch jobs for all pending frames."""
        from tools.dataset.frame_selector import load_embedding_index

        index = load_embedding_index(DEFAULT_ASSETS_DIR)
        progress = _read_json(DEFAULT_OUTPUT_DIR / "caption_progress.json")
        completed = set(progress.get("completed_scenes", []))

        chunk_paths = prepare_batch_chunks(
            index=index,
            completed_scenes=completed,
            images_dir=DEFAULT_OUTPUT_DIR / "images",
            frames_dir=DEFAULT_ASSETS_DIR.parent / "embedded_frames",
            max_frames=max_frames,
            temperature=temperature,
        )

        new_jobs: list[BatchJob] = []
        for i, chunk_path in enumerate(chunk_paths):
            # Count frames and scenes in chunk
            frame_count = sum(1 for _ in open(chunk_path))
            scene_ids = set()
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
        """Poll all active jobs and auto-collect succeeded ones."""
        with self._lock:
            self._jobs = poll_batch_jobs(self._jobs, api_key)

            # Auto-collect succeeded jobs
            for job in self._jobs:
                if job.state == "JOB_STATE_SUCCEEDED" and not job.collected:
                    try:
                        collect_batch_results(
                            job,
                            DEFAULT_OUTPUT_DIR / "images",
                            api_key,
                        )
                    except Exception:
                        pass  # Will retry on next poll

            save_batch_state(self._state_file, self._jobs)
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
                    j.state in ("JOB_STATE_PENDING", "JOB_STATE_RUNNING")
                    for j in self._jobs
                ),
            }


batch_manager = BatchManager()
```

**Step 2: Add API endpoints to do_GET**

In `DashboardHandler.do_GET`, before the final `else: self.send_error(404)`:

```python
        elif path == "/api/batch/status":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                batch_manager.poll(api_key)
            self._send_json(batch_manager.status())
```

**Step 3: Add API endpoints to do_POST**

In `DashboardHandler.do_POST`, before the final `else: self.send_error(404)`:

```python
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
                _requests.post(
                    f"{GEMINI_API_BASE}/{job_name}:cancel",
                    headers={"x-goog-api-key": api_key},
                    timeout=30,
                )
                self._send_json({"status": "cancelled", "job": job_name})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
```

**Step 4: Test manually via curl**

```bash
# Start dashboard
uv run python tools/dataset/caption_dashboard.py &

# Check batch status (should be empty)
curl -s http://localhost:8766/api/batch/status | python -m json.tool
```

Expected: `{"jobs": [], "total_submitted": 0, "total_collected": 0, ...}`

**Step 5: Commit**

```bash
git add tools/dataset/caption_dashboard.py
git commit -m "feat(dashboard): add batch API endpoints for submit, status, and cancel"
```

---

## Task 6: Dashboard Frontend — Batch Jobs UI

**Files:**
- Modify: `tools/dataset/caption_dashboard.html`

**Step 1: Add Batch Jobs section to HTML**

In the Pipeline Controls area (after the existing launch controls), add a new section.
Find the closing `</div>` of the `.pipeline-controls` section and add before it:

```html
<!-- Batch Jobs Section -->
<div class="batch-section" id="batchSection">
  <h3>Batch Jobs <span class="batch-discount">50% off</span></h3>
  <div class="batch-controls">
    <button class="btn btn-batch" id="btnSubmitBatch" onclick="submitBatch()">
      Submit Batch
    </button>
    <span class="batch-status-summary" id="batchSummary"></span>
  </div>
  <div class="batch-jobs-list" id="batchJobsList"></div>
</div>
```

**Step 2: Add CSS for batch section**

Add to the `<style>` section:

```css
.batch-section {
  margin-top: 1.5rem;
  padding: 1rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface-1);
}
.batch-section h3 {
  margin: 0 0 0.75rem;
  font-size: 0.95rem;
  color: var(--text-1);
}
.batch-discount {
  font-size: 0.7rem;
  padding: 2px 8px;
  border-radius: 20px;
  background: rgba(16, 185, 129, 0.15);
  color: #10b981;
  font-weight: 600;
}
.batch-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.75rem;
}
.btn-batch {
  background: linear-gradient(135deg, #10b981, #059669);
  color: white;
  border: none;
  padding: 0.5rem 1.2rem;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.85rem;
  transition: opacity 0.2s;
}
.btn-batch:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-batch:hover:not(:disabled) {
  opacity: 0.85;
}
.batch-status-summary {
  font-size: 0.8rem;
  color: var(--text-2);
}
.batch-jobs-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.batch-job-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.6rem 0.85rem;
  border-radius: 8px;
  background: var(--surface-2);
  font-size: 0.8rem;
}
.batch-job-card .job-name { color: var(--text-1); font-weight: 500; }
.batch-job-card .job-frames { color: var(--text-2); }
.batch-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: 600;
}
.batch-badge.pending { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
.batch-badge.running { background: rgba(59, 130, 246, 0.15); color: #3b82f6; animation: pulse 2s infinite; }
.batch-badge.succeeded { background: rgba(16, 185, 129, 0.15); color: #10b981; }
.batch-badge.failed { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
.batch-badge.collected { background: rgba(16, 185, 129, 0.25); color: #10b981; }
```

**Step 3: Add JavaScript for batch management**

Add to the `<script>` section:

```javascript
// ── Batch API ──────────────────────────────

async function submitBatch() {
  const btn = $('#btnSubmitBatch');
  btn.disabled = true;
  btn.textContent = 'Preparing...';

  try {
    const resp = await fetch('/api/batch/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: $('#ctrlModel').value,
        max_frames: parseInt($('#ctrlMaxFrames').value) || 20,
        temperature: parseFloat($('#ctrlTemp').value) || 1.0,
      }),
    });
    const data = await resp.json();
    if (data.error) {
      btn.textContent = 'Error: ' + data.error;
      setTimeout(() => { btn.textContent = 'Submit Batch'; btn.disabled = false; }, 3000);
      return;
    }
    btn.textContent = `Submitted ${data.jobs_created} jobs`;
    setTimeout(() => { btn.textContent = 'Submit Batch'; btn.disabled = false; }, 3000);
    pollBatchStatus();
  } catch (e) {
    btn.textContent = 'Submit Batch';
    btn.disabled = false;
  }
}

async function pollBatchStatus() {
  try {
    const resp = await fetch('/api/batch/status');
    const data = await resp.json();
    renderBatchJobs(data);
  } catch (e) { /* silent */ }
}

function renderBatchJobs(data) {
  const list = $('#batchJobsList');
  const summary = $('#batchSummary');
  const jobs = data.jobs || [];

  if (!jobs.length) {
    list.innerHTML = '';
    summary.textContent = '';
    return;
  }

  const collected = jobs.filter(j => j.collected).length;
  const active = jobs.filter(j => ['JOB_STATE_PENDING', 'JOB_STATE_RUNNING'].includes(j.state)).length;
  summary.textContent = `${collected}/${jobs.length} collected · ${data.total_collected.toLocaleString()} captions · ${data.total_errors} errors`;

  list.innerHTML = jobs.map(j => {
    const stateClass = j.collected ? 'collected' :
      j.state.includes('SUCCEEDED') ? 'succeeded' :
      j.state.includes('RUNNING') ? 'running' :
      j.state.includes('FAILED') ? 'failed' : 'pending';
    const stateLabel = j.collected ? 'Collected' :
      j.state.replace('JOB_STATE_', '').toLowerCase();
    const statsText = j.stats
      ? ` · ${j.stats.captions_written} captions, ${j.stats.errors} errors`
      : '';
    return `<div class="batch-job-card">
      <span class="job-name">${j.display_name}</span>
      <span class="job-frames">${j.frame_count.toLocaleString()} frames${statsText}</span>
      <span class="batch-badge ${stateClass}">${stateLabel}</span>
    </div>`;
  }).join('');

  // Disable submit button if jobs are active
  $('#btnSubmitBatch').disabled = active > 0;
}

// Add batch polling to the main status poll loop
// (find the existing setInterval for status polling and add pollBatchStatus() call)
```

**Step 4: Wire batch polling into existing poll loop**

Find the existing `setInterval` that polls `/api/status` and add `pollBatchStatus()` alongside it. The exact location depends on the current polling code — look for `setInterval` in the script section.

**Step 5: Test visually**

Open the dashboard at `http://localhost:8766`, verify the Batch Jobs section appears with a "Submit Batch" button and "50% off" badge.

**Step 6: Commit**

```bash
git add tools/dataset/caption_dashboard.html
git commit -m "feat(dashboard): add Batch Jobs UI with auto-polling and status cards"
```

---

## Task 7: Integration Test — End-to-End Dry Run

**Files:**
- Modify: `tests/tools/test_batch_api.py`

**Step 1: Write integration test**

```python
def test_full_lifecycle_mocked(tmp_path: Path) -> None:
    """Full lifecycle: prepare → submit → poll → collect."""
    # Setup mock filesystem
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    scene_dir = frames_dir / "scene_1"
    scene_dir.mkdir()
    (scene_dir / "frame_0001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 20)
    batch_dir = tmp_path / "batch_jobs"
    state_file = tmp_path / "batch_state.json"

    # Mock frame selector
    mock_sel = MagicMock()
    mock_sel.path = str(scene_dir / "frame_0001.jpg")

    with patch("tools.dataset.batch_api.select_frames_for_scene", return_value=[mock_sel]), \
         patch("tools.dataset.batch_api.dataset_image_name", return_value="s1_f0001.jpg"):
        chunks = prepare_batch_chunks(
            index=MagicMock(scene_id_list=[1]),
            completed_scenes=set(),
            images_dir=images_dir,
            frames_dir=frames_dir,
            batch_dir=batch_dir,
            max_frames=20,
            chunk_size=100,
        )

    assert len(chunks) == 1

    # Mock upload + submit
    with patch("tools.dataset.batch_api._requests") as mock_req:
        init_resp = MagicMock(status_code=200)
        init_resp.headers = {"X-Goog-Upload-URL": "https://example.com/upload"}
        upload_resp = MagicMock(status_code=200)
        upload_resp.json.return_value = {"file": {"name": "files/f1"}}
        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {
            "name": "batches/b1",
            "metadata": {"state": "JOB_STATE_PENDING"},
        }
        mock_req.post.side_effect = [init_resp, upload_resp, submit_resp]

        job = submit_batch_job(
            file_name="files/f1", model="gemini-3-flash-preview",
            api_key="key", display_name="chunk-001",
            frame_count=1, scene_ids=[1],
        )

    assert job.name == "batches/b1"

    # Mock poll → succeeded
    with patch("tools.dataset.batch_api._requests") as mock_req:
        resp = MagicMock()
        resp.json.return_value = {
            "name": "batches/b1",
            "metadata": {"state": "JOB_STATE_SUCCEEDED"},
            "response": {"responsesFile": "files/r1"},
        }
        mock_req.get.return_value = resp
        poll_batch_jobs([job], "key")

    assert job.state == "JOB_STATE_SUCCEEDED"
    assert job.result_file == "files/r1"

    # Mock collect
    result_jsonl = '{"key": "s1_f0001", "response": {"candidates": [{"content": {"parts": [{"text": "Test caption"}]}, "finishReason": "STOP"}], "usageMetadata": {}}}\n'
    with patch("tools.dataset.batch_api._requests") as mock_req:
        resp = MagicMock()
        resp.content = result_jsonl.encode()
        mock_req.get.return_value = resp
        collect_batch_results(job, images_dir, "key")

    assert job.collected is True
    assert (images_dir / "s1_f0001.txt").read_text() == "Test caption"

    # Save and reload state
    save_batch_state(state_file, [job])
    loaded = load_batch_state(state_file)
    assert loaded[0].collected is True
    assert loaded[0].stats.captions_written == 1
```

**Step 2: Run all tests**

Run: `uv run pytest tests/tools/test_batch_api.py -v`
Expected: All PASSED

**Step 3: Commit**

```bash
git add tests/tools/test_batch_api.py
git commit -m "test(batch): add full lifecycle integration test"
```

---

## Task 8: Visual Testing with Dashboard

**Step 1: Restart dashboard**

```bash
# Kill existing dashboard if running
pkill -f "caption_dashboard.py" || true
# Start fresh
uv run python tools/dataset/caption_dashboard.py --port 8766 &
```

**Step 2: Open dashboard and verify**

Open `http://localhost:8766` in browser. Verify:
- Batch Jobs section visible below Pipeline Controls
- "50% off" badge shows in green
- "Submit Batch" button is enabled
- Empty state shows no job cards

**Step 3: Take screenshots**

Save screenshots to `tests/screenshots/batch-*.png`

**Step 4: Commit**

```bash
git add tests/screenshots/
git commit -m "test(batch): add dashboard visual test screenshots"
```
