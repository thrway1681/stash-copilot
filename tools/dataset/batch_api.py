"""Gemini Batch API integration for caption pipeline.

Provides 50% cost reduction over real-time captioning by submitting
frames as batch jobs. Each job processes a JSONL chunk of up to 25K
frames with a target turnaround of 24 hours.

Lifecycle: prepare_chunks -> submit_job -> poll_jobs -> collect_results
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from tools.dataset.constants import CAPTION_PROMPT
from tools.dataset.frame_selector import EmbeddingIndex, select_frames_for_scene
from tools.dataset.io_utils import dataset_image_name


# -- Types -----------------------------------------------------------------


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

    name: str  # e.g. "batches/abc123"
    display_name: str  # e.g. "chunk-001"
    state: str  # JOB_STATE_*
    submitted_at: str  # ISO timestamp
    frame_count: int
    scene_ids: list[int] = field(default_factory=list)
    file_name: str | None = None  # uploaded JSONL file ref
    result_file: str | None = None  # result file ref (when succeeded)
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


# -- State persistence -----------------------------------------------------


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


# -- API constants -------------------------------------------------------------

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta"

SAFETY_SETTINGS: list[dict[str, str]] = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "lora_dataset"
CHUNK_SIZE = 25_000


# -- Chunk preparation ---------------------------------------------------------


def prepare_batch_chunks(
    index: EmbeddingIndex,
    completed_scenes: set[int],
    images_dir: Path,
    frames_dir: Path,
    batch_dir: Path | None = None,
    max_frames: int = 20,
    chunk_size: int = CHUNK_SIZE,
    temperature: float = 1.0,
    scene_limit: int = 0,
) -> list[Path]:
    """Select frames for pending scenes and build JSONL chunks.

    Each JSONL line:
    {"key": "s{sid}_f{idx}", "request": {GenerateContentRequest with base64 image}}

    Args:
        scene_limit: Max number of pending scenes to include. 0 = all.

    Returns list of .jsonl file paths, each containing up to chunk_size requests.
    """
    if batch_dir is None:
        batch_dir = DEFAULT_OUTPUT_DIR / "batch_jobs"
    batch_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = index.scene_id_list
    pending = [s for s in all_scenes if s not in completed_scenes]
    if scene_limit > 0:
        pending = pending[:scene_limit]

    frame_pairs: list[tuple[int, Path]] = []
    for scene_id in pending:
        selections = select_frames_for_scene(
            index, scene_id, max_frames=max_frames, frames_dir=frames_dir,
        )
        for sel in selections:
            img_name = dataset_image_name(str(scene_id), Path(sel.path))
            txt_path = images_dir / img_name.replace(".jpg", ".txt")
            if txt_path.exists():
                content = txt_path.read_text(encoding="utf-8")
                if not content.startswith("[ERROR"):
                    continue
            frame_pairs.append((scene_id, Path(sel.path)))

    if not frame_pairs:
        return []

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    chunk_paths: list[Path] = []

    for chunk_idx in range(0, len(frame_pairs), chunk_size):
        chunk = frame_pairs[chunk_idx : chunk_idx + chunk_size]
        chunk_path = batch_dir / f"chunk-{timestamp}-{chunk_idx // chunk_size + 1:03d}.jsonl"

        with open(chunk_path, "w", encoding="utf-8") as f:
            for scene_id, frame_path in chunk:
                img_name = dataset_image_name(str(scene_id), frame_path)
                key = img_name.replace(".jpg", "")

                frame_b64 = base64.b64encode(frame_path.read_bytes()).decode("utf-8")

                request: dict[str, Any] = {
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


# -- File upload & job submission ----------------------------------------------


def upload_jsonl(jsonl_path: Path, api_key: str, max_retries: int = 1) -> str:
    """Upload a JSONL file via Gemini File API (resumable upload).

    Retries once with exponential backoff on failure.
    Streams file data to avoid loading multi-GB files into memory.

    Returns the file name (e.g. "files/abc123") for use in batch submission.
    """
    file_size = jsonl_path.stat().st_size
    display_name = jsonl_path.stem
    last_error: Exception | None = None

    for attempt in range(1 + max_retries):
        try:
            init_resp = requests.post(
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

            with open(jsonl_path, "rb") as f:
                upload_resp = requests.post(
                    upload_url,
                    headers={
                        "Content-Length": str(file_size),
                        "X-Goog-Upload-Offset": "0",
                        "X-Goog-Upload-Command": "upload, finalize",
                    },
                    data=f,  # Stream file instead of reading into memory
                    timeout=600,
                )
            upload_resp.raise_for_status()

            result = upload_resp.json()
            return result["file"]["name"]

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 1s, 2s exponential backoff

    raise last_error  # type: ignore[misc]


def submit_batch_job(
    file_name: str,
    model: str,
    api_key: str,
    display_name: str,
    frame_count: int,
    scene_ids: list[int],
) -> BatchJob:
    """Submit a batch job for a previously uploaded JSONL file.

    Returns BatchJob with initial state (typically BATCH_STATE_PENDING).
    """
    url = f"{GEMINI_API_BASE}/models/{model}:batchGenerateContent"
    payload: dict[str, Any] = {
        "batch": {
            "display_name": display_name,
            "input_config": {"file_name": file_name},
        }
    }

    resp = requests.post(
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
        state=data.get("metadata", {}).get("state", "BATCH_STATE_PENDING"),
        submitted_at=datetime.now(UTC).isoformat(),
        frame_count=frame_count,
        scene_ids=scene_ids,
        file_name=file_name,
        result_file=None,
        collected=False,
        stats=None,
    )


# -- Polling & result collection -----------------------------------------------

_TERMINAL_STATES = frozenset({
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
})


def poll_batch_jobs(jobs: list[BatchJob], api_key: str) -> list[BatchJob]:
    """Check status of all active batch jobs.

    Updates state and result_file for each job. Does not modify
    already-collected jobs.
    """
    for job in jobs:
        if job.collected:
            continue
        if job.state in _TERMINAL_STATES:
            continue

        try:
            resp = requests.get(
                f"{GEMINI_API_BASE}/{job.name}",
                headers={"x-goog-api-key": api_key},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            job.state = data.get("metadata", {}).get("state", job.state)

            response = data.get("response", {})
            if response.get("responsesFile"):
                job.result_file = response["responsesFile"]
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

    download_url = (
        f"https://generativelanguage.googleapis.com/download/v1beta"
        f"/{job.result_file}:download"
    )
    resp = requests.get(
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

        key: str = record.get("key", "")
        response: dict[str, Any] | None = record.get("response")
        error: Any = record.get("error")

        txt_path = images_dir / f"{key}.txt"

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
        scenes_completed=0,  # Updated by check_scene_completion() after collection
    )

    return job


# -- Scene completion & progress tracking --------------------------------------


def check_scene_completion(
    scene_ids: list[int],
    images_dir: Path,
    index: EmbeddingIndex,
    frames_dir: Path,
    max_frames: int = 20,
) -> list[int]:
    """Check which scenes have ALL selected frames captioned (non-error).

    For each scene, re-runs SmartFrameSelector to get the expected frame list,
    then checks that every selected frame has a .txt file without an [ERROR prefix.

    Returns list of fully-completed scene IDs.
    """
    completed: list[int] = []
    for sid in scene_ids:
        selections = select_frames_for_scene(
            index, sid, max_frames=max_frames, frames_dir=frames_dir,
        )
        if not selections:
            continue

        all_done = True
        for sel in selections:
            img_name = dataset_image_name(str(sid), Path(sel.path))
            txt_path = images_dir / img_name.replace(".jpg", ".txt")
            if not txt_path.exists():
                all_done = False
                break
            content = txt_path.read_text(encoding="utf-8")
            if content.startswith("[ERROR"):
                all_done = False
                break

        if all_done:
            completed.append(sid)

    return completed


def update_caption_progress(
    progress_path: Path,
    newly_completed: list[int],
    captions_written: int,
    errors: int,
) -> None:
    """Merge batch results into caption_progress.json.

    Adds newly completed scenes (deduped) and increments frame/error counters.
    """
    if not newly_completed and captions_written == 0 and errors == 0:
        return

    data: dict[str, Any] = {}
    if progress_path.exists():
        try:
            data = json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            data = {}

    existing_completed: list[int] = data.get("completed_scenes", [])
    existing_set = set(existing_completed)

    for sid in newly_completed:
        if sid not in existing_set:
            existing_completed.append(sid)
            existing_set.add(sid)

    data["completed_scenes"] = existing_completed
    data["total_frames_captioned"] = data.get("total_frames_captioned", 0) + captions_written
    data["errors"] = data.get("errors", 0) + errors
    data["last_updated"] = datetime.now(UTC).isoformat()

    progress_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
