"""Progress checkpoint management for multi-session dataset construction."""
import json
from datetime import UTC, datetime
from pathlib import Path


def load_progress(progress_path: Path) -> dict:
    """Load the progress checkpoint from disk."""
    return json.loads(progress_path.read_text(encoding="utf-8"))


def save_progress(progress: dict, progress_path: Path) -> None:
    """Persist the progress checkpoint to disk."""
    progress["last_updated"] = datetime.now(UTC).isoformat()
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def get_next_batch(progress: dict, batch_size: int = 150) -> list[dict]:
    """Return the next batch of unprocessed scene work queue entries."""
    pending_ids = progress["pending"][:batch_size]
    return [progress["work_queue"][sid] for sid in pending_ids]


def mark_scene_complete(progress: dict, scene_id: str, pairs_added: int) -> None:
    """Mark a scene as done in the in-memory progress dict."""
    if scene_id in progress["pending"]:
        progress["pending"].remove(scene_id)
    if scene_id not in progress["completed"]:
        progress["completed"].append(scene_id)
    progress["pairs_written"] += pairs_added


def session_summary(progress: dict) -> str:
    """Return a human-readable summary of current progress."""
    total = progress["total_scenes"]
    done = len(progress["completed"])
    pending = len(progress["pending"])
    pairs = progress["pairs_written"]
    pct = (done / total * 100) if total else 0
    sessions_est = max(1, pending // 150)
    return (
        f"Progress: {done}/{total} scenes ({pct:.1f}%) | "
        f"{pairs:,} pairs written | "
        f"~{sessions_est} sessions remaining"
    )
