"""Initialize the LoRA training dataset from the Stash library.

Usage:
    uv run python tools/dataset/init_dataset.py
"""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

from tools.dataset.constants import (
    ADMIN_TAGS, DATASET_DIR, FRAMES_DIR, FRAMES_PER_SCENE,
    MIN_CONTENT_TAGS, STASH_GRAPHQL,
)


_QUERY = """
{
  allScenes {
    id
    tags { name }
    performers { name }
    studio { name }
  }
}
"""


def fetch_scenes(graphql_url: str = STASH_GRAPHQL) -> list[dict]:
    """Fetch all scenes from Stash GraphQL API."""
    resp = requests.post(graphql_url, json={"query": _QUERY}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]["allScenes"]


def filter_scenes(
    scenes: list[dict],
    min_content_tags: int = MIN_CONTENT_TAGS,
) -> list[dict]:
    """Return only scenes with at least min_content_tags non-admin tags."""
    result = []
    for scene in scenes:
        content_tags = [t["name"] for t in scene["tags"] if t["name"] not in ADMIN_TAGS]
        if len(content_tags) >= min_content_tags:
            result.append(scene)
    return result


def compute_frame_paths(
    scene_id: str,
    frames_dir: Path = FRAMES_DIR,
    n: int = FRAMES_PER_SCENE,
) -> list[Path]:
    """Return n evenly-spaced frame paths from a scene's embedded_frames dir."""
    scene_dir = frames_dir / f"scene_{scene_id}"
    if not scene_dir.exists():
        return []

    all_frames = sorted(scene_dir.glob("frame_*.jpg"))
    if not all_frames:
        return []

    if len(all_frames) <= n:
        return all_frames

    step = (len(all_frames) - 1) / (n - 1)
    indices = [round(i * step) for i in range(n)]
    return [all_frames[i] for i in indices]


def build_work_queue(scenes: list[dict]) -> list[dict]:
    """Build the work queue entries for each scene."""
    return [
        {
            "scene_id": s["id"],
            "tags": [t["name"] for t in s["tags"] if t["name"] not in ADMIN_TAGS],
            "performers": [p["name"] for p in (s.get("performers") or [])],
            "studio": (s.get("studio") or {}).get("name") or None,
        }
        for s in scenes
    ]


def init_dataset(
    dataset_dir: Path = DATASET_DIR,
    frames_dir: Path = FRAMES_DIR,
    graphql_url: str = STASH_GRAPHQL,
) -> None:
    """Initialize the dataset directory and progress checkpoint."""
    print("Fetching scenes from Stash...")
    all_scenes = fetch_scenes(graphql_url)
    print(f"  Total scenes: {len(all_scenes)}")

    selected = filter_scenes(all_scenes)
    print(f"  Scenes with {MIN_CONTENT_TAGS}+ content tags: {len(selected)}")

    print("Computing frame paths...")
    work_queue = build_work_queue(selected)
    missing_frames = []
    valid_queue = []
    for entry in work_queue:
        paths = compute_frame_paths(entry["scene_id"], frames_dir)
        if not paths:
            missing_frames.append(entry["scene_id"])
            continue
        entry["frame_paths"] = [str(p) for p in paths]
        n = len(paths)
        entry["analysis_frames"] = [
            str(paths[n // 3]),
            str(paths[n // 2]),
        ]
        valid_queue.append(entry)

    if missing_frames:
        print(f"  WARNING: {len(missing_frames)} scenes have no embedded frames - skipped")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images").mkdir(exist_ok=True)

    progress = {
        "total_scenes": len(valid_queue),
        "completed": [],
        "pending": [e["scene_id"] for e in valid_queue],
        "last_updated": datetime.now(UTC).isoformat(),
        "pairs_written": 0,
        "sessions": 0,
        "work_queue": {e["scene_id"]: e for e in valid_queue},
    }
    progress_path = dataset_dir / "progress.json"
    progress_path.write_text(json.dumps(progress, indent=2))
    print(f"  Progress checkpoint written: {progress_path}")

    (dataset_dir / "metadata.jsonl").touch()

    print(f"\nDataset initialized:")
    print(f"  Directory: {dataset_dir}")
    print(f"  Scenes queued: {len(valid_queue)}")
    print(f"  Expected pairs: {len(valid_queue) * FRAMES_PER_SCENE:,}")


if __name__ == "__main__":
    init_dataset()
