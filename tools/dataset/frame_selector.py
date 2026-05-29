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

    # Pre-built scene -> row range lookup for fast slicing
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

    # Pre-build scene -> row range lookup.
    # Data is ordered by (scene_id, frame_index), so each scene is a contiguous block.
    _scene_ranges: dict[int, tuple[int, int]] = {}
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
    max_frames: int = 20,
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

    # Extract this scene's data (contiguous slice -- fast for mmap)
    embeddings = np.array(index.vectors[start:end], dtype=np.float32)
    frame_idxs = index.frame_indices[start:end]
    scene_timestamps = index.timestamps[start:end].tolist()

    # Build frame paths from frame indices
    scene_dir = frames_dir / f"scene_{scene_id}"
    frame_paths: list[str] = []
    for fidx in frame_idxs:
        frame_paths.append(str(scene_dir / f"frame_{int(fidx) + 1:04d}.jpg"))

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
