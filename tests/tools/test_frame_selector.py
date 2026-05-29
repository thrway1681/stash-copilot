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
    for i in range(1, 51):  # 1-based: frame_0001.jpg through frame_0050.jpg
        (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"fake")

    selected = select_frames_for_scene(
        index, scene_id=1, max_frames=512, frames_dir=tmp_path / "embedded_frames",
    )
    for s in selected:
        assert Path(s.path).exists()
