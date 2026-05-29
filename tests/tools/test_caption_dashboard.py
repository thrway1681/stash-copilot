"""Tests for tools.dataset.caption_dashboard — HTTP server for pipeline monitoring."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tools.dataset.caption_dashboard import (
    load_status,
    load_recent_scenes,
    load_scene_frames,
    load_errors,
    RunnerManager,
    _errors_cache,
    _run_errors_scan,
)


# ── Helpers ──────────────────────────────────────────────────────────────


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


# ── Tests ────────────────────────────────────────────────────────────────


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

    scenes = load_recent_scenes(output_dir, n=5)
    assert scenes[0]["frame_count"] == 3
    assert scenes[0]["error_count"] == 0


def test_load_recent_scenes_counts_errors(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()

    record = _sample_metadata_record(5, n_frames=4)
    # Inject 2 error captions
    names = record["image_names"]
    record["captions"][names[0]] = "[ERROR: Prompt blocked: SAFETY]"
    record["captions"][names[2]] = "[ERROR: timeout]"
    _write_metadata(output_dir / "metadata.jsonl", [record])

    scenes = load_recent_scenes(output_dir, n=5)
    assert scenes[0]["frame_count"] == 4
    assert scenes[0]["error_count"] == 2


def test_load_errors_finds_error_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)

    (images_dir / "s1_f0001.txt").write_text("Good caption.", encoding="utf-8")
    (images_dir / "s1_f0002.txt").write_text("[ERROR: Prompt blocked: SAFETY]", encoding="utf-8")
    (images_dir / "s1_f0003.txt").write_text("[ERROR: timeout]", encoding="utf-8")
    (images_dir / "s1_f0004.txt").write_text("Another good caption.", encoding="utf-8")

    # Run scan synchronously so cache is populated immediately
    _errors_cache.invalidate()
    _run_errors_scan(images_dir)
    result = load_errors(output_dir)

    assert result["scanning"] is False
    assert result["total"] == 2
    assert result["errors"][0]["image_name"] == "s1_f0002.jpg"
    assert "SAFETY" in result["errors"][0]["error"]
    assert result["errors"][1]["image_name"] == "s1_f0003.jpg"
    assert "timeout" in result["errors"][1]["error"]


def test_load_errors_empty_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    output_dir.mkdir()
    _errors_cache.invalidate()
    # No images dir at all — no scan started
    result = load_errors(output_dir)
    assert result == {"total": 0, "errors": [], "scanning": False}


def test_load_errors_no_errors(tmp_path: Path) -> None:
    output_dir = tmp_path / "lora_dataset"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)

    (images_dir / "s1_f0001.txt").write_text("Good caption.", encoding="utf-8")

    _errors_cache.invalidate()
    _run_errors_scan(images_dir)
    result = load_errors(output_dir)
    assert result == {"total": 0, "errors": [], "scanning": False}


# ── RunnerManager tests ─────────────────────────────────────────────────


def test_runner_manager_not_running_initially() -> None:
    mgr = RunnerManager()
    assert mgr.is_running is False
    assert mgr.pid is None
    assert mgr.exit_code is None
    assert mgr.get_log() == []


def test_runner_manager_launch_and_stop() -> None:
    mgr = RunnerManager()
    # Launch a trivial subprocess that sleeps
    result = mgr.launch(
        api_key="fake-key",
        limit=1,
        max_cost=0.01,
        model="gemini-3-flash-preview",
    )
    assert "pid" in result
    assert result["status"] == "launched"
    assert mgr.is_running is True
    assert mgr.pid is not None

    # Stop it
    stop_result = mgr.stop()
    assert stop_result["status"] == "stopped"
    assert mgr.is_running is False


def test_runner_manager_prevents_double_launch() -> None:
    mgr = RunnerManager()
    mgr.launch(
        api_key="fake-key",
        limit=1,
        max_cost=0.01,
    )
    try:
        with pytest.raises(RuntimeError, match="already active"):
            mgr.launch(api_key="fake-key")
    finally:
        mgr.stop()


def test_runner_manager_log_captures_output() -> None:
    mgr = RunnerManager()
    mgr.launch(
        api_key="fake-key",
        limit=1,
        max_cost=0.01,
    )
    # Give it a moment to start and produce output
    time.sleep(2)
    log = mgr.get_log()
    # Should have at least the dashboard launch message
    assert any("[dashboard]" in line for line in log)
    mgr.stop()


def test_runner_manager_stop_when_not_running() -> None:
    mgr = RunnerManager()
    result = mgr.stop()
    assert result["status"] == "not_running"
