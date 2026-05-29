"""Tests for tools.dataset.caption_runner — scene processing and orchestration."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tools.dataset.caption_runner import (
    process_scene_frames,
    _find_error_captions,
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


@patch("tools.dataset.caption_runner.gemini_caption_frame")
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
        model="gemini-3-flash-preview",
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


@patch("tools.dataset.caption_runner.gemini_caption_frame")
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


@patch("tools.dataset.caption_runner.gemini_caption_frame")
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


def test_find_error_captions(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Good caption
    (images_dir / "s1_f0001.txt").write_text("A nice caption.", encoding="utf-8")
    # Error captions
    (images_dir / "s1_f0002.txt").write_text("[ERROR: Prompt blocked: SAFETY]", encoding="utf-8")
    (images_dir / "s1_f0003.txt").write_text("[ERROR: timeout]", encoding="utf-8")
    # Non-.txt file (should be ignored)
    (images_dir / "s1_f0001.jpg").write_bytes(b"\xff\xd8fake")

    errors = _find_error_captions(images_dir)

    assert len(errors) == 2
    assert errors[0].name == "s1_f0002.txt"
    assert errors[1].name == "s1_f0003.txt"


def test_find_error_captions_empty_dir(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    assert _find_error_captions(images_dir) == []


@patch("tools.dataset.caption_runner.gemini_caption_frame")
def test_process_scene_frames_retries_error_captions(
    mock_caption: MagicMock, tmp_path: Path,
) -> None:
    """Error captions from a previous run should be retried, not skipped."""
    mock_caption.return_value = CaptionResult(
        caption="Fixed caption.", input_tokens=1500, output_tokens=60,
    )

    scene_dir = _make_scene_dir(tmp_path, "88", n_frames=2)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    frame_paths = [str(p) for p in sorted(scene_dir.glob("frame_*.jpg"))]

    # Pre-create both: one good, one error
    name0 = dataset_image_name("88", Path(frame_paths[0]))
    name1 = dataset_image_name("88", Path(frame_paths[1]))
    (images_dir / name0).write_bytes(b"img")
    (images_dir / name0.replace(".jpg", ".txt")).write_text("Good caption.", encoding="utf-8")
    (images_dir / name1).write_bytes(b"img")
    (images_dir / name1.replace(".jpg", ".txt")).write_text(
        "[ERROR: Prompt blocked: SAFETY]", encoding="utf-8",
    )

    mock_budget = MagicMock()

    result = process_scene_frames(
        scene_id="88",
        frame_paths=frame_paths,
        prompt="test",
        model="m",
        api_key="k",
        temperature=1.0,
        images_dir=images_dir,
        budget=mock_budget,
        workers=1,
    )

    # Good caption skipped, error caption retried → 1 API call
    assert mock_caption.call_count == 1
    assert mock_budget.acquire.call_count == 1
    # The error caption should now be fixed
    assert (images_dir / name1.replace(".jpg", ".txt")).read_text() == "Fixed caption."
    # Good caption untouched
    assert (images_dir / name0.replace(".jpg", ".txt")).read_text() == "Good caption."
