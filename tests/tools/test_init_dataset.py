import json
from pathlib import Path
from tools.dataset.init_dataset import (
    filter_scenes, compute_frame_paths, build_work_queue,
)

MOCK_SCENES = [
    {"id": "19", "tags": [
        {"name": "Embedded"}, {"name": "PAWG"}, {"name": "Big Ass"},
        {"name": "Small Tits"}, {"name": "Tan"}, {"name": "Adorable"},
    ], "performers": [{"name": "Mikaela Lafuente"}], "studio": None},
    {"id": "23", "tags": [{"name": "Embedded"}],
     "performers": [], "studio": None},
    {"id": "25", "tags": [
        {"name": "Embedded"}, {"name": "Medium Tits"}, {"name": "PAWG"},
        {"name": "Natural Tits"}, {"name": "Perfect Tits"}, {"name": "Adorable"},
    ], "performers": [], "studio": None},
]

def test_filter_scenes_removes_low_tag_scenes() -> None:
    filtered = filter_scenes(MOCK_SCENES)
    ids = [s["id"] for s in filtered]
    assert "19" in ids
    assert "23" not in ids
    assert "25" in ids

def test_filter_scenes_counts_content_tags_only() -> None:
    filtered = filter_scenes(MOCK_SCENES)
    assert any(s["id"] == "19" for s in filtered)

def test_compute_frame_paths_returns_20_paths(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_19"
    scene_dir.mkdir()
    for i in range(1, 101):
        (scene_dir / f"frame_{i:04d}.jpg").touch()
    paths = compute_frame_paths("19", frames_dir=tmp_path, n=20)
    assert len(paths) == 20
    assert paths[0] != paths[-1]

def test_compute_frame_paths_fewer_than_n(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_99"
    scene_dir.mkdir()
    for i in range(1, 8):
        (scene_dir / f"frame_{i:04d}.jpg").touch()
    paths = compute_frame_paths("99", frames_dir=tmp_path, n=20)
    assert len(paths) == 7

def test_build_work_queue_structure() -> None:
    scenes = [s for s in MOCK_SCENES if s["id"] in {"19", "25"}]
    queue = build_work_queue(scenes)
    assert len(queue) == 2
    entry = queue[0]
    assert "scene_id" in entry
    assert "tags" in entry
    assert "performers" in entry
