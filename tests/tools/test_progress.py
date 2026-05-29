import json
from pathlib import Path
from tools.dataset.progress import (
    load_progress, save_progress, mark_scene_complete, get_next_batch,
)

def _make_progress(tmp_path: Path, completed: list[str], pending: list[str]) -> Path:
    p = tmp_path / "progress.json"
    data = {
        "total_scenes": len(completed) + len(pending),
        "completed": completed,
        "pending": pending,
        "last_updated": "2026-02-18T00:00:00+00:00",
        "pairs_written": len(completed) * 20,
        "sessions": 1,
        "work_queue": {sid: {"scene_id": sid, "tags": [], "frame_paths": []} for sid in completed + pending},
    }
    p.write_text(json.dumps(data))
    return p

def test_load_progress(tmp_path: Path) -> None:
    path = _make_progress(tmp_path, completed=["1"], pending=["2", "3"])
    prog = load_progress(path)
    assert prog["total_scenes"] == 3
    assert "1" in prog["completed"]
    assert "2" in prog["pending"]

def test_get_next_batch(tmp_path: Path) -> None:
    path = _make_progress(tmp_path, completed=[], pending=["1","2","3","4","5"])
    prog = load_progress(path)
    batch = get_next_batch(prog, batch_size=3)
    assert len(batch) == 3
    assert batch[0]["scene_id"] == "1"

def test_mark_scene_complete(tmp_path: Path) -> None:
    path = _make_progress(tmp_path, completed=[], pending=["10","20"])
    prog = load_progress(path)
    mark_scene_complete(prog, scene_id="10", pairs_added=20)
    assert "10" in prog["completed"]
    assert "10" not in prog["pending"]
    assert prog["pairs_written"] == 20

def test_save_and_reload(tmp_path: Path) -> None:
    path = _make_progress(tmp_path, completed=[], pending=["1"])
    prog = load_progress(path)
    mark_scene_complete(prog, "1", 20)
    save_progress(prog, path)
    reloaded = load_progress(path)
    assert "1" in reloaded["completed"]
    assert reloaded["pairs_written"] == 20
