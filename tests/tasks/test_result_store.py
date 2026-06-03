"""Tests for the dispatch seam's ResultStore (issue #4, commit 2).

Exercise :class:`ResultStore` through its public surface against a temp dir:
results are written under ``{result_key}_{request_id}.json``, read back
unchanged, the empty-``request_id`` guard skips writing, and the assets
directory is created on demand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stash_ai.tasks.result_store import ResultStore


def test_save_writes_under_result_key_and_request_id(tmp_path: Path) -> None:
    """save writes assets/{result_key}_{request_id}.json and returns its path."""
    store = ResultStore(tmp_path)
    result = {"status": "complete", "candidates": [1, 2, 3]}

    path = store.save("tag_dedup", "req-42", result)

    assert path == tmp_path / "tag_dedup_req-42.json"
    assert path is not None and path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == result


def test_load_round_trips_the_saved_result(tmp_path: Path) -> None:
    """A saved result reads back unchanged through load (the frontend-poll path)."""
    store = ResultStore(tmp_path)
    result: dict[str, Any] = {"status": "complete", "error": None, "candidates": []}

    store.save("tag_dedup", "abc", result)

    assert store.load("tag_dedup", "abc") == result


def test_save_skips_when_request_id_empty(tmp_path: Path) -> None:
    """No request id means nothing for the frontend to poll: no file, returns None."""
    store = ResultStore(tmp_path)

    path = store.save("tag_dedup", "", {"status": "complete"})

    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_save_creates_assets_dir_when_missing(tmp_path: Path) -> None:
    """The assets directory is created on demand."""
    assets_dir = tmp_path / "assets"
    store = ResultStore(assets_dir)
    assert not assets_dir.exists()

    path = store.save("tag_merge", "r1", {"ok": True})

    assert assets_dir.is_dir()
    assert path == assets_dir / "tag_merge_r1.json"


def test_path_for_does_not_write(tmp_path: Path) -> None:
    """path_for only computes the path; it does not touch the filesystem."""
    store = ResultStore(tmp_path)

    path = store.path_for("tag_dedup", "req-1")

    assert path == tmp_path / "tag_dedup_req-1.json"
    assert not path.exists()


def test_save_accepts_string_assets_dir(tmp_path: Path) -> None:
    """A str assets dir is accepted (matches the PLUGIN_DIR/os.path.join call site)."""
    store = ResultStore(str(tmp_path))

    path = store.save("tag_dedup", "s1", {"value": 1})

    assert path == tmp_path / "tag_dedup_s1.json"
    assert path is not None and path.exists()
