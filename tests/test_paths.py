"""Tests for ``stash_ai.paths`` — the data-directory resolution seam (#14, commit 1).

These assert observable outcomes, not internal string building: where the data root
resolves with/without ``STASH_CONFIG_DIR``, that it is created on demand, and that
each typed sub-path sits under that root. All filesystem effects are redirected into
``tmp_path`` (via ``STASH_CONFIG_DIR`` or a patched ``Path.home``) so the suite never
touches the real ``~/.stash``.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from stash_ai import paths


def test_data_dir_uses_stash_config_dir_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With STASH_CONFIG_DIR set, the root is <that>/stash-copilot/ and exists."""
    monkeypatch.setenv("STASH_CONFIG_DIR", str(tmp_path))

    root = paths.data_dir()

    assert root == tmp_path / "stash-copilot"
    assert root.is_dir()


def test_data_dir_falls_back_to_home_stash_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With STASH_CONFIG_DIR unset, the root is ~/.stash/stash-copilot/."""
    monkeypatch.delenv("STASH_CONFIG_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    root = paths.data_dir()

    assert root == tmp_path / ".stash" / "stash-copilot"
    assert root.is_dir()


def test_data_dir_is_created_on_demand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The data root does not need to pre-exist; data_dir() creates it."""
    config_dir = tmp_path / "stash-config"
    monkeypatch.setenv("STASH_CONFIG_DIR", str(config_dir))
    assert not (config_dir / "stash-copilot").exists()

    root = paths.data_dir()

    assert root.is_dir()


@pytest.mark.parametrize(
    ("subpath", "leaf"),
    [
        (paths.embeddings_db_path, "stash_copilot.sqlite"),
        (paths.frames_cache_dir, "embedded_frames"),
        (paths.scene_vision_dir, "scene_vision"),
        (paths.o_moment_cache_dir, "o_moment_cache"),
        (paths.performer_frames_dir, "performer_frames"),
        (paths.exports_dir, "exports"),
        (paths.eroscripts_auth_path, "eroscripts_auth.json"),
    ],
)
def test_typed_subpaths_sit_directly_under_the_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subpath: Callable[[], Path],
    leaf: str,
) -> None:
    """Each typed sub-path is <data_dir>/<leaf>."""
    monkeypatch.setenv("STASH_CONFIG_DIR", str(tmp_path))
    root = tmp_path / "stash-copilot"

    resolved = subpath()

    assert resolved == root / leaf
    assert resolved.parent == root
