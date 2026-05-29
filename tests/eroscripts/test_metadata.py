"""Unit tests for stash_ai.eroscripts.metadata sidecar persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from stash_ai.eroscripts import metadata as metadata_mod


@pytest.fixture
def isolated_sidecar_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SIDECAR_DIR to a tmp path so tests don't touch real plugin state."""
    monkeypatch.setattr(metadata_mod, "SIDECAR_DIR", str(tmp_path))
    return tmp_path


class TestSidecar:
    def test_read_returns_none_for_missing(self, isolated_sidecar_dir: Path) -> None:
        assert metadata_mod.read(42) is None

    def test_write_then_read_roundtrip(self, isolated_sidecar_dir: Path) -> None:
        payload: metadata_mod.SidecarMetadata = {
            "scene_id": 42,
            "eroscripts_topic_id": 1234,
            "eroscripts_thread_url": "https://discuss.eroscripts.com/t/1234",
            "eroscripts_thread_title": "Test Thread",
            "eroscripts_creator_username": "alice",
            "eroscripts_creator_avatar_url": "https://example.com/alice.png",
            "eroscripts_like_count": 5,
            "eroscripts_tags": ["vr", "multi-axis"],
            "eroscripts_post_created_at": "2024-01-01T00:00:00Z",
            "funscript_filename": "myscene.funscript",
            "funscript_path": "/videos/myscene.funscript",
            "funscript_sha256": "deadbeef" * 8,
            "attachment_original_filename": "Original Name.funscript",
            "attachment_url": "/uploads/short-url/abc.funscript",
            "downloaded_at": "2024-01-15T12:00:00Z",
        }
        metadata_mod.write(42, payload)
        loaded = metadata_mod.read(42)
        assert loaded is not None
        # All caller-provided fields preserved.
        for k, v in payload.items():
            assert loaded[k] == v
        # Schema version stamped.
        assert loaded["schema_version"] == metadata_mod.SCHEMA_VERSION

    def test_write_overwrites_existing(self, isolated_sidecar_dir: Path) -> None:
        metadata_mod.write(7, {"scene_id": 7, "eroscripts_thread_title": "v1"})
        metadata_mod.write(7, {"scene_id": 7, "eroscripts_thread_title": "v2"})
        loaded = metadata_mod.read(7)
        assert loaded is not None
        assert loaded["eroscripts_thread_title"] == "v2"

    def test_read_handles_corrupted_json(self, isolated_sidecar_dir: Path) -> None:
        path = isolated_sidecar_dir / "99.json"
        path.write_text("{this is not json")
        # Corrupted sidecar should not crash — return None so caller treats
        # it as missing and re-writes on next download.
        assert metadata_mod.read(99) is None

    def test_remove_deletes_sidecar(self, isolated_sidecar_dir: Path) -> None:
        metadata_mod.write(11, {"scene_id": 11})
        assert metadata_mod.read(11) is not None
        metadata_mod.remove(11)
        assert metadata_mod.read(11) is None

    def test_remove_is_idempotent(self, isolated_sidecar_dir: Path) -> None:
        # Removing a non-existent sidecar must not raise.
        metadata_mod.remove(404)

    def test_sidecar_path_uses_scene_id(self, isolated_sidecar_dir: Path) -> None:
        path = metadata_mod.sidecar_path(42)
        assert path.endswith("42.json")

    def test_string_scene_id_accepted(self, isolated_sidecar_dir: Path) -> None:
        # The sidecar API accepts either int or str scene id; the JSON
        # filename is the literal value either way.
        metadata_mod.write("abc", {"scene_id": 0})
        assert (isolated_sidecar_dir / "abc.json").exists()
