"""Unit tests for stash_ai.eroscripts.download — collision + content sanity."""

from __future__ import annotations

from pathlib import Path

from stash_ai.eroscripts.download import (
    is_valid_funscript,
    save_funscript,
    sha256_bytes,
    sha256_path,
)


# A minimal valid funscript payload reused across tests.
VALID_FS_A = b'{"actions": [{"at": 0, "pos": 50}, {"at": 100, "pos": 75}]}'
VALID_FS_B = b'{"actions": [{"at": 0, "pos": 50}, {"at": 100, "pos": 25}]}'


class TestIsValidFunscript:
    def test_valid_payload(self) -> None:
        ok, err = is_valid_funscript(VALID_FS_A)
        assert ok is True
        assert err is None

    def test_empty_payload_rejected(self) -> None:
        ok, err = is_valid_funscript(b"")
        assert ok is False
        assert err is not None and "empty" in err.lower()

    def test_oversize_payload_rejected(self) -> None:
        # 51 MB of garbage — over the 50 MB cap.
        ok, err = is_valid_funscript(b"x" * (51 * 1024 * 1024))
        assert ok is False
        assert err is not None and "50" in err

    def test_non_json_rejected(self) -> None:
        ok, err = is_valid_funscript(b"this is not json")
        assert ok is False
        assert err is not None and "json" in err.lower()

    def test_missing_actions_array_rejected(self) -> None:
        ok, err = is_valid_funscript(b'{"version": "1.0"}')
        assert ok is False
        assert err is not None and "actions" in err.lower()

    def test_actions_not_array_rejected(self) -> None:
        ok, err = is_valid_funscript(b'{"actions": "not a list"}')
        assert ok is False
        assert err is not None

    def test_action_missing_at_pos_rejected(self) -> None:
        ok, err = is_valid_funscript(b'{"actions": [{"foo": 1}]}')
        assert ok is False
        assert err is not None and ("at" in err or "pos" in err)

    def test_empty_actions_array_accepted(self) -> None:
        # An empty actions array is technically valid (a no-op funscript).
        ok, err = is_valid_funscript(b'{"actions": []}')
        assert ok is True
        assert err is None


class TestSha256Helpers:
    def test_sha256_bytes_deterministic(self) -> None:
        assert sha256_bytes(b"hello") == sha256_bytes(b"hello")
        assert sha256_bytes(b"hello") != sha256_bytes(b"world")

    def test_sha256_path_matches_bytes(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        data = b"some bytes here"
        f.write_bytes(data)
        assert sha256_path(str(f)) == sha256_bytes(data)

    def test_sha256_path_returns_none_on_missing(self, tmp_path: Path) -> None:
        assert sha256_path(str(tmp_path / "nope.bin")) is None


class TestSaveFunscript:
    def test_first_save_creates_primary(self, tmp_path: Path) -> None:
        out = save_funscript(VALID_FS_A, str(tmp_path), "myscene")
        assert out.saved is True
        assert out.was_duplicate is False
        assert out.suffix_applied == 0
        assert out.saved_filename == "myscene.funscript"
        assert (tmp_path / "myscene.funscript").exists()
        assert (tmp_path / "myscene.funscript").read_bytes() == VALID_FS_A

    def test_identical_second_save_is_duplicate_no_write(self, tmp_path: Path) -> None:
        save_funscript(VALID_FS_A, str(tmp_path), "myscene")
        out = save_funscript(VALID_FS_A, str(tmp_path), "myscene")
        assert out.saved is False
        assert out.was_duplicate is True
        # Pre-existing file is untouched.
        assert (tmp_path / "myscene.funscript").read_bytes() == VALID_FS_A
        # No -1 created.
        assert not (tmp_path / "myscene-1.funscript").exists()

    def test_different_content_creates_suffix_one(self, tmp_path: Path) -> None:
        save_funscript(VALID_FS_A, str(tmp_path), "myscene")
        out = save_funscript(VALID_FS_B, str(tmp_path), "myscene")
        assert out.saved is True
        assert out.suffix_applied == 1
        assert out.saved_filename == "myscene-1.funscript"
        # Primary preserved.
        assert (tmp_path / "myscene.funscript").read_bytes() == VALID_FS_A
        # Variant present.
        assert (tmp_path / "myscene-1.funscript").read_bytes() == VALID_FS_B

    def test_suffix_chain_on_repeated_distinct_content(self, tmp_path: Path) -> None:
        save_funscript(VALID_FS_A, str(tmp_path), "myscene")
        save_funscript(VALID_FS_B, str(tmp_path), "myscene")
        # Yet another distinct content → -2.
        third = b'{"actions": [{"at": 200, "pos": 10}]}'
        out = save_funscript(third, str(tmp_path), "myscene")
        assert out.suffix_applied == 2
        assert out.saved_filename == "myscene-2.funscript"

    def test_duplicate_of_existing_suffix_returns_duplicate(self, tmp_path: Path) -> None:
        save_funscript(VALID_FS_A, str(tmp_path), "myscene")
        save_funscript(VALID_FS_B, str(tmp_path), "myscene")  # creates -1
        # Same content as -1 again → duplicate, no -2 created.
        out = save_funscript(VALID_FS_B, str(tmp_path), "myscene")
        assert out.was_duplicate is True
        assert out.saved_filename == "myscene-1.funscript"
        assert not (tmp_path / "myscene-2.funscript").exists()

    def test_missing_directory_returns_error(self, tmp_path: Path) -> None:
        out = save_funscript(VALID_FS_A,
                             str(tmp_path / "does_not_exist"), "myscene")
        assert out.saved is False
        assert out.error is not None
        assert "directory" in out.error.lower()

    def test_filename_with_special_characters_preserved(self, tmp_path: Path) -> None:
        # Stash scene basenames may contain spaces, parens, etc. We must
        # round-trip them faithfully so Stash's auto-detection works.
        out = save_funscript(VALID_FS_A, str(tmp_path), "My Scene (2024) [4K]")
        assert out.saved is True
        assert out.saved_filename == "My Scene (2024) [4K].funscript"
        assert (tmp_path / "My Scene (2024) [4K].funscript").exists()

    def test_suffix_cap_eventually_errors(self, tmp_path: Path) -> None:
        # Pre-create the primary plus 99 distinct suffix files.
        # Each must contain DIFFERENT content so save_funscript treats them
        # as occupied non-duplicates.
        for i in range(0, 100):
            name = "myscene.funscript" if i == 0 else f"myscene-{i}.funscript"
            (tmp_path / name).write_bytes(
                b'{"actions": [{"at": ' + str(i).encode() + b', "pos": 1}]}'
            )
        out = save_funscript(VALID_FS_A, str(tmp_path), "myscene")
        assert out.saved is False
        assert out.error is not None
        assert "99" in out.error or "suffix" in out.error.lower()
