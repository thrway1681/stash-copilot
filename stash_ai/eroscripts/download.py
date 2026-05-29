"""Funscript download orchestration with collision-safe save + content sanity.

Encapsulates the Q6 collision rules:
- target is always ``<scene_basename>.funscript`` (the only name Stash and
  external players auto-detect)
- if the target already exists: byte-compare via SHA256
    * identical → return ``DuplicateOutcome`` (caller deletes the temp)
    * different → save under a numeric suffix (-1, -2, ..., capped at 99)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional


_MAX_SUFFIX = 99


@dataclass
class SaveOutcome:
    """Result of placing a downloaded funscript next to a scene file."""

    saved: bool
    saved_path: Optional[str]      # absolute path of the funscript on disk
    saved_filename: Optional[str]  # basename only (e.g. "myscene-1.funscript")
    sha256: str
    suffix_applied: Optional[int]  # 0 = primary, 1 = -1, ..., None on dup
    was_duplicate: bool            # bytes matched an existing file
    error: Optional[str] = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            h = hashlib.sha256()
            while chunk := f.read(65536):
                h.update(chunk)
            return h.hexdigest()
    except OSError:
        return None


def is_valid_funscript(content: bytes) -> tuple[bool, Optional[str]]:
    """Best-effort content sanity. Cheap protection against eroscripts threads
    where the linked file has the right extension on something else.
    """
    if not content:
        return False, "Downloaded file is empty."
    if len(content) > 50 * 1024 * 1024:
        return False, "Downloaded file is over 50 MB — likely not a funscript."
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False, "Downloaded file is not valid JSON."
    if not isinstance(data, dict):
        return False, "Funscript JSON root is not an object."
    actions = data.get("actions")
    if not isinstance(actions, list):
        return False, "Funscript is missing the `actions` array."
    if actions:
        first = actions[0]
        if not isinstance(first, dict) or "at" not in first or "pos" not in first:
            return False, "Funscript `actions` entries are missing `at`/`pos` fields."
    return True, None


def save_funscript(
    content: bytes,
    video_directory: str,
    video_basename_no_ext: str,
) -> SaveOutcome:
    """Write a funscript to the video's directory honoring collision rules.

    Args:
        content: Raw bytes downloaded from eroscripts.
        video_directory: Absolute directory containing the video file.
        video_basename_no_ext: Filename of the video without its extension
            (e.g. ``"my.scene.1080p"``).
    """
    new_sha = sha256_bytes(content)
    primary_name = f"{video_basename_no_ext}.funscript"
    primary_path = os.path.join(video_directory, primary_name)

    if not os.path.isdir(video_directory):
        return SaveOutcome(
            saved=False, saved_path=None, saved_filename=None, sha256=new_sha,
            suffix_applied=None, was_duplicate=False,
            error=f"Video directory does not exist: {video_directory}",
        )

    if os.path.exists(primary_path):
        existing_sha = sha256_path(primary_path)
        if existing_sha == new_sha:
            return SaveOutcome(
                saved=False, saved_path=primary_path, saved_filename=primary_name,
                sha256=new_sha, suffix_applied=None, was_duplicate=True,
            )
        for suffix in range(1, _MAX_SUFFIX + 1):
            candidate_name = f"{video_basename_no_ext}-{suffix}.funscript"
            candidate_path = os.path.join(video_directory, candidate_name)
            if not os.path.exists(candidate_path):
                return _write(content, candidate_path, candidate_name, new_sha, suffix)
            if sha256_path(candidate_path) == new_sha:
                return SaveOutcome(
                    saved=False, saved_path=candidate_path,
                    saved_filename=candidate_name, sha256=new_sha,
                    suffix_applied=suffix, was_duplicate=True,
                )
        return SaveOutcome(
            saved=False, saved_path=None, saved_filename=None, sha256=new_sha,
            suffix_applied=None, was_duplicate=False,
            error=(f"Refusing to save: {_MAX_SUFFIX} suffix variants already exist for "
                   f"{video_basename_no_ext}.funscript"),
        )

    return _write(content, primary_path, primary_name, new_sha, 0)


def _write(content: bytes, path: str, name: str, sha: str, suffix: int) -> SaveOutcome:
    try:
        with open(path, "wb") as f:
            f.write(content)
    except PermissionError as e:
        return SaveOutcome(
            saved=False, saved_path=None, saved_filename=None, sha256=sha,
            suffix_applied=None, was_duplicate=False,
            error=f"Permission denied writing {path}: {e}",
        )
    except OSError as e:
        return SaveOutcome(
            saved=False, saved_path=None, saved_filename=None, sha256=sha,
            suffix_applied=None, was_duplicate=False,
            error=f"Could not write {path}: {e}",
        )
    return SaveOutcome(
        saved=True, saved_path=path, saved_filename=name, sha256=sha,
        suffix_applied=suffix, was_duplicate=False,
    )
