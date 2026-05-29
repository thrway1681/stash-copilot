"""Sidecar metadata for downloaded eroscripts funscripts.

One JSON sidecar per scene at ``assets/eroscripts_metadata/{scene_id}.json``.
Captures the eroscripts thread context that Stash itself doesn't track —
creator, like count, forum tags, original attachment filename — so the
sidebar tab's matched-state UI can render it without re-querying eroscripts.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIDECAR_DIR = os.path.join(_PLUGIN_ROOT, "assets", "eroscripts_metadata")
SCHEMA_VERSION = 1


class SidecarMetadata(TypedDict, total=False):
    schema_version: int
    scene_id: int
    eroscripts_topic_id: int
    eroscripts_thread_url: str
    eroscripts_thread_title: str
    eroscripts_creator_username: str | None
    eroscripts_creator_avatar_url: str | None
    eroscripts_like_count: int
    eroscripts_tags: list[str]
    eroscripts_post_created_at: str | None
    funscript_filename: str
    funscript_path: str
    funscript_sha256: str
    attachment_original_filename: str
    attachment_url: str
    downloaded_at: str


def sidecar_path(scene_id: int | str) -> str:
    return os.path.join(SIDECAR_DIR, f"{scene_id}.json")


def read(scene_id: int | str) -> SidecarMetadata | None:
    path = sidecar_path(scene_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write(scene_id: int | str, payload: SidecarMetadata) -> None:
    os.makedirs(SIDECAR_DIR, exist_ok=True)
    payload["schema_version"] = SCHEMA_VERSION
    with open(sidecar_path(scene_id), "w") as f:
        json.dump(payload, f, indent=2)


def remove(scene_id: int | str) -> None:
    try:
        os.remove(sidecar_path(scene_id))
    except FileNotFoundError:
        pass
