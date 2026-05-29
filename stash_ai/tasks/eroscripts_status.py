"""Scene-level matched-state probe for the Scripts sidebar tab.

Cheap, read-only check the JS calls when the Scripts tab opens. Tells the
frontend which of four states to render:

- **matched** — sidecar JSON exists *and* a ``<basename>.funscript`` is
  present on disk. Render the rich matched-state card (thumbnail, creator,
  likes, tags, "View thread" / "Re-search" buttons).
- **orphan_local** — funscript file exists alongside the video but no
  sidecar (the user got it from somewhere other than this plugin). Render
  a slim "local funscript present" note plus a "Search EroScripts" button
  in case they want to attach metadata.
- **orphan_metadata** — sidecar exists but the funscript file is gone
  (user deleted it). Render a warning and a "Re-download" button.
- **empty** — neither. Render the standard "Find on EroScripts" empty
  state.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

from ..eroscripts import metadata as metadata_mod
from ..eroscripts.metadata import SidecarMetadata
from ..stash_client import StashClient

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(_PLUGIN_ROOT, "assets", "eroscripts")
_RESULT_TTL_SECONDS = 3600


class StatusResult(TypedDict, total=False):
    status: str  # "complete" or "error"
    state: str  # matched / orphan_local / orphan_metadata / empty
    has_sidecar: bool
    has_funscript_on_disk: bool
    funscript_filename: str | None  # basename of the .funscript Stash will use
    sidecar: SidecarMetadata | None  # raw sidecar metadata, if present
    error: str | None
    request_id: str


def run(stash: StashClient, args: dict[str, Any], log: Any) -> None:
    """Entry point dispatched from ``stash-copilot.py``."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    _cleanup_stale("status_")

    request_id = str(args.get("request_id") or "")
    scene_id = str(args.get("scene_id") or "").strip()

    result: StatusResult = {
        "status": "error",
        "state": "empty",
        "has_sidecar": False,
        "has_funscript_on_disk": False,
        "funscript_filename": None,
        "sidecar": None,
        "error": None,
        "request_id": request_id,
    }

    try:
        if not scene_id:
            result["status"] = "complete"
            result["error"] = "Missing scene_id."
            _write_result(request_id, result)
            return

        # Sidecar lookup is filesystem-only — no Stash call needed.
        sidecar = metadata_mod.read(int(scene_id))
        result["has_sidecar"] = sidecar is not None
        result["sidecar"] = sidecar  # may be None

        # Filesystem probe for the primary funscript file. Mirrors the Q6
        # decision: only the unsuffixed `<basename>.funscript` is what Stash
        # auto-detects; any -1/-2 variants are inert.
        try:
            scene = stash.find_scene(int(scene_id))
        except Exception as e:
            log(f"find_scene({scene_id}) failed: {e}", "warning")
            scene = None
        if scene:
            files = scene.get("files") or []
            if files and isinstance(files[0], dict):
                path = files[0].get("path")
                if isinstance(path, str) and path:
                    base, _ = os.path.splitext(os.path.basename(path))
                    candidate = os.path.join(os.path.dirname(path), f"{base}.funscript")
                    if os.path.isfile(candidate):
                        result["has_funscript_on_disk"] = True
                        result["funscript_filename"] = os.path.basename(candidate)

        # Decide state.
        if result["has_sidecar"] and result["has_funscript_on_disk"]:
            result["state"] = "matched"
        elif result["has_funscript_on_disk"]:
            result["state"] = "orphan_local"
        elif result["has_sidecar"]:
            result["state"] = "orphan_metadata"
        else:
            result["state"] = "empty"

        result["status"] = "complete"
    except Exception as e:
        log(f"EroScripts status task crashed: {e}", "error")
        result["status"] = "error"
        result["error"] = f"Internal error: {e}"

    _write_result(request_id, result)


def _write_result(request_id: str, payload: StatusResult) -> None:
    suffix = request_id or "default"
    path = os.path.join(_RESULTS_DIR, f"status_{suffix}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _cleanup_stale(prefix: str) -> None:
    now = time.time()
    try:
        entries = os.listdir(_RESULTS_DIR)
    except OSError:
        return
    for name in entries:
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        path = os.path.join(_RESULTS_DIR, name)
        try:
            if now - os.path.getmtime(path) > _RESULT_TTL_SECONDS:
                os.remove(path)
        except OSError:
            pass
