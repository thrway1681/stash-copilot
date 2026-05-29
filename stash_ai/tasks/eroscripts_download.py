"""EroScripts attachment-list / download task.

Two sub-actions distinguished by whether ``attachment_url`` is provided:
- empty → fetch the topic, list ``.funscript`` attachments and external links,
  return them so the JS can show a dropdown when there's more than one.
- non-empty → download that specific attachment, save with collision rules,
  write the metadata sidecar, append the eroscripts URL to ``scene.urls``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from typing import Any, TypedDict

from ..stash_client import StashClient

from ..eroscripts import auth as auth_store
from ..eroscripts import download as download_mod
from ..eroscripts import metadata as metadata_mod
from ..eroscripts.client import EroScriptsClient, TopicAttachment


_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(_PLUGIN_ROOT, "assets", "eroscripts")
_RESULT_TTL_SECONDS = 3600


class DownloadResult(TypedDict, total=False):
    status: str               # "complete" or "error"
    phase: str                # "list" or "download"
    auth_required: bool
    error: str | None
    request_id: str
    # Phase=list:
    attachments: list[dict]
    external_links: list[str]
    topic_title: str
    # Phase=download:
    saved_path: str | None
    saved_filename: str | None
    suffix_applied: int | None
    was_duplicate: bool
    sha256: str
    sidecar_path: str | None


def run(stash: StashClient, args: dict[str, Any], log: Any) -> None:
    """Entry point dispatched from ``stash-copilot.py``."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    _cleanup_stale("download_")

    request_id = str(args.get("request_id") or "")
    scene_id = str(args.get("scene_id") or "").strip()
    topic_id_raw = str(args.get("topic_id") or "").strip()
    attachment_url = (args.get("attachment_url") or "").strip()

    result: DownloadResult = {
        "status": "error", "phase": "list", "auth_required": False,
        "error": None, "request_id": request_id,
    }

    try:
        if not topic_id_raw or not scene_id:
            result["status"] = "complete"
            result["error"] = "Missing scene_id or topic_id."
            _write_result(request_id, result)
            return
        topic_id = int(topic_id_raw)

        stored = auth_store.load()
        if stored is None:
            result["status"] = "complete"
            result["auth_required"] = True
            result["error"] = "Not authenticated. Paste your eroscripts `_t` cookie."
            _write_result(request_id, result)
            return

        client = EroScriptsClient(stored.cookie)
        topic_resp = client.get_topic(topic_id)
        if not topic_resp.ok:
            result["status"] = "complete"
            if topic_resp.status_code in (401, 403):
                result["auth_required"] = True
            result["error"] = topic_resp.error
            _write_result(request_id, result)
            return

        attachments = topic_resp.funscript_attachments or []
        external = topic_resp.external_links or []

        if not attachment_url:
            # Phase 1: just list attachments back to the JS so it can decide
            # auto-pick (1) vs dropdown (2+) vs surface external-only (0).
            result.update({
                "status": "complete",
                "phase": "list",
                "topic_title": topic_resp.title or "",
                "attachments": [{"filename": a.filename, "url": a.url,
                                 "size_bytes": a.size_bytes}
                                for a in attachments],
                "external_links": external,
            })
            _write_result(request_id, result)
            return

        chosen = _match_attachment(attachments, attachment_url)
        if chosen is None:
            result["status"] = "complete"
            result["error"] = "Selected attachment is not in this topic."
            _write_result(request_id, result)
            return

        ok, content, dl_err = client.download_attachment(chosen.url)
        if not ok:
            result["status"] = "complete"
            if dl_err and "expired" in dl_err.lower():
                result["auth_required"] = True
            result["error"] = dl_err or "Download failed."
            _write_result(request_id, result)
            return

        ok_content, content_err = download_mod.is_valid_funscript(content)
        if not ok_content:
            result["status"] = "complete"
            result["error"] = content_err or "Downloaded file is not a valid funscript."
            _write_result(request_id, result)
            return

        scene_video = _scene_video_target(stash, scene_id, log)
        if scene_video is None:
            result["status"] = "complete"
            result["error"] = "Scene has no video file on disk."
            _write_result(request_id, result)
            return
        video_dir, basename_no_ext, video_path = scene_video

        outcome = download_mod.save_funscript(content, video_dir, basename_no_ext)
        if outcome.error:
            result["status"] = "complete"
            result["error"] = outcome.error
            _write_result(request_id, result)
            return

        # Build sidecar metadata. Pull the matching search-result-shape fields
        # we have from topic_resp + client metadata. We keep this best-effort —
        # if a write fails the funscript itself is still saved on disk.
        try:
            sidecar = _build_sidecar(
                scene_id=int(scene_id),
                topic_id=topic_id,
                topic_resp=topic_resp,
                chosen=chosen,
                outcome=outcome,
                args=args,
            )
            metadata_mod.write(int(scene_id), sidecar)
            sidecar_path = metadata_mod.sidecar_path(int(scene_id))
        except Exception as e:  # noqa: BLE001
            log(f"Sidecar write failed (continuing): {e}", "warning")
            sidecar_path = None

        # Append the eroscripts URL to scene.urls (idempotent).
        try:
            _append_scene_url(stash, scene_id, _build_thread_url(topic_id), log)
        except Exception as e:  # noqa: BLE001
            log(f"scene.urls update failed (continuing): {e}", "warning")

        # Trigger a targeted Stash rescan of the video's directory so Stash
        # detects the newly-dropped sidecar funscript. Without this, the
        # `<basename>.funscript` file sits on disk but Stash doesn't know
        # about it until the user manually scans (or until next library
        # scan). We disable the expensive generators (sprites, previews,
        # phashes) — we only need the metadata pass to pick up the sidecar.
        if not outcome.was_duplicate:
            try:
                _trigger_targeted_scan(stash, video_path, log)
            except Exception as e:  # noqa: BLE001
                log(f"Stash scan trigger failed (continuing): {e}", "warning")

        result.update({
            "status": "complete",
            "phase": "download",
            "saved_path": outcome.saved_path,
            "saved_filename": outcome.saved_filename,
            "suffix_applied": outcome.suffix_applied,
            "was_duplicate": outcome.was_duplicate,
            "sha256": outcome.sha256,
            "sidecar_path": sidecar_path,
            "topic_title": topic_resp.title or "",
        })
        if outcome.was_duplicate:
            log(f"EroScripts: duplicate funscript ({outcome.saved_filename}) "
                f"already on disk for scene {scene_id}", "info")
        else:
            log(f"EroScripts: saved {outcome.saved_filename} for scene {scene_id}", "info")
    except Exception as e:  # noqa: BLE001 — task entry: surface to frontend
        log(f"EroScripts download task crashed: {e}", "error")
        result["status"] = "error"
        result["error"] = f"Internal error: {e}"

    _write_result(request_id, result)


# ============================================================ helpers


def _scene_video_target(
    stash: StashClient, scene_id: str, log: Any
) -> tuple[str, str, str] | None:
    """Return ``(video_directory, basename_without_extension, video_path)``.

    Uses ``scene.files[0]`` per the Q6 multi-file decision (download once,
    aimed at the primary file). The full ``video_path`` is returned so the
    targeted-scan trigger can pass it to ``metadata_scan`` and have Stash
    rescan *just that file*, not the rest of the directory.
    """
    try:
        scene = stash.find_scene(int(scene_id))
    except Exception as e:  # noqa: BLE001
        log(f"find_scene({scene_id}) failed: {e}", "warning")
        return None
    if not scene:
        return None
    files = scene.get("files") or []
    if not files or not isinstance(files[0], dict):
        return None
    path = files[0].get("path")
    if not path or not isinstance(path, str):
        return None
    video_dir = os.path.dirname(path)
    base = os.path.basename(path)
    base_no_ext, _ = os.path.splitext(base)
    return video_dir, base_no_ext, path


def _match_attachment(attachments: list[TopicAttachment],
                      attachment_url: str) -> TopicAttachment | None:
    for a in attachments:
        if a.url == attachment_url:
            return a
    return None


def _build_thread_url(topic_id: int) -> str:
    return f"https://discuss.eroscripts.com/t/{topic_id}"


def _build_sidecar(
    scene_id: int,
    topic_id: int,
    topic_resp: Any,
    chosen: TopicAttachment,
    outcome: download_mod.SaveOutcome,
    args: dict[str, Any],
) -> metadata_mod.SidecarMetadata:
    """Pack a sidecar payload from the topic response + download outcome.

    The JS layer also passes through search-result fields it already has
    (creator/avatar/like_count/tags/created_at) under ``hint_*`` arg keys
    so we avoid a second roundtrip just to recover them.
    """
    return {
        "scene_id": scene_id,
        "eroscripts_topic_id": topic_id,
        "eroscripts_thread_url": _build_thread_url(topic_id),
        "eroscripts_thread_title": topic_resp.title or "",
        "eroscripts_creator_username": (args.get("hint_creator_username") or None) or None,
        "eroscripts_creator_avatar_url": (args.get("hint_creator_avatar_url") or None) or None,
        "eroscripts_like_count": _coerce_int(args.get("hint_like_count")),
        "eroscripts_tags": _coerce_str_list(args.get("hint_tags")),
        "eroscripts_post_created_at": (args.get("hint_created_at") or None) or None,
        "funscript_filename": outcome.saved_filename or "",
        "funscript_path": outcome.saved_path or "",
        "funscript_sha256": outcome.sha256,
        "attachment_original_filename": chosen.filename,
        "attachment_url": chosen.url,
        "downloaded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _coerce_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _coerce_str_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v:
        try:
            data = json.loads(v)
            if isinstance(data, list):
                return [str(x) for x in data if x]
        except json.JSONDecodeError:
            pass
        return [t.strip() for t in v.split(",") if t.strip()]
    return []


def _trigger_targeted_scan(stash: StashClient, video_path: str, log: Any) -> None:
    """Ask Stash to rescan a single video file so it links the new sidecar.

    Passes the **video file path** (not its directory) to ``metadata_scan``.
    Stash resolves that one file and only checks its basename-matching
    sidecars (``.funscript``, ``.vtt``, ``-cover.jpg``, etc.), leaving
    other files in the same directory untouched. All generators are
    disabled — for a sidecar-only change there's no new sprites/previews/
    phashes to compute.
    """
    if not video_path:
        return
    flags = {
        "scanGenerateCovers": False,
        "scanGeneratePreviews": False,
        "scanGenerateImagePreviews": False,
        "scanGenerateSprites": False,
        "scanGeneratePhashes": False,
        "scanGenerateThumbnails": False,
        "scanGenerateClipPreviews": False,
    }
    job_id = stash.metadata_scan(paths=[video_path], flags=flags)
    log(f"Triggered Stash scan of {video_path} (job {job_id}) "
        f"to detect the new funscript", "info")


def _append_scene_url(stash: StashClient, scene_id: str,
                      url: str, log: Any) -> None:
    """Idempotently append a URL to ``scene.urls`` via stashapi."""
    if not url:
        return
    scene = stash.find_scene(int(scene_id))
    if not scene:
        return
    existing = scene.get("urls") or []
    if not isinstance(existing, list):
        existing = []
    if url in existing:
        log(f"scene.urls already contains eroscripts URL for scene {scene_id}", "debug")
        return
    new_urls = list(existing) + [url]
    stash.update_scene({"id": str(scene_id), "urls": new_urls})


def _write_result(request_id: str, payload: DownloadResult) -> None:
    suffix = request_id or "default"
    path = os.path.join(_RESULTS_DIR, f"download_{suffix}.json")
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
