"""EroScripts search task.

Multi-query merge per Q5(e): runs filename / title / enriched variants in
sequence (politeness over parallel since we already have low latency for one
search and want to stay well below Discourse's rate limit), dedupes by
topic id, and ranks results by like_count then by topic id descending so
recent threads break ties first.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, TypedDict

from ..eroscripts import auth as auth_store
from ..eroscripts.client import EroScriptsClient, SearchResult
from ..eroscripts.query_builder import (
    QueryInputs,
    build_modal_queries,
    default_initial_query,
    with_category,
)
from ..stash_client import StashClient

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(_PLUGIN_ROOT, "assets", "eroscripts")
_RESULT_TTL_SECONDS = 3600
_MAX_RESULTS = 25


class SearchTaskResult(TypedDict, total=False):
    status: str  # "complete" or "error"
    results: list[dict]
    queries_run: list[str]
    suggested_query: str
    auth_required: bool
    rate_limited: bool
    error: str | None
    request_id: str


def run(stash: StashClient, args: dict[str, Any], log: Any) -> None:
    """Entry point dispatched from ``stash-copilot.py``.

    Args:
        request_id: Frontend correlation id.
        scene_id: Stash scene id used to build the default query (optional
            if ``query`` is provided).
        query: Optional explicit search string. If empty, the task derives
            one from the scene's title / first file basename.
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    _cleanup_stale("search_")

    request_id = str(args.get("request_id") or "")
    scene_id = str(args.get("scene_id") or "").strip()
    explicit_query = (args.get("query") or "").strip()

    result: SearchTaskResult = {
        "status": "error",
        "results": [],
        "queries_run": [],
        "suggested_query": "",
        "auth_required": False,
        "rate_limited": False,
        "error": None,
        "request_id": request_id,
    }

    try:
        stored = auth_store.load()
        if stored is None:
            result["status"] = "complete"
            result["auth_required"] = True
            result["error"] = "Not authenticated. Paste your eroscripts `_t` cookie."
            _write_result(request_id, result)
            return

        scene_inputs = _scene_query_inputs(stash, scene_id, log) if scene_id else None
        suggested = explicit_query or (default_initial_query(scene_inputs) if scene_inputs else "")
        result["suggested_query"] = suggested

        if explicit_query:
            queries = [with_category(explicit_query)]
        elif scene_inputs is not None:
            queries = build_modal_queries(scene_inputs)
        else:
            queries = []

        if not queries:
            result["status"] = "complete"
            result["error"] = (
                "Couldn't build a search query — the scene has no title or filename to derive from."
            )
            _write_result(request_id, result)
            return

        client = EroScriptsClient(stored.cookie)
        merged: dict[int, SearchResult] = {}
        for q in queries:
            result["queries_run"].append(q)
            resp = client.search(q)
            if not resp.ok:
                if resp.status_code in (401, 403):
                    result["status"] = "complete"
                    result["auth_required"] = True
                    result["error"] = resp.error
                    _write_result(request_id, result)
                    return
                if resp.rate_limited:
                    result["status"] = "complete"
                    result["rate_limited"] = True
                    result["error"] = resp.error
                    _write_result(request_id, result)
                    return
                # Continue with other variants on transient errors; first
                # success or final-failure determines outcome.
                log(f"Eroscripts search variant failed: {resp.error}", "warning")
                continue
            for item in resp.results:
                if item.topic_id not in merged:
                    merged[item.topic_id] = item

        ranked = sorted(merged.values(), key=lambda r: (r.like_count, r.topic_id), reverse=True)
        result["results"] = [_serialize_result(r) for r in ranked[:_MAX_RESULTS]]
        result["status"] = "complete"
        log(
            f"Eroscripts search returned {len(ranked)} merged results "
            f"across {len(queries)} queries",
            "info",
        )
    except Exception as e:
        log(f"EroScripts search task crashed: {e}", "error")
        result["status"] = "error"
        result["error"] = f"Internal error: {e}"

    _write_result(request_id, result)


def _scene_query_inputs(stash: StashClient, scene_id: str, log: Any) -> QueryInputs | None:
    """Look up scene title, file basename, studio, performer via stashapi."""
    try:
        scene = stash.find_scene(int(scene_id))
    except Exception as e:
        log(f"find_scene({scene_id}) failed: {e}", "warning")
        return None
    if not scene:
        return None
    title = scene.get("title") or None
    files = scene.get("files") or []
    filename = None
    if files and isinstance(files[0], dict):
        filename = files[0].get("basename") or files[0].get("path")
    studio = (scene.get("studio") or {}).get("name") if scene.get("studio") else None
    performers = [p.get("name") for p in (scene.get("performers") or []) if p.get("name")]
    return QueryInputs(title=title, filename=filename, studio=studio, performers=performers)


def _serialize_result(r: SearchResult) -> dict:
    """Convert a SearchResult dataclass into the JSON shape consumed by JS."""
    payload = asdict(r)
    # Pick the closest-to-200px thumbnail for card display.
    payload["thumbnail_url"] = _pick_thumbnail(r.thumbnails)
    payload["avatar_url"] = _resolve_avatar(r.creator_avatar_template)
    return payload


def _pick_thumbnail(thumbnails: list[dict]) -> str | None:
    if not thumbnails:
        return None
    target = 200
    best = min(
        thumbnails,
        key=lambda t: abs((t.get("width") or 0) - target) if t.get("url") else 1e9,
    )
    return best.get("url")


def _resolve_avatar(template: str | None) -> str | None:
    if not template:
        return None
    url = template.replace("{size}", "64")
    if url.startswith("/"):
        url = f"https://discuss.eroscripts.com{url}"
    return url


def _write_result(request_id: str, payload: SearchTaskResult) -> None:
    suffix = request_id or "default"
    path = os.path.join(_RESULTS_DIR, f"search_{suffix}.json")
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
