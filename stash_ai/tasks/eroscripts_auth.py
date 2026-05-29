"""EroScripts auth-validation task.

Invoked by the frontend's first-run modal (and the re-auth screen). Validates
a pasted ``_t`` cookie against ``/session/current.json`` and persists it on
success.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

from ..eroscripts import auth as auth_store
from ..eroscripts.client import EroScriptsClient

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RESULTS_DIR = os.path.join(_PLUGIN_ROOT, "assets", "eroscripts")
_RESULT_TTL_SECONDS = 3600  # delete polling files older than 1 hour at task start


class AuthResult(TypedDict, total=False):
    status: str  # "complete" or "error"
    valid: bool
    username: str | None
    error: str | None
    request_id: str


def run(args: dict[str, Any], log: Any) -> None:
    """Entry point dispatched from ``stash-copilot.py``.

    Args (from Stash plugin task input):
        request_id: Frontend correlation id; result is written at
            ``assets/eroscripts/auth_<request_id>.json``.
        cookie: The raw ``_t`` cookie value the user pasted.
        action: Optional ``"validate"`` (default), ``"clear"`` (delete stored
            credentials), or ``"check"`` (re-validate already-stored cookie).
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    _cleanup_stale("auth_")

    request_id = str(args.get("request_id") or "")
    action = (args.get("action") or "validate").strip().lower()
    result: AuthResult = {
        "status": "error",
        "valid": False,
        "username": None,
        "error": None,
        "request_id": request_id,
    }

    try:
        if action == "clear":
            auth_store.clear()
            result["status"] = "complete"
            result["valid"] = False
            log("EroScripts auth cleared", "info")
        elif action == "check":
            stored = auth_store.load()
            if not stored:
                result["status"] = "complete"
                result["valid"] = False
                result["error"] = "No stored cookie."
            else:
                info = EroScriptsClient(stored.cookie).validate_session()
                result["status"] = "complete"
                result["valid"] = info.valid
                result["username"] = info.username
                if not info.valid:
                    result["error"] = info.error
                else:
                    auth_store.save(stored.cookie, info.username)
        else:
            cookie = (args.get("cookie") or "").strip()
            if not auth_store.looks_like_cookie(cookie):
                result["status"] = "complete"
                result["valid"] = False
                result["error"] = (
                    "That doesn't look like a `_t` cookie value. "
                    "Re-copy from your browser's devtools."
                )
            else:
                info = EroScriptsClient(cookie).validate_session()
                result["status"] = "complete"
                result["valid"] = info.valid
                result["username"] = info.username
                if info.valid:
                    auth_store.save(cookie, info.username)
                    log(f"EroScripts auth validated as @{info.username}", "info")
                else:
                    result["error"] = info.error
                    log(f"EroScripts auth validation failed: {info.error}", "warning")
    except Exception as e:
        log(f"EroScripts auth task crashed: {e}", "error")
        result["status"] = "error"
        result["error"] = f"Internal error: {e}"

    _write_result(request_id, result)


def _write_result(request_id: str, payload: AuthResult) -> None:
    suffix = request_id or "default"
    path = os.path.join(_RESULTS_DIR, f"auth_{suffix}.json")
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
