"""EroScripts session cookie storage.

Persists the user's `_t` cookie (URL-encoded Discourse session token) to a
plugin-local file with restrictive permissions, outside of the web-served
``assets/`` directory.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUTH_FILE_PATH = os.path.join(_PLUGIN_ROOT, ".eroscripts_auth.json")


@dataclass
class StoredAuth:
    """Persisted auth state. ``cookie`` is the raw ``_t`` value (URL-encoded)."""

    cookie: str
    username: str | None = None


def load() -> StoredAuth | None:
    """Return the stored auth or ``None`` if not configured / unreadable."""
    if not os.path.isfile(AUTH_FILE_PATH):
        return None
    try:
        with open(AUTH_FILE_PATH) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    cookie = data.get("cookie")
    if not isinstance(cookie, str) or not cookie:
        return None
    username = data.get("username")
    return StoredAuth(cookie=cookie, username=username if isinstance(username, str) else None)


def save(cookie: str, username: str | None) -> None:
    """Write the cookie to the auth file with mode 0600."""
    payload = {"cookie": cookie, "username": username}
    with open(AUTH_FILE_PATH, "w") as f:
        json.dump(payload, f)
    os.chmod(AUTH_FILE_PATH, stat.S_IRUSR | stat.S_IWUSR)


def clear() -> None:
    """Remove the auth file if it exists."""
    try:
        os.remove(AUTH_FILE_PATH)
    except FileNotFoundError:
        pass


def looks_like_cookie(value: str) -> bool:
    """Cheap heuristic to reject obvious garbage before sending to eroscripts.

    Discourse ``_t`` values are URL-encoded, contain ``--`` separators, and are
    typically a few hundred chars long. We reject empty / very-short / values
    that contain whitespace or newlines.
    """
    if not value or len(value) < 50:
        return False
    if any(ch.isspace() for ch in value):
        return False
    return "--" in value
