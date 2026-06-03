"""Result persistence for the task-dispatch seam.

The frontend polls a task's output from a file at
``assets/{result_key}_{request_id}.json``: the backend writes the result there
after a task runs, and the JS side fetches it by request id. That convention was
hand-rolled in a dozen handlers (one bespoke serializer, the rest raw
``json.dump`` with inconsistent indentation and an ad-hoc ``if request_id``
guard). :class:`ResultStore` owns it in one place.

This is commit 2 of #4. A result-producing task declares its ``result_key`` and
its handler routes the result through the store instead of writing the file by
hand; later commits migrate the remaining handlers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultStore:
    """Reads and writes the frontend's ``{result_key}_{request_id}.json`` files.

    A single store is rooted at the plugin's ``assets`` directory and is the one
    place the result-file naming convention lives. Writing is keyed by the
    task's declared ``result_key`` and the frontend's ``request_id``; reading
    back is the same path, used by the frontend-polling contract (and by tests).
    """

    def __init__(self, assets_dir: str | Path) -> None:
        self._assets_dir = Path(assets_dir)

    def path_for(self, result_key: str, request_id: str) -> Path:
        """Return the result-file path for ``result_key`` / ``request_id``."""
        return self._assets_dir / f"{result_key}_{request_id}.json"

    def save(self, result_key: str, request_id: str, result: Any) -> Path | None:
        """Persist ``result`` for the frontend to poll.

        Writes ``assets/{result_key}_{request_id}.json`` (creating the assets
        directory if needed) and returns its path. When ``request_id`` is empty
        there is nothing for the frontend to poll, so nothing is written and
        ``None`` is returned — preserving the ``if request_id`` guard the
        hand-rolled writers used.
        """
        if not request_id:
            return None
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(result_key, request_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return path

    def load(self, result_key: str, request_id: str) -> Any:
        """Read back a previously saved result (the frontend-poll read path)."""
        with self.path_for(result_key, request_id).open(encoding="utf-8") as f:
            return json.load(f)
