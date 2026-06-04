"""Clear chat history task — log-only, routed through the dispatch seam.

A deliberately dependency-free task (#4, commit 4): unlike :class:`ChatTask`,
whose ``__init__`` eagerly builds an LLM provider and the full DB toolset and
creates the assets directory, this only needs the history-file path — so
dispatching ``clear_chat`` stays as cheap as the original hand-rolled handler.

Log-only: it declares no ``result_key`` and does not route through
``ResultStore``. It deletes a file; the frontend polls nothing.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dispatch import TaskContext


def chat_history_path() -> str:
    """Absolute path to ``assets/chat_history.json`` — the file ``ChatTask`` writes.

    Resolved as ``<plugin-root>/assets/chat_history.json`` from this module's
    location (``stash_ai/tasks/`` sits two levels below the plugin root), mirroring
    :meth:`ChatTask._get_assets_dir` so clearing targets exactly the file the chat
    task persists. Unlike that method this does NOT create the directory: clearing
    a non-existent history is a no-op, never a side effect.
    """
    plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(plugin_dir, "assets", "chat_history.json")


class ClearChatTask:
    """Delete the chat history file if present (log-only)."""

    def __init__(
        self,
        history_file: str,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.history_file = history_file
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> ClearChatTask:
        """Build from the standard :class:`TaskContext`.

        Reads nothing from plugin settings or args (the ``clear_chat`` mode
        ignores its args); only needs the history path and the log callback.
        """
        return cls(history_file=chat_history_path(), log_callback=ctx.log)

    def run(self) -> None:
        """Remove the history file if it exists; log the outcome either way."""
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
            self.log("Chat history cleared", "info")
        else:
            self.log("No chat history to clear", "info")
