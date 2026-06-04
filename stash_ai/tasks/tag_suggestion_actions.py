"""Tag-suggestion action tasks — log-only, side-effecting dispatch-seam tasks.

These back the dismissed-tag actions on the scene Tags tab. Each is a small
``SelfBuildingTask``: log-only (no ``result_key`` / ``ResultStore``), it performs
an :class:`EmbeddingStorage` side effect and logs the outcome. Sibling actions
(``apply_suggested_tag``, ``clear_dismissed_tags``) join this module as they
migrate under #4, commit 4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from stash_ai.embeddings.storage import EmbeddingStorage

if TYPE_CHECKING:
    from .dispatch import TaskContext


class DismissSuggestedTagTask:
    """Dismiss a tag suggestion for a scene (log-only; writes to EmbeddingStorage)."""

    def __init__(
        self,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        scene_id: int = 0,
        tag_id: int = 0,
    ) -> None:
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.scene_id = scene_id
        self.tag_id = tag_id

    @classmethod
    def from_context(cls, ctx: TaskContext) -> DismissSuggestedTagTask:
        """Build from the standard :class:`TaskContext`.

        ``scene_id`` / ``tag_id`` come from ``ctx.args``; the missing-arg guard
        stays in :meth:`run` (dispatch always calls ``run()``, so a built task
        cannot be skipped here) to keep the no-op-on-missing behavior identical.
        ``model_key`` is hardcoded ``"siglip"`` because the dismissed-tag table is
        model-agnostic (keyed by scene_id/tag_id), matching the old handler.
        """
        return cls(
            storage=EmbeddingStorage(model_key="siglip"),
            log_callback=ctx.log,
            scene_id=int(ctx.args.get("scene_id", 0)),
            tag_id=int(ctx.args.get("tag_id", 0)),
        )

    def run(self) -> None:
        """Persist the dismissal; no-op + error log if scene_id/tag_id missing."""
        if not self.scene_id or not self.tag_id:
            self.log("Missing scene_id or tag_id", "error")
            return
        self.storage.save_dismissed_tag(self.scene_id, self.tag_id)
        self.log(f"Dismissed tag {self.tag_id} for scene {self.scene_id}", "info")
