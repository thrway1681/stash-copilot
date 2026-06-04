"""Tag-suggestion action tasks — log-only, side-effecting dispatch-seam tasks.

These back the tag-suggestion actions on the scene Tags tab (apply / dismiss /
clear). Each is a small ``SelfBuildingTask``: log-only (no ``result_key`` /
``ResultStore``), performing a Stash or :class:`EmbeddingStorage` side effect and
logging the outcome. Houses the apply / dismiss / clear actions migrated under
#4, commit 4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from stash_ai.embeddings.storage import EmbeddingStorage

if TYPE_CHECKING:
    from ..stash_client import StashClient
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


class ApplySuggestedTagTask:
    """Apply a suggested tag to a scene via Stash (log-only).

    Reads the scene's current tags, adds ``tag_id`` if not already present
    (idempotent), and writes them back with ``sceneUpdate``. Log-only: no
    ``result_key`` — the frontend calls it fire-and-forget. Touches no
    embeddings, so unlike its sibling tasks it needs only the Stash client.
    """

    def __init__(
        self,
        stash: StashClient,
        log_callback: Callable[[str, str], None] | None = None,
        scene_id: int = 0,
        tag_id: int = 0,
    ) -> None:
        self.stash = stash
        self.log = log_callback or (lambda msg, level: None)
        self.scene_id = scene_id
        self.tag_id = tag_id

    @classmethod
    def from_context(cls, ctx: TaskContext) -> ApplySuggestedTagTask:
        """Build from the standard :class:`TaskContext`.

        ``scene_id`` / ``tag_id`` come from ``ctx.args``; the missing-arg guard
        stays in :meth:`run` (dispatch always calls ``run()``) to preserve the
        no-op-on-missing behavior. Reads nothing from plugin settings.
        """
        return cls(
            stash=ctx.stash,
            log_callback=ctx.log,
            scene_id=int(ctx.args.get("scene_id", 0)),
            tag_id=int(ctx.args.get("tag_id", 0)),
        )

    def run(self) -> None:
        """Add the tag to the scene if missing; no-op + log on missing args / already-present."""
        if not self.scene_id or not self.tag_id:
            self.log("Missing scene_id or tag_id", "error")
            return

        # Get current tags
        result = self.stash.call_GQL(
            """
            query FindScene($id: ID!) {
                findScene(id: $id) { tags { id } }
            }
            """,
            {"id": str(self.scene_id)},
        )

        current_ids = [int(t["id"]) for t in result["findScene"]["tags"]]
        if self.tag_id in current_ids:
            self.log("Tag already on scene", "info")
            return

        new_ids = current_ids + [self.tag_id]

        # Update scene
        self.stash.call_GQL(
            """
            mutation SceneUpdate($input: SceneUpdateInput!) {
                sceneUpdate(input: $input) { id }
            }
            """,
            {"input": {"id": str(self.scene_id), "tag_ids": [str(i) for i in new_ids]}},
        )

        self.log(f"Applied tag {self.tag_id} to scene {self.scene_id}", "info")


class ClearDismissedTagsTask:
    """Clear all dismissed tag suggestions for a scene (log-only).

    Writes to :class:`EmbeddingStorage` and logs how many rows were cleared.
    ``model_key`` is hardcoded ``"siglip"`` because the dismissed-tag table is
    model-agnostic (keyed by scene_id), matching the old handler.
    """

    def __init__(
        self,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        scene_id: int = 0,
    ) -> None:
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.scene_id = scene_id

    @classmethod
    def from_context(cls, ctx: TaskContext) -> ClearDismissedTagsTask:
        """Build from the standard :class:`TaskContext`.

        ``scene_id`` comes from ``ctx.args``; the missing-arg guard stays in
        :meth:`run` (dispatch always calls ``run()``) to preserve the
        no-op-on-missing behavior.
        """
        return cls(
            storage=EmbeddingStorage(model_key="siglip"),
            log_callback=ctx.log,
            scene_id=int(ctx.args.get("scene_id", 0)),
        )

    def run(self) -> None:
        """Clear the scene's dismissed tags; no-op + error log if scene_id missing."""
        if not self.scene_id:
            self.log("Missing scene_id", "error")
            return
        count = self.storage.clear_dismissed_tags(self.scene_id)
        self.log(f"Cleared {count} dismissed tags for scene {self.scene_id}", "info")
