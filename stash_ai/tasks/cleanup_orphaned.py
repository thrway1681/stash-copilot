"""Clean up embeddings for scenes that no longer exist in Stash (log-only).

Maintenance task: finds embeddings whose scene was deleted (e.g. while the plugin
was disabled or before the Scene.Destroy.Post hook existed) and removes them.
Reads valid scene IDs straight from the Stash sqlite DB — it needs no
``StashClient``, no plugin settings, and writes no result file (log-only, so no
``result_key`` / ``ResultStore``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.tools.database import get_readonly_connection, get_stash_db_path

if TYPE_CHECKING:
    from .dispatch import TaskContext


class CleanupOrphanedTask:
    """Remove embeddings for scenes no longer present in Stash (log-only)."""

    def __init__(
        self,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        dry_run: bool = True,
    ) -> None:
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)
        self.dry_run = dry_run

    @classmethod
    def from_context(cls, ctx: TaskContext) -> CleanupOrphanedTask:
        """Build from the standard :class:`TaskContext`.

        ``dry_run`` is resolved from ``ctx.args`` and defaults to the safe string
        ``"true"`` (cleanup only deletes when explicitly ``dry_run="false"``),
        matching the old handler. Reads nothing from plugin settings.
        """
        dry_run = str(ctx.args.get("dry_run", "true")).lower() == "true"
        return cls(log_callback=ctx.log, progress_callback=ctx.progress, dry_run=dry_run)

    def run(self) -> None:
        """Find and (unless dry-run) delete orphaned embeddings, logging progress."""
        self.log(f"Starting orphaned embeddings cleanup (dry_run={self.dry_run})...", "info")

        # Get all valid scene IDs from the Stash database.
        db_path = get_stash_db_path()
        if not db_path.exists():
            # Surfaced at error level via dispatch's RuntimeError branch (the old
            # handler called self.error here, which is unavailable inside a task).
            raise RuntimeError("Stash database not found")

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM scenes")
        valid_scene_ids = [row["id"] for row in cursor.fetchall()]
        conn.close()

        self.log(f"Found {len(valid_scene_ids)} scenes in Stash database", "info")

        # Storage model_key doesn't matter for orphan detection.
        storage = EmbeddingStorage()
        orphaned_ids = storage.get_orphaned_scene_ids(valid_scene_ids)

        if not orphaned_ids:
            self.log("No orphaned embeddings found", "info")
            return

        self.log(f"Found {len(orphaned_ids)} orphaned scene IDs", "info")

        if self.dry_run:
            self.log(
                f"DRY RUN: Would delete embeddings for {len(orphaned_ids)} "
                f"orphaned scenes: {orphaned_ids[:10]}{'...' if len(orphaned_ids) > 10 else ''}",
                "info",
            )
            return

        # Delete embeddings for each orphaned scene.
        total_deleted = 0
        for i, scene_id in enumerate(orphaned_ids):
            result = storage.delete_all_scene_data(scene_id)
            deleted = sum(result.values())
            total_deleted += deleted

            if (i + 1) % 10 == 0 or i == len(orphaned_ids) - 1:
                self.log(f"Progress: {i + 1}/{len(orphaned_ids)} scenes processed", "info")
                self.progress(i + 1, len(orphaned_ids))

        self.log(
            f"Cleanup complete: deleted {total_deleted} items "
            f"from {len(orphaned_ids)} orphaned scenes",
            "info",
        )
