"""Find performers visually similar to a given performer.

Migrated from the inline entry-point handler under #4, commit 4. Result-producing:
the handler routes ``run()``'s dict through the seam's ``ResultStore`` as
``similar_performers_{request_id}.json`` (request_id defaults to the performer id,
which the frontend's profile path polls by bare id). Wraps the existing
``EmbedPerformersTask.find_similar_performers`` — the visual-similarity logic lives
there; this task just resolves settings and maps the result into the polled shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..embeddings.config import EmbeddingConfig
from .embed_performers import EmbedPerformersTask

if TYPE_CHECKING:
    from ..stash_client import StashClient
    from .dispatch import TaskContext


class FindSimilarPerformersTask:
    """Rank performers by embedding similarity to a query performer."""

    result_key = "similar_performers"

    def __init__(
        self,
        stash: StashClient,
        performer_id: int = 0,
        limit: int = 10,
        min_similarity: float = 0.0,
        image_provider: str | None = None,
        image_model: str | None = None,
        image_device: str = "auto",
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.stash = stash
        self.performer_id = performer_id
        self.limit = limit
        self.min_similarity = min_similarity
        self.image_provider = image_provider
        self.image_model = image_model
        self.image_device = image_device
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> FindSimilarPerformersTask:
        """Build from a standard :class:`TaskContext` — resolves performer/limits + image settings."""
        args = ctx.args
        ps = ctx.plugin_settings
        return cls(
            stash=ctx.stash,
            performer_id=int(args.get("performer_id") or 0),
            limit=int(args.get("limit") or "10"),
            min_similarity=float(args.get("min_similarity") or "0.0"),
            image_provider=ps.get("image_embedding_provider"),
            image_model=ps.get("image_embedding_model"),
            image_device=ps.get("image_embedding_device") or "auto",
            log_callback=ctx.log,
            progress_callback=ctx.progress,
        )

    def run(self) -> dict[str, Any]:
        """Find similar performers and return the polled result dict (or an error dict)."""
        try:
            self.log(f"Finding performers similar to {self.performer_id}...", "info")

            if not self.image_provider or not self.image_model:
                return {"status": "error", "error": "Image embedding provider not configured"}

            embedding_config = EmbeddingConfig(
                provider=self.image_provider,
                model=self.image_model,
                device=self.image_device,
            )

            # Reuse EmbedPerformersTask's find_similar_performers (the visual-match logic).
            task = EmbedPerformersTask(
                stash=self.stash,
                embedding_config=embedding_config,
                log_callback=self.log,
                progress_callback=self.progress,
            )

            result = task.find_similar_performers(
                performer_id=self.performer_id,
                limit=self.limit,
                min_similarity=self.min_similarity,
            )

            if result.get("success"):
                self.log(
                    f"Found {len(result.get('similar_performers', []))} similar performers", "info"
                )
                return {
                    "status": "complete",
                    "source_performer": result.get("source_performer"),
                    "results": result.get("similar_performers", []),
                    "total_found": result.get("total_found", 0),
                }
            return {"status": "error", "error": result.get("error")}

        except Exception as e:
            self.log(f"Find similar performers error: {e}", "error")
            return {"status": "error", "error": str(e)}
