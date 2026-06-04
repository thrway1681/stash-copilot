"""Task for generating performer embeddings from aggregated scene embeddings."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..embeddings.config import EmbeddingConfig
from ..embeddings.storage import EmbeddingStorage
from ..recommendations.performer_profile import PerformerProfileBuilder
from ..recommendations.performer_types import PerformerEmbeddingConfig
from ..recommendations.types import (
    EngagementScoringMethod,
    EngagementWeights,
    TimeDecayConfig,
)

if TYPE_CHECKING:
    from ..stash_client import StashClient
    from .dispatch import TaskContext


@dataclass
class EmbedPerformersTaskConfig:
    """Configuration for performer embedding generation task."""

    # Minimum scenes required to create embedding
    min_scenes: int = 2

    # Maximum scenes to use for embedding
    max_scenes: int = 50

    # Use engagement weighting
    use_engagement_weighting: bool = True

    # Include scenes without engagement data
    include_unwatched: bool = True

    # Scoring method for engagement
    scoring_method: EngagementScoringMethod = EngagementScoringMethod.BASE_WEIGHTED


class EmbedPerformersTask:
    """
    Task for generating performer embeddings from aggregated scene embeddings.

    Workflow:
    1. Get all performers from Stash database
    2. For each performer, get their scenes with embeddings
    3. Calculate engagement-weighted average of scene embeddings
    4. Store as performer embedding

    This enables:
    - Find similar performers (visual similarity)
    - Performer-based scene recommendations
    - AI-generated performer descriptions
    """

    def __init__(
        self,
        stash: "StashClient",
        embedding_config: EmbeddingConfig,
        task_config: EmbedPerformersTaskConfig | None = None,
        weights: EngagementWeights | None = None,
        time_decay: TimeDecayConfig | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Initialize the performer embedding task.

        Args:
            stash: StashClient instance
            embedding_config: Config for embedding model selection
            task_config: Optional task-specific configuration
            weights: Optional engagement weights
            time_decay: Optional time decay configuration
            log_callback: Optional logging callback
            progress_callback: Optional progress callback
        """
        self.stash = stash
        self.embedding_config = embedding_config
        self.config = task_config or EmbedPerformersTaskConfig()
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

        # Run-time selectors resolved by ``from_context``; ``run()`` reads these
        # to choose single-performer vs all-performers mode (the dispatch seam
        # calls ``run()`` with no arguments).
        self.performer_id: int | None = None
        self.force: bool = False

        # Initialize storage with model_key
        model_key = embedding_config.model_key
        self.storage = EmbeddingStorage(model_key=model_key)
        self.log(f"Using embedding model key: {model_key}", "debug")

        # Initialize profile builder
        self.profile_builder = PerformerProfileBuilder(
            storage=self.storage,
            weights=weights,
            time_decay=time_decay,
            log_callback=self.log,
        )

    @classmethod
    def from_context(cls, ctx: "TaskContext") -> "EmbedPerformersTask":
        """Build the task from a standard :class:`TaskContext` (dispatch seam, #4).

        Resolves the image-embedding config, the engagement weights, and the task
        config (``min_scenes``/``max_scenes`` from args), and caches the
        ``performer_id``/``force`` run selectors for :meth:`run`. Log-only:
        declares no ``result_key`` — it writes no result file (``find_similar_
        performers`` owns its own output).
        """
        ctx.log("Initializing performer embedding generation...", "info")

        image_provider = ctx.plugin_settings.get("image_embedding_provider")
        image_model = ctx.plugin_settings.get("image_embedding_model")
        image_device = ctx.plugin_settings.get("image_embedding_device") or "auto"
        if not image_provider or not image_model:
            raise RuntimeError(
                "Image embedding provider and model are required for performer embedding. "
                "Please configure image_embedding_provider and image_embedding_model in plugin settings."
            )

        embedding_config = EmbeddingConfig(
            provider=image_provider,
            model=image_model,
            device=image_device,
        )
        ctx.log(f"Using {image_provider}/{image_model} for performer embeddings", "info")

        # Engagement weights — note the rec_* settings map onto the
        # EngagementWeights keys (o_count/view_count/play_duration/rating).
        weights = cast(
            "EngagementWeights",
            {
                "o_count": float(ctx.plugin_settings.get("rec_o_weight") or "20.0"),
                "view_count": float(ctx.plugin_settings.get("rec_view_weight") or "2.0"),
                "play_duration": float(ctx.plugin_settings.get("rec_duration_weight") or "1.0"),
                "rating": float(ctx.plugin_settings.get("rec_rating_weight") or "1.5"),
            },
        )

        task_config = EmbedPerformersTaskConfig(
            min_scenes=int(ctx.args.get("min_scenes") or "2"),
            max_scenes=int(ctx.args.get("max_scenes") or "50"),
            use_engagement_weighting=True,
            include_unwatched=True,
        )

        task = cls(
            stash=ctx.stash,
            embedding_config=embedding_config,
            task_config=task_config,
            weights=weights,
            log_callback=ctx.log,
            progress_callback=ctx.progress,
        )

        performer_id = ctx.args.get("performer_id")
        task.performer_id = int(performer_id) if performer_id else None
        task.force = str(ctx.args.get("force", "")).lower() == "true"
        return task

    def run(self) -> dict[str, Any]:
        """Embed one performer or all (mode from the cached selectors). Log-only."""
        if self.performer_id is not None:
            self.log(f"Embedding performer {self.performer_id}...", "info")
            result = self.embed_performer(self.performer_id, force=self.force)

            if result.get("success"):
                if result.get("skipped"):
                    self.log(f"Skipped: {result.get('message')}", "info")
                else:
                    self.log(
                        f"Embedded {result.get('performer_name')}: "
                        f"{result.get('contributing_scenes')} scenes, "
                        f"score {result.get('total_engagement_score', 0):.2f}",
                        "info",
                    )
            else:
                # self.error in the old handler == log(msg, "error").
                self.log(f"Failed: {result.get('error')}", "error")
            return result

        self.log("Embedding all performers...", "info")
        result = self.embed_all_performers(force=self.force)

        self.log("=" * 50, "info")
        self.log("PERFORMER EMBEDDING COMPLETE", "info")
        self.log("=" * 50, "info")
        self.log(f"Total performers: {result.get('total_performers', 0)}", "info")
        self.log(f"Embedded: {result.get('embedded', 0)}", "info")
        self.log(f"Skipped (already embedded): {result.get('skipped', 0)}", "info")
        self.log(f"Insufficient scenes: {result.get('insufficient_scenes', 0)}", "info")
        self.log(f"Errors: {result.get('errors', 0)}", "info")

        if result.get("error_details"):
            for err in result["error_details"][:5]:
                self.log(f"  - {err}", "warning")
        return result

    def embed_performer(
        self,
        performer_id: int,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Generate and store embedding for a single performer.

        Args:
            performer_id: Stash performer ID
            force: If True, regenerate even if embedding exists

        Returns:
            Dict with success status and embedding info
        """
        # Get performer info
        performer = self.profile_builder.get_performer_by_id(performer_id)
        if not performer:
            return {
                "success": False,
                "performer_id": performer_id,
                "error": "Performer not found",
            }

        performer_name = performer["name"]

        # Check if already embedded
        if not force and self.storage.has_performer_embedding(performer_id):
            self.log(f"Performer {performer_name} already embedded", "debug")
            return {
                "success": True,
                "performer_id": performer_id,
                "performer_name": performer_name,
                "skipped": True,
                "message": "Already embedded",
            }

        # Build embedding config
        embed_config = PerformerEmbeddingConfig(
            min_scenes=self.config.min_scenes,
            max_scenes=self.config.max_scenes,
            use_engagement_weighting=self.config.use_engagement_weighting,
            include_unwatched=self.config.include_unwatched,
        )

        # Build embedding
        result = self.profile_builder.build_performer_embedding(
            performer_id,
            config=embed_config,
            scoring_method=self.config.scoring_method,
        )

        if result is None:
            return {
                "success": True,
                "performer_id": performer_id,
                "performer_name": performer_name,
                "skipped": True,
                "message": f"Insufficient scenes (minimum: {self.config.min_scenes})",
            }

        embedding, contributing_scenes, total_score, top_tags = result

        # Get total scene count
        scenes = self.profile_builder.get_performer_scenes(performer_id)

        # Store embedding
        import json

        self.storage.store_performer_embedding(
            performer_id=performer_id,
            embedding=embedding,
            contributing_scenes=contributing_scenes,
            total_engagement_score=total_score,
            scene_count=len(scenes),
            top_tags=json.dumps(top_tags) if top_tags else None,
        )

        self.log(
            f"Embedded {performer_name}: {contributing_scenes} scenes, score {total_score:.2f}",
            "info",
        )

        return {
            "success": True,
            "performer_id": performer_id,
            "performer_name": performer_name,
            "contributing_scenes": contributing_scenes,
            "total_engagement_score": total_score,
            "scene_count": len(scenes),
            "top_tags": top_tags,
        }

    def embed_all_performers(
        self,
        force: bool = False,
        performer_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Generate embeddings for all performers.

        Args:
            force: If True, regenerate all embeddings
            performer_ids: Optional list of specific performer IDs to process

        Returns:
            Summary of embedding generation
        """
        # Get performers to process
        if performer_ids:
            performers = []
            for pid in performer_ids:
                p = self.profile_builder.get_performer_by_id(pid)
                if p:
                    performers.append(p)
        else:
            performers = self.profile_builder.get_all_performers()

        if not performers:
            self.log("No performers found", "warning")
            return {
                "total_performers": 0,
                "embedded": 0,
                "skipped": 0,
                "insufficient_scenes": 0,
                "errors": 0,
            }

        self.log(f"Processing {len(performers)} performers", "info")

        embedded = 0
        skipped = 0
        insufficient = 0
        errors = 0
        error_details: list[str] = []

        for i, performer in enumerate(performers):
            self.progress(i, len(performers))

            performer_id = performer["id"]
            performer_name = performer["name"]

            try:
                result = self.embed_performer(performer_id, force=force)

                if result.get("success"):
                    if result.get("skipped"):
                        if "Insufficient" in result.get("message", ""):
                            insufficient += 1
                        else:
                            skipped += 1
                    else:
                        embedded += 1
                else:
                    errors += 1
                    if len(error_details) < 10:
                        error_details.append(f"{performer_name}: {result.get('error')}")

            except (ValueError, KeyError, TypeError) as e:
                errors += 1
                self.log(f"Error embedding {performer_name}: {e}", "error")
                if len(error_details) < 10:
                    error_details.append(f"{performer_name}: {type(e).__name__}: {e}")

        self.progress(len(performers), len(performers))

        # Get stats
        stats = self.storage.get_performer_stats()

        return {
            "total_performers": len(performers),
            "embedded": embedded,
            "skipped": skipped,
            "insufficient_scenes": insufficient,
            "errors": errors,
            "error_details": error_details,
            "storage_stats": stats,
        }

    def find_similar_performers(
        self,
        performer_id: int,
        limit: int = 10,
        min_similarity: float = 0.5,
    ) -> dict[str, Any]:
        """
        Find performers visually similar to the given performer.

        Args:
            performer_id: Source performer ID
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            Dict with similar performers
        """
        # Get source performer embedding
        record = self.storage.get_performer_embedding(performer_id)
        if not record:
            return {
                "success": False,
                "error": f"Performer {performer_id} has no embedding",
            }

        # Find similar performers
        similar = self.storage.find_similar_performers(
            query_embedding=record["embedding"],
            limit=limit + 1,  # +1 to exclude self
            exclude_performer_ids=[performer_id],
            min_similarity=min_similarity,
        )

        # Get performer details
        results = []
        for pid, similarity in similar[:limit]:
            performer = self.profile_builder.get_performer_by_id(pid)
            if performer:
                perf_record = self.storage.get_performer_embedding(pid)
                results.append(
                    {
                        "performer_id": pid,
                        "name": performer["name"],
                        "similarity": similarity,
                        "scene_count": perf_record["scene_count"] if perf_record else 0,
                        "gender": performer["gender"],
                        "country": performer["country"],
                    }
                )

        source_performer = self.profile_builder.get_performer_by_id(performer_id)

        return {
            "success": True,
            "source_performer": {
                "id": performer_id,
                "name": source_performer["name"] if source_performer else None,
            },
            "similar_performers": results,
            "total_found": len(results),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get performer embedding statistics."""
        return self.storage.get_performer_stats()

    def clear_all(self) -> int:
        """Clear all performer embeddings."""
        deleted = self.storage.clear_all_performer_embeddings()
        self.log(f"Cleared {deleted} performer embeddings", "info")
        return deleted
