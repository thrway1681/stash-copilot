"""Find scenes visually similar to a given scene (embedding similarity + filters).

Migrated from the inline entry-point handler under #4, commit 4. Result-producing:
the handler routes ``run()``'s dict through the seam's ``ResultStore`` as
``similar_results_{scene_id}.json`` (keyed by scene_id, not request_id).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..embeddings.config import EmbeddingConfig
from ..embeddings.storage import EmbeddingStorage
from ..tools.database import get_readonly_connection, get_stash_db_path
from ..tools.scene_details import get_scene_details_batch

if TYPE_CHECKING:
    from ..stash_client import StashClient
    from .dispatch import TaskContext


class FindSimilarTask:
    """Rank scenes by embedding similarity to a query scene, with optional filters."""

    result_key = "similar_results"

    def __init__(
        self,
        stash: StashClient,
        model_key: str,
        scene_id: int = 0,
        limit: int = 10,
        offset: int = 0,
        min_similarity: float = 0.0,
        exclude_common_performers: bool = False,
        visual_weight: float | None = None,
        exclude_performer_names: list[str] | None = None,
        exclude_tag_names: list[str] | None = None,
        request_id: str = "",
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.stash = stash
        self.model_key = model_key
        self.scene_id = scene_id
        self.limit = limit
        self.offset = offset
        self.min_similarity = min_similarity
        self.exclude_common_performers = exclude_common_performers
        self.visual_weight = visual_weight
        self.exclude_performer_names = exclude_performer_names or []
        self.exclude_tag_names = exclude_tag_names or []
        self.request_id = request_id
        self.log = log_callback or (lambda msg, level: None)
        self.storage = EmbeddingStorage(model_key=model_key)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> FindSimilarTask:
        """Build from a standard :class:`TaskContext` — resolves model_key + all the args."""
        args = ctx.args

        image_provider = ctx.plugin_settings.get("image_embedding_provider")
        image_model = ctx.plugin_settings.get("image_embedding_model")
        model_key = "siglip"
        if image_provider and image_model:
            model_key = EmbeddingConfig(provider=image_provider, model=image_model).model_key

        # Visual weight for dynamic embedding blend (0.0-1.0); None = stored composite.
        visual_weight_str = args.get("visual_weight", "")
        visual_weight: float | None = None
        if visual_weight_str:
            try:
                visual_weight = max(0.0, min(1.0, float(visual_weight_str)))
            except ValueError:
                ctx.log(f"Invalid visual_weight: {visual_weight_str}, using default", "warning")

        exclude_performer_names = [
            n.strip() for n in args.get("exclude_performer_names", "").split(",") if n.strip()
        ]
        exclude_tag_names = [
            n.strip() for n in args.get("exclude_tag_names", "").split(",") if n.strip()
        ]
        exclude_common_performers = args.get("exclude_common_performers", "false").lower() == "true"

        scene_id = int(args.get("scene_id", 0))
        limit = int(args.get("limit", 10))
        offset = int(args.get("offset", 0))

        filter_desc = ""
        if exclude_common_performers:
            filter_desc += " (excluding common performers)"
        if exclude_performer_names:
            filter_desc += f" (excluding performers: {', '.join(exclude_performer_names)})"
        if exclude_tag_names:
            filter_desc += f" (excluding tags: {', '.join(exclude_tag_names)})"
        ctx.log(
            f"Finding scenes similar to: {scene_id} (offset={offset}, limit={limit}){filter_desc}",
            "info",
        )
        ctx.log(f"Using embedding model: {model_key}", "info")

        return cls(
            stash=ctx.stash,
            model_key=model_key,
            scene_id=scene_id,
            limit=limit,
            offset=offset,
            min_similarity=float(args.get("min_similarity", 0.0)),
            exclude_common_performers=exclude_common_performers,
            visual_weight=visual_weight,
            exclude_performer_names=exclude_performer_names,
            exclude_tag_names=exclude_tag_names,
            request_id=args.get("request_id", ""),
            log_callback=ctx.log,
        )

    def run(self) -> dict[str, Any]:
        """Compute similar scenes and return the result dict (or an error dict)."""
        try:
            scene_id = self.scene_id

            # Get query embedding
            query_record = self.storage.get_embedding(scene_id)
            if not query_record:
                error_msg = f"Scene {scene_id} has no embedding. Run embed_scenes task first."
                self.log(error_msg, "error")
                return {"status": "error", "error": error_msg}

            # Resolve exclusion IDs from the Stash DB if filtering.
            source_performer_ids: set[int] = set()
            exclude_performer_ids: set[int] = set()
            exclude_tag_ids: set[int] = set()
            needs_db_filtering = (
                self.exclude_common_performers
                or self.exclude_performer_names
                or self.exclude_tag_names
            )

            if needs_db_filtering:
                db_path = get_stash_db_path()
                if db_path.exists():
                    conn = get_readonly_connection(db_path)
                    cursor = conn.cursor()

                    if self.exclude_common_performers:
                        cursor.execute(
                            "SELECT performer_id FROM performers_scenes WHERE scene_id = ?",
                            (scene_id,),
                        )
                        source_performer_ids = {row["performer_id"] for row in cursor.fetchall()}
                        self.log(
                            f"Source scene has {len(source_performer_ids)} performers", "debug"
                        )

                    if self.exclude_performer_names:
                        placeholders = ",".join("?" * len(self.exclude_performer_names))
                        cursor.execute(
                            f"SELECT id FROM performers WHERE name IN ({placeholders}) COLLATE NOCASE",
                            self.exclude_performer_names,
                        )
                        exclude_performer_ids = {row["id"] for row in cursor.fetchall()}
                        self.log(
                            f"Found {len(exclude_performer_ids)} performer IDs to exclude", "debug"
                        )

                    if self.exclude_tag_names:
                        placeholders = ",".join("?" * len(self.exclude_tag_names))
                        cursor.execute(
                            f"SELECT id FROM tags WHERE name IN ({placeholders}) COLLATE NOCASE",
                            self.exclude_tag_names,
                        )
                        exclude_tag_ids = {row["id"] for row in cursor.fetchall()}
                        self.log(f"Found {len(exclude_tag_ids)} tag IDs to exclude", "debug")

                    conn.close()

            # Find similar - fetch more if filtering to ensure we find enough results
            fetch_limit = 500 if needs_db_filtering else self.limit
            fetch_offset = 0 if needs_db_filtering else self.offset

            find_similar_kwargs: dict[str, Any] = {
                "query_embedding": query_record["composite_embedding"],
                "limit": fetch_limit + self.offset if needs_db_filtering else self.limit,
                "offset": fetch_offset,
                "exclude_scene_ids": [scene_id],
                "min_similarity": self.min_similarity,
            }

            if self.visual_weight is not None:
                query_visual = query_record.get("visual_embedding")
                query_metadata = query_record.get("metadata_embedding")
                if query_visual and query_metadata:
                    find_similar_kwargs["visual_weight"] = self.visual_weight
                    find_similar_kwargs["query_visual_embedding"] = query_visual
                    find_similar_kwargs["query_metadata_embedding"] = query_metadata
                    self.log(f"Using dynamic visual weight: {self.visual_weight:.2f}", "info")
                else:
                    self.log(
                        f"Query scene missing separate embeddings (visual={query_visual is not None}, "
                        f"metadata={query_metadata is not None}). Re-run 'Embed Scenes' to enable dynamic weights.",
                        "warning",
                    )

            results = self.storage.find_similar(**find_similar_kwargs)

            # Apply database-level filtering (performers and tags)
            if needs_db_filtering:
                db_path = get_stash_db_path()
                if db_path.exists():
                    conn = get_readonly_connection(db_path)
                    cursor = conn.cursor()

                    filtered_results = []
                    for r in results:
                        should_exclude = False

                        if self.exclude_common_performers and source_performer_ids:
                            cursor.execute(
                                "SELECT performer_id FROM performers_scenes WHERE scene_id = ?",
                                (r.scene_id,),
                            )
                            scene_performer_ids = {row["performer_id"] for row in cursor.fetchall()}
                            if source_performer_ids.intersection(scene_performer_ids):
                                should_exclude = True

                        if not should_exclude and exclude_performer_ids:
                            cursor.execute(
                                "SELECT performer_id FROM performers_scenes WHERE scene_id = ?",
                                (r.scene_id,),
                            )
                            scene_performer_ids = {row["performer_id"] for row in cursor.fetchall()}
                            if exclude_performer_ids.intersection(scene_performer_ids):
                                should_exclude = True

                        if not should_exclude and exclude_tag_ids:
                            cursor.execute(
                                "SELECT tag_id FROM scenes_tags WHERE scene_id = ?", (r.scene_id,)
                            )
                            scene_tag_ids = {row["tag_id"] for row in cursor.fetchall()}
                            if exclude_tag_ids.intersection(scene_tag_ids):
                                should_exclude = True

                        if not should_exclude:
                            filtered_results.append(r)

                    conn.close()
                    self.log(
                        f"Filtered {len(results)} results to {len(filtered_results)} after applying exclusions",
                        "debug",
                    )
                    results = filtered_results

                # Apply offset and limit to filtered results
                results = results[self.offset : self.offset + self.limit]

            # Output results
            self.log("=" * 50, "info")
            self.log("SIMILAR SCENES", "info")
            self.log("=" * 50, "info")

            if not results:
                self.log("No similar scenes found", "info")
            else:
                for r in results:
                    self.log(f"Scene {r.scene_id}: similarity={r.similarity:.4f}", "info")
                    if r.visual_description:
                        preview = (
                            r.visual_description[:100] + "..."
                            if len(r.visual_description) > 100
                            else r.visual_description
                        )
                        self.log(f"  Description: {preview}", "info")

            self.log("=" * 50, "info")

            # Fetch full scene details from SQLite for all results
            scene_details = get_scene_details_batch([r.scene_id for r in results], self.log)

            result_data = []
            for r in results:
                scene = scene_details.get(r.scene_id, {})
                result_data.append(
                    {"scene_id": r.scene_id, "similarity": r.similarity, "scene": scene}
                )

            # has_more is true if we got a full page of results
            has_more = len(results) == self.limit

            # Include filter mode so frontend knows which tab these results belong to
            filter_mode = "different-performers" if self.exclude_common_performers else "all"
            result_json: dict[str, Any] = {
                "status": "complete",
                "query_scene_id": scene_id,
                "model_key": self.model_key,
                "results": result_data,
                "offset": self.offset,
                "limit": self.limit,
                "has_more": has_more,
                "filter_mode": filter_mode,
                "request_id": self.request_id,
            }
            if self.visual_weight is not None:
                result_json["visual_weight"] = self.visual_weight

            # Also output as JSON for programmatic access
            self.log("JSON_RESULT:" + json.dumps(result_data), "debug")
            return result_json

        except Exception as e:
            self.log(f"Unexpected error: {e}", "error")
            return {"status": "error", "error": f"Unexpected error: {e}"}
