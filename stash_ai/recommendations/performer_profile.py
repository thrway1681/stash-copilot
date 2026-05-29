"""Performer profile generation from aggregated scene embeddings."""

import json
from collections import Counter
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..embeddings.storage import EmbeddingStorage
from ..tools.database import get_readonly_connection, get_stash_db_path
from .engagement import EngagementCalculator
from .performer_types import (
    PerformerData,
    PerformerEmbeddingConfig,
    PerformerEngagementData,
    PerformerSceneData,
)
from .types import (
    EngagementScoringMethod,
    EngagementWeights,
    TimeDecayConfig,
)


class PerformerProfileBuilder:
    """
    Build performer profiles by aggregating scene embeddings.

    The performer embedding is a weighted average of embeddings from their scenes,
    where each scene's contribution is proportional to its engagement score.

    Algorithm:
    1. Query scenes via performers_scenes table
    2. Filter to scenes with embeddings
    3. Calculate engagement score per scene
    4. Compute weighted average: sum(weight_i * scene_embedding_i)
    5. Normalize to unit vector
    6. Store with model_key namespace
    """

    def __init__(
        self,
        storage: EmbeddingStorage | None = None,
        weights: EngagementWeights | None = None,
        time_decay: TimeDecayConfig | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ):
        self.storage = storage or EmbeddingStorage()
        self.calculator = EngagementCalculator(
            weights=weights,
            time_decay=time_decay,
            log_callback=log_callback,
        )
        self.log = log_callback or (lambda msg, level: None)

    def get_all_performers(self) -> list[PerformerData]:
        """Get all performers from Stash database."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return []

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, name, disambiguation, gender, birthdate, ethnicity,
                   country, height, weight, measurements, fake_tits, tattoos,
                   piercings, favorite, rating, details, image_blob
            FROM performers
            ORDER BY name
        """
        )

        performers: list[PerformerData] = []
        for row in cursor.fetchall():
            performers.append(
                cast(
                    "PerformerData",
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "disambiguation": row["disambiguation"],
                        "gender": row["gender"],
                        "birthdate": row["birthdate"],
                        "ethnicity": row["ethnicity"],
                        "country": row["country"],
                        "height": row["height"],
                        "weight": row["weight"],
                        "measurements": row["measurements"],
                        "fake_tits": row["fake_tits"],
                        "tattoos": row["tattoos"],
                        "piercings": row["piercings"],
                        "aliases": None,  # Aliases stored in separate table
                        "favorite": bool(row["favorite"]),
                        "rating": row["rating"],
                        "details": row["details"],
                        "image_blob": row["image_blob"],
                    },
                )
            )

        conn.close()
        return performers

    def get_performer_by_id(self, performer_id: int) -> PerformerData | None:
        """Get a single performer by ID."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return None

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, name, disambiguation, gender, birthdate, ethnicity,
                   country, height, weight, measurements, fake_tits, tattoos,
                   piercings, favorite, rating, details, image_blob
            FROM performers
            WHERE id = ?
        """,
            (performer_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row["id"],
            "name": row["name"],
            "disambiguation": row["disambiguation"],
            "gender": row["gender"],
            "birthdate": row["birthdate"],
            "ethnicity": row["ethnicity"],
            "country": row["country"],
            "height": row["height"],
            "weight": row["weight"],
            "measurements": row["measurements"],
            "fake_tits": row["fake_tits"],
            "tattoos": row["tattoos"],
            "piercings": row["piercings"],
            "aliases": None,  # Aliases stored in separate table
            "favorite": bool(row["favorite"]),
            "rating": row["rating"],
            "details": row["details"],
            "image_blob": row["image_blob"],
        }

    def get_performer_scenes(self, performer_id: int) -> list[PerformerSceneData]:
        """Get all scenes featuring a performer with engagement data."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return []

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                s.id as scene_id,
                ps.performer_id,
                s.title,
                s.date,
                COALESCE(view_agg.view_count, 0) as play_count,
                COALESCE(o_agg.o_count, 0) as o_count,
                COALESCE(s.play_duration, 0) as play_duration,
                s.rating as rating
            FROM scenes s
            JOIN performers_scenes ps ON s.id = ps.scene_id
            LEFT JOIN (
                SELECT scene_id, COUNT(*) as view_count
                FROM scenes_view_dates
                GROUP BY scene_id
            ) view_agg ON s.id = view_agg.scene_id
            LEFT JOIN (
                SELECT scene_id, COUNT(*) as o_count
                FROM scenes_o_dates
                GROUP BY scene_id
            ) o_agg ON s.id = o_agg.scene_id
            WHERE ps.performer_id = ?
            ORDER BY s.date DESC
        """,
            (performer_id,),
        )

        scenes: list[PerformerSceneData] = []
        for row in cursor.fetchall():
            scene_id = row["scene_id"]

            # Get tags for this scene
            cursor.execute(
                """
                SELECT t.name
                FROM tags t
                JOIN scenes_tags st ON t.id = st.tag_id
                WHERE st.scene_id = ?
            """,
                (scene_id,),
            )
            tags = [tag_row["name"] for tag_row in cursor.fetchall()]

            scenes.append(
                cast(
                    "PerformerSceneData",
                    {
                        "scene_id": scene_id,
                        "performer_id": row["performer_id"],
                        "title": row["title"],
                        "date": row["date"],
                        "play_count": row["play_count"] or 0,
                        "o_count": row["o_count"] or 0,
                        "play_duration": float(row["play_duration"] or 0),
                        "rating": row["rating"],
                        "tags": tags,
                    },
                )
            )

        conn.close()
        return scenes

    def get_performer_engagement_data(self, performer_id: int) -> PerformerEngagementData | None:
        """Get aggregated engagement data for a performer's scenes."""
        scenes = self.get_performer_scenes(performer_id)
        if not scenes:
            return None

        # Get embedded scene IDs
        embedded_ids = set(self.storage.get_embedded_scene_ids())
        scenes_with_embeddings = [s for s in scenes if s["scene_id"] in embedded_ids]

        # Aggregate stats
        total_play_count = sum(s["play_count"] for s in scenes)
        total_o_count = sum(s["o_count"] for s in scenes)
        total_play_duration = sum(s["play_duration"] for s in scenes) / 3600  # hours

        # Average rating (only from rated scenes)
        rated_scenes = [s for s in scenes if s["rating"] is not None]
        avg_rating: float | None = (
            sum(cast("int", s["rating"]) for s in rated_scenes) / len(rated_scenes)
            if rated_scenes
            else None
        )

        # Count tags across all scenes
        tag_counter: Counter[str] = Counter()
        for scene in scenes:
            tag_counter.update(scene["tags"])
        top_tags = [tag for tag, _ in tag_counter.most_common(10)]

        return {
            "performer_id": performer_id,
            "total_scenes": len(scenes),
            "scenes_with_embeddings": len(scenes_with_embeddings),
            "total_play_count": total_play_count,
            "total_o_count": total_o_count,
            "total_play_duration": total_play_duration,
            "avg_rating": avg_rating,
            "top_tags": top_tags,
        }

    def build_performer_embedding(
        self,
        performer_id: int,
        config: PerformerEmbeddingConfig | None = None,
        scoring_method: EngagementScoringMethod = EngagementScoringMethod.BASE_WEIGHTED,
    ) -> tuple[list[float], int, float, list[str]] | None:
        """
        Build embedding for a single performer from their scene embeddings.

        Args:
            performer_id: Stash performer ID
            config: Optional configuration for embedding generation
            scoring_method: Engagement scoring method to use

        Returns:
            Tuple of (embedding, contributing_scenes, total_score, top_tags) or None
        """
        config = config or PerformerEmbeddingConfig()

        # Get performer's scenes
        scenes = self.get_performer_scenes(performer_id)
        if len(scenes) < config.min_scenes:
            self.log(
                f"Performer {performer_id} has only {len(scenes)} scenes "
                f"(minimum: {config.min_scenes})",
                "debug",
            )
            return None

        # Get engagement data for scoring
        all_engagement = self.calculator.get_all_scene_engagement()

        # Filter to scenes with embeddings
        embedded_ids = set(self.storage.get_embedded_scene_ids())
        valid_scenes = [s for s in scenes if s["scene_id"] in embedded_ids]

        if len(valid_scenes) < config.min_scenes:
            self.log(
                f"Performer {performer_id} has only {len(valid_scenes)} scenes with embeddings "
                f"(minimum: {config.min_scenes})",
                "debug",
            )
            return None

        # Calculate scores and collect embeddings
        scene_data: list[tuple[int, float, NDArray[np.float32]]] = []
        tag_counter: Counter[str] = Counter()

        for scene in valid_scenes:
            scene_id = scene["scene_id"]
            record = self.storage.get_embedding(scene_id)
            if not record:
                continue

            emb = np.array(record["composite_embedding"], dtype=np.float32)

            # Get engagement score
            if scene_id in all_engagement and config.use_engagement_weighting:
                eng_data = all_engagement[scene_id]
                score = self.calculator.calculate_score(eng_data, scoring_method)
                weight = (
                    score.time_decayed_score
                    if scoring_method == EngagementScoringMethod.TIME_DECAYED
                    else score.raw_score
                )
            else:
                # Default weight for unwatched scenes
                weight = 1.0 if config.include_unwatched else 0.0

            if weight > 0:
                scene_data.append((scene_id, weight, emb))
                tag_counter.update(scene["tags"])

        if not scene_data:
            return None

        # Sort by weight and limit to max_scenes
        scene_data.sort(key=lambda x: x[1], reverse=True)
        scene_data = scene_data[: config.max_scenes]

        # Compute weighted average
        total_weight = sum(w for _, w, _ in scene_data)
        if total_weight <= 0:
            return None

        weighted_sum: NDArray[np.float32] | None = None
        for _, weight, emb in scene_data:
            if weighted_sum is None:
                weighted_sum = weight * emb
            else:
                weighted_sum += weight * emb

        if weighted_sum is None:
            return None

        # Normalize to unit vector
        profile_emb = weighted_sum / total_weight
        norm = float(np.linalg.norm(profile_emb))
        if norm > 0:
            profile_emb = profile_emb / norm

        # Get top tags
        top_tags = [tag for tag, _ in tag_counter.most_common(10)]

        return (
            profile_emb.tolist(),
            len(scene_data),
            total_weight,
            top_tags,
        )

    def build_all_performer_embeddings(
        self,
        config: PerformerEmbeddingConfig | None = None,
        scoring_method: EngagementScoringMethod = EngagementScoringMethod.BASE_WEIGHTED,
        force: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        Build embeddings for all performers.

        Args:
            config: Optional configuration for embedding generation
            scoring_method: Engagement scoring method to use
            force: If True, regenerate existing embeddings
            progress_callback: Optional progress callback

        Returns:
            Summary dict with counts
        """
        config = config or PerformerEmbeddingConfig()
        performers = self.get_all_performers()

        self.log(f"Processing {len(performers)} performers", "info")

        embedded = 0
        skipped = 0
        insufficient = 0
        errors = 0

        for i, performer in enumerate(performers):
            if progress_callback:
                progress_callback(i, len(performers))

            performer_id = performer["id"]
            performer_name = performer["name"]

            # Check if already embedded
            if not force and self.storage.has_performer_embedding(performer_id):
                self.log(f"Skipping {performer_name} (already embedded)", "debug")
                skipped += 1
                continue

            try:
                result = self.build_performer_embedding(
                    performer_id,
                    config=config,
                    scoring_method=scoring_method,
                )

                if result is None:
                    insufficient += 1
                    continue

                embedding, contributing_scenes, total_score, top_tags = result

                # Get total scene count for this performer
                scenes = self.get_performer_scenes(performer_id)

                # Store embedding
                self.storage.store_performer_embedding(
                    performer_id=performer_id,
                    embedding=embedding,
                    contributing_scenes=contributing_scenes,
                    total_engagement_score=total_score,
                    scene_count=len(scenes),
                    top_tags=json.dumps(top_tags) if top_tags else None,
                )

                embedded += 1
                self.log(
                    f"Embedded {performer_name}: {contributing_scenes} scenes, "
                    f"score {total_score:.2f}",
                    "info",
                )

            except (ValueError, KeyError, TypeError) as e:
                self.log(f"Error embedding {performer_name}: {e}", "error")
                errors += 1

        if progress_callback:
            progress_callback(len(performers), len(performers))

        stats = self.storage.get_performer_stats()

        return {
            "total_performers": len(performers),
            "embedded": embedded,
            "skipped": skipped,
            "insufficient_scenes": insufficient,
            "errors": errors,
            "storage_stats": stats,
        }

    def get_performers_for_scene(self, scene_id: int) -> list[int]:
        """Get list of performer IDs for a scene."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return []

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT performer_id FROM performers_scenes WHERE scene_id = ?",
            (scene_id,),
        )
        ids = [row["performer_id"] for row in cursor.fetchall()]
        conn.close()
        return ids

    def get_favorite_performer_embedding(
        self,
        top_n: int = 5,
    ) -> list[float] | None:
        """
        Build a combined embedding from user's favorite performers.

        Uses the weighted average of top N favorite performers' embeddings,
        weighted by their total engagement scores.

        Args:
            top_n: Number of top favorite performers to include

        Returns:
            Combined embedding or None if no favorite performers have embeddings
        """
        db_path = get_stash_db_path()
        if not db_path.exists():
            return None

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        # Get favorite performers
        cursor.execute(
            """
            SELECT id, name FROM performers
            WHERE favorite = 1
            ORDER BY name
        """
        )
        favorites = cursor.fetchall()
        conn.close()

        if not favorites:
            self.log("No favorite performers found", "debug")
            return None

        # Get embeddings for favorites
        performer_data: list[tuple[int, float, NDArray[np.float32]]] = []

        for row in favorites:
            performer_id = row["id"]
            record = self.storage.get_performer_embedding(performer_id)
            if record:
                emb = np.array(record["embedding"], dtype=np.float32)
                score = record["total_engagement_score"]
                performer_data.append((performer_id, score, emb))

        if not performer_data:
            self.log("No favorite performers have embeddings", "debug")
            return None

        # Sort by engagement and take top N
        performer_data.sort(key=lambda x: x[1], reverse=True)
        performer_data = performer_data[:top_n]

        # Compute weighted average
        total_weight = sum(w for _, w, _ in performer_data)
        if total_weight <= 0:
            # Use equal weighting if no engagement data
            total_weight = float(len(performer_data))
            performer_data = [(p, 1.0, e) for p, _, e in performer_data]

        weighted_sum: NDArray[np.float32] | None = None
        for _, weight, emb in performer_data:
            if weighted_sum is None:
                weighted_sum = (weight / total_weight) * emb
            else:
                weighted_sum += (weight / total_weight) * emb

        if weighted_sum is None:
            return None

        # Normalize
        norm = float(np.linalg.norm(weighted_sum))
        if norm > 0:
            weighted_sum = weighted_sum / norm

        self.log(
            f"Built favorite performer embedding from {len(performer_data)} performers",
            "info",
        )

        return list(weighted_sum.tolist())
