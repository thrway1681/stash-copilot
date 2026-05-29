"""Recommendation generation task."""

import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from ..embeddings.storage import EmbeddingStorage
from ..recommendations.engine import RecommendationEngine
from ..recommendations.profile import UserProfileBuilder
from ..recommendations.types import (
    EngagementScoringMethod,
    FileDetails,
    PaginationInfo,
    ProfileInfo,
    RecommendationConfig,
    RecommendationMode,
    RecommendationResponse,
    SceneDetails,
)
from ..tools.database import get_readonly_connection, get_stash_db_path

if TYPE_CHECKING:
    from ..stash_client import StashClient


class RecommendationsTask:
    """
    Task for generating personalized scene recommendations.

    Workflow:
    1. Build user preference profile from engagement data
    2. Find similar scenes based on mode (discover/rewatch)
    3. Enrich results with scene details
    4. Write results to JSON file for frontend
    """

    def __init__(
        self,
        stash: "StashClient",
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        model_key: str = "siglip",
    ):
        """
        Initialize the recommendations task.

        Args:
            stash: StashClient instance for API calls
            log_callback: Optional callback for logging (message, level)
            progress_callback: Optional callback for progress (current, total)
            model_key: Embedding model key (e.g., "siglip", "openclip:ViT-H-14")
        """
        self.stash = stash
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

        self.storage = EmbeddingStorage(model_key=model_key)
        self.engine = RecommendationEngine(
            storage=self.storage,
            log_callback=self.log,
        )

    def run(
        self,
        mode: str = "discover_new",
        scoring_method: str = "base_weighted",
        limit: int = 120,
        per_page: int = 12,
        top_scenes_for_profile: int = 20,
        o_weight: float = 3.0,
        view_weight: float = 1.5,
        duration_weight: float = 1.0,
        rating_weight: float = 1.5,
        half_life_days: float = 30.0,
        min_similarity: float = 0.3,
        request_id: str = "",
        seed_scene_id: int | None = None,
        seed_weight: float = 0.3,
        engagement_weight: float = 0.6,
        session_scene_ids: list[int] | None = None,
    ) -> RecommendationResponse:
        """
        Run the recommendation task.

        Args:
            mode: "discover_new" or "rewatch"
            scoring_method: "base_weighted" or "time_decayed"
            limit: Number of recommendations (default 120 = 10 pages)
            per_page: Results per page for pagination metadata (default 12)
            top_scenes_for_profile: Scenes to use for profile
            o_weight: Weight for o_count
            view_weight: Weight for view_count
            duration_weight: Weight for play_duration (per hour)
            rating_weight: Weight for rating (per star on 5-star scale, only adds if rated)
            half_life_days: Half-life for time decay
            min_similarity: Minimum cosine similarity
            request_id: Optional request ID for frontend tracking
            seed_scene_id: Optional scene ID to boost similarity toward
            seed_weight: Weight for seed scene similarity (0-1, default 0.3)
            engagement_weight: Rewatch mode balance (0=similarity, 1=engagement, default 0.6)
            session_scene_ids: Optional list of scene IDs to use for profile
                               (overrides engagement-based profile building)

        Returns:
            RecommendationResponse with results
        """
        self.log("Starting recommendation generation", "info")
        self.progress(0, 4)

        # Parse enums
        if mode == "rewatch":
            rec_mode = RecommendationMode.REWATCH_FAVORITES
        elif mode == "o_moments":
            rec_mode = RecommendationMode.O_MOMENTS
        elif mode == "performer_preference":
            rec_mode = RecommendationMode.PERFORMER_PREFERENCE
        else:
            rec_mode = RecommendationMode.DISCOVER_NEW

        score_method = (
            EngagementScoringMethod.TIME_DECAYED
            if scoring_method == "time_decayed"
            else EngagementScoringMethod.BASE_WEIGHTED
        )

        # Calculate adaptive min_score_threshold based on half_life_days
        if score_method == EngagementScoringMethod.TIME_DECAYED:
            if half_life_days <= 7:
                min_score_threshold: float = 5.0
            elif half_life_days <= 14:
                min_score_threshold = 3.0
            elif half_life_days <= 30:
                min_score_threshold = 1.5
            else:
                min_score_threshold = 0.5
            self.log(
                f"Time decay threshold: {min_score_threshold:.1f} "
                f"(half_life={half_life_days:.0f}d)",
                "debug",
            )
        else:
            min_score_threshold = 0.0

        # Build config
        config = RecommendationConfig(
            mode=rec_mode,
            scoring_method=score_method,
            top_scenes_for_profile=top_scenes_for_profile,
            weights={
                "o_count": o_weight,
                "view_count": view_weight,
                "play_duration": duration_weight,
                "rating": rating_weight,
            },
            time_decay={
                "half_life_days": half_life_days,
                "min_weight": 0.1,
                "min_score_threshold": min_score_threshold,
            },
            limit=limit,
            per_page=per_page,
            min_similarity=min_similarity,
            seed_scene_id=seed_scene_id,
            seed_weight=seed_weight,
            engagement_weight=engagement_weight,
        )

        self.log(f"Mode: {rec_mode.value}, Scoring: {score_method.value}", "info")
        if session_scene_ids:
            self.log(f"Session mode: {len(session_scene_ids)} scenes", "info")
        if seed_scene_id:
            self.log(f"Seed scene: {seed_scene_id} (weight: {seed_weight})", "info")
        self.progress(1, 4)

        # Build user profile (skip for modes that build their own profile in engine)
        profile_modes = {RecommendationMode.O_MOMENTS, RecommendationMode.PERFORMER_PREFERENCE}
        if rec_mode in profile_modes:
            profile = None  # These modes build their own profile in engine
        else:
            builder = UserProfileBuilder(
                storage=self.storage,
                weights=config.weights,
                time_decay=config.time_decay,
                log_callback=self.log,
            )

            # Use session scene IDs if provided, otherwise use engagement-based profile
            if session_scene_ids:
                profile = builder.build_profile_from_scene_ids(session_scene_ids)
            else:
                profile = builder.build_profile(config)

        if profile is None and rec_mode not in profile_modes:
            self.log("Failed to build user profile - insufficient data", "error")
            error_profile: ProfileInfo = {
                "contributing_scenes": [],
                "total_engagement_score": 0.0,
                "scene_count": 0,
            }
            error_pagination: PaginationInfo = {
                "total_results": 0,
                "per_page": per_page,
                "total_pages": 0,
            }
            error_response: RecommendationResponse = {
                "status": "error",
                "mode": mode,
                "scoring_method": scoring_method,
                "profile": error_profile,
                "results": [],
                "pagination": error_pagination,
                "generated_at": datetime.now().isoformat(),
                "request_id": request_id,
            }
            self._save_results(error_response, request_id)
            return error_response

        self.progress(2, 4)

        # Generate recommendations
        results = self.engine.generate_recommendations(config, profile)

        self.progress(3, 4)

        # Enrich with scene details
        scene_ids = [r["scene_id"] for r in results]
        scene_details = self._get_scene_details_batch(scene_ids)

        for r in results:
            if r["scene_id"] in scene_details:
                r["scene"] = scene_details[r["scene_id"]]

        self.progress(4, 4)

        # Build response - handle special modes that build their own profiles
        if rec_mode == RecommendationMode.O_MOMENTS:
            o_stats = self.storage.get_o_moment_stats()
            profile_info: ProfileInfo = {
                "contributing_scenes": [],  # O-moments doesn't expose contributing scenes
                "total_engagement_score": 0.0,
                "scene_count": o_stats.get("scenes_with_o_moments", 0),
            }
        elif rec_mode == RecommendationMode.PERFORMER_PREFERENCE:
            perf_stats = self.storage.get_performer_stats()
            profile_info = {
                "contributing_scenes": [],  # Performer mode uses favorite performers, not scenes
                "total_engagement_score": 0.0,
                "scene_count": perf_stats.get("total_performers", 0),
            }
        elif profile:
            profile_info = {
                "contributing_scenes": profile.contributing_scenes,
                "total_engagement_score": profile.total_engagement_score,
                "scene_count": len(profile.contributing_scenes),
            }
        else:
            profile_info = {
                "contributing_scenes": [],
                "total_engagement_score": 0.0,
                "scene_count": 0,
            }

        # Calculate pagination metadata
        total_results = len(results)
        total_pages = (total_results + per_page - 1) // per_page if total_results > 0 else 0
        pagination_info: PaginationInfo = {
            "total_results": total_results,
            "per_page": per_page,
            "total_pages": total_pages,
        }

        response: RecommendationResponse = {
            "status": "complete",
            "mode": mode,
            "scoring_method": scoring_method,
            "profile": profile_info,
            "results": results,
            "pagination": pagination_info,
            "generated_at": datetime.now().isoformat(),
            "request_id": request_id,
        }

        self._save_results(response, request_id)

        self.log(f"Generated {len(results)} recommendations ({total_pages} pages)", "info")
        return response

    def _get_scene_details_batch(self, scene_ids: list[int]) -> dict[int, SceneDetails]:
        """Fetch scene details from database including file info and stats."""
        if not scene_ids:
            return {}

        db_path = get_stash_db_path()
        if not db_path.exists():
            self.log(f"Database not found at {db_path}", "warning")
            return {}

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(scene_ids))
        scene_ids_tuple = tuple(scene_ids)

        # Fetch scene base data
        cursor.execute(
            f"""
            SELECT
                s.id,
                s.title,
                s.date,
                s.rating,
                st.name as studio_name
            FROM scenes s
            LEFT JOIN studios st ON s.studio_id = st.id
            WHERE s.id IN ({placeholders})
            """,
            scene_ids_tuple,
        )

        scenes: dict[int, SceneDetails] = {}
        for row in cursor.fetchall():
            scene_id = row["id"]
            scenes[scene_id] = {
                "id": scene_id,
                "title": row["title"],
                "date": row["date"],
                "rating100": row["rating"],
                "studio": {"name": row["studio_name"]} if row["studio_name"] else None,
                "performers": [],
                "tags": [],
                "files": [],
                "play_count": 0,
                "o_counter": 0,
                "interactive": False,
            }

        # Fetch file info (duration, size, resolution, interactive)
        cursor.execute(
            f"""
            SELECT
                sf.scene_id,
                sf.file_id,
                f.basename as path,
                f.size,
                vf.duration,
                vf.height,
                vf.width,
                vf.interactive
            FROM scenes_files sf
            JOIN files f ON sf.file_id = f.id
            JOIN video_files vf ON f.id = vf.file_id
            WHERE sf.scene_id IN ({placeholders}) AND sf."primary" = 1
            """,
            scene_ids_tuple,
        )

        file_ids: dict[int, int] = {}  # scene_id -> file_id for oshash lookup
        for row in cursor.fetchall():
            scene_id = row["scene_id"]
            if scene_id in scenes:
                file_details: FileDetails = {
                    "path": row["path"],
                    "size": row["size"],
                    "duration": row["duration"],
                    "height": row["height"],
                    "width": row["width"],
                    "fingerprints": [],
                }
                scenes[scene_id]["files"].append(file_details)
                scenes[scene_id]["interactive"] = bool(row["interactive"])
                file_ids[scene_id] = row["file_id"]

        # Fetch oshash fingerprints for files
        if file_ids:
            file_id_list = list(file_ids.values())
            file_placeholders = ",".join("?" * len(file_id_list))
            cursor.execute(
                f"""
                SELECT file_id, fingerprint
                FROM files_fingerprints
                WHERE file_id IN ({file_placeholders}) AND type = 'oshash'
                """,
                tuple(file_id_list),
            )

            # Map file_id back to scene_id
            file_to_scene = {v: k for k, v in file_ids.items()}
            for row in cursor.fetchall():
                file_id = row["file_id"]
                if file_id in file_to_scene:
                    scene_id = file_to_scene[file_id]
                    if scenes[scene_id]["files"]:
                        scenes[scene_id]["files"][0]["fingerprints"].append(
                            {
                                "type": "oshash",
                                "value": row["fingerprint"],
                            }
                        )

        # Fetch play counts
        cursor.execute(
            f"""
            SELECT scene_id, COUNT(*) as play_count
            FROM scenes_view_dates
            WHERE scene_id IN ({placeholders})
            GROUP BY scene_id
            """,
            scene_ids_tuple,
        )

        for row in cursor.fetchall():
            if row["scene_id"] in scenes:
                scenes[row["scene_id"]]["play_count"] = row["play_count"]

        # Fetch o counts
        cursor.execute(
            f"""
            SELECT scene_id, COUNT(*) as o_count
            FROM scenes_o_dates
            WHERE scene_id IN ({placeholders})
            GROUP BY scene_id
            """,
            scene_ids_tuple,
        )

        for row in cursor.fetchall():
            if row["scene_id"] in scenes:
                scenes[row["scene_id"]]["o_counter"] = row["o_count"]

        # Fetch performers
        cursor.execute(
            f"""
            SELECT ps.scene_id, p.id, p.name
            FROM performers_scenes ps
            JOIN performers p ON ps.performer_id = p.id
            WHERE ps.scene_id IN ({placeholders})
            """,
            scene_ids_tuple,
        )

        for row in cursor.fetchall():
            if row["scene_id"] in scenes:
                scenes[row["scene_id"]]["performers"].append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                    }
                )

        # Fetch tags
        cursor.execute(
            f"""
            SELECT st.scene_id, t.id, t.name
            FROM scenes_tags st
            JOIN tags t ON st.tag_id = t.id
            WHERE st.scene_id IN ({placeholders})
            """,
            scene_ids_tuple,
        )

        for row in cursor.fetchall():
            if row["scene_id"] in scenes:
                scenes[row["scene_id"]]["tags"].append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                    }
                )

        conn.close()
        return scenes

    def _save_results(self, response: RecommendationResponse, request_id: str) -> None:
        """Save results to JSON file for frontend."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        filename = f"recommendations_{request_id}.json" if request_id else "recommendations.json"
        filepath = os.path.join(assets_dir, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(response, f, indent=2)
            self.log(f"Saved recommendations to {filepath}", "debug")
        except Exception as e:
            self.log(f"Failed to save recommendations: {e}", "warning")
