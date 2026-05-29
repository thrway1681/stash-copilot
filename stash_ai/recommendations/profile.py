"""User preference profile generation from engagement data."""

from collections.abc import Callable
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from ..embeddings.storage import EmbeddingStorage
from ..tools.database import get_readonly_connection, get_stash_db_path
from .engagement import EngagementCalculator
from .types import (
    EngagementScore,
    EngagementScoringMethod,
    EngagementWeights,
    RecommendationConfig,
    TimeDecayConfig,
    UserPreferenceProfile,
)


class UserProfileBuilder:
    """
    Build user preference profiles from engagement-weighted scene embeddings.

    The profile is a weighted average of embeddings from top engaged scenes,
    where each scene's contribution is proportional to its engagement score.
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

    def _get_valid_scene_ids(self) -> list[int]:
        """Get list of all scene IDs that exist in Stash database."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return []

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM scenes")
        ids = [row["id"] for row in cursor.fetchall()]

        conn.close()
        return ids

    def build_profile(
        self,
        config: RecommendationConfig,
    ) -> UserPreferenceProfile | None:
        """
        Build user preference profile from top engaged scenes.

        Process:
        1. Get top N scenes by engagement score
        2. Filter to scenes with embeddings
        3. Compute weighted average embedding
        4. Normalize to unit vector

        Returns:
            UserPreferenceProfile or None if insufficient data
        """
        self.log(
            f"Building user profile with top {config.top_scenes_for_profile} scenes",
            "info",
        )

        # Step 1: Get top engaged scenes (over-fetch for filtering)
        top_scores = self.calculator.get_top_engaged_scenes(
            limit=config.top_scenes_for_profile * 2,
            method=config.scoring_method,
        )

        if not top_scores:
            self.log("No engagement data found - cannot build profile", "warning")
            return None

        # Step 2: Filter to scenes with embeddings that still exist in Stash
        valid_scene_ids = self._get_valid_scene_ids()
        embedded_scene_ids = set(self.storage.get_embedded_scene_ids_validated(valid_scene_ids))

        # Extract min_score_threshold for time-decayed filtering
        min_score_threshold: float = config.time_decay.get("min_score_threshold", 0.0)
        use_threshold = (
            config.scoring_method == EngagementScoringMethod.TIME_DECAYED
            and min_score_threshold > 0.0
        )

        valid_scores: list[EngagementScore] = []
        threshold_excluded: int = 0
        for score in top_scores:
            if score.scene_id not in embedded_scene_ids:
                continue
            # Filter out scenes whose decayed score falls below the threshold
            if use_threshold and score.time_decayed_score < min_score_threshold:
                threshold_excluded += 1
                continue
            valid_scores.append(score)
            if len(valid_scores) >= config.top_scenes_for_profile:
                break

        if threshold_excluded > 0:
            self.log(
                f"Excluded {threshold_excluded} scenes below time-decay "
                f"threshold {min_score_threshold:.2f}",
                "debug",
            )

        min_scenes_required = 3  # Minimum for meaningful profile
        if len(valid_scores) < min_scenes_required and use_threshold:
            self.log(
                f"Only {len(valid_scores)} scenes above threshold "
                f"{min_score_threshold} - falling back to unfiltered profile",
                "warning",
            )
            # Retry without threshold
            valid_scores = []
            for score in top_scores:
                if score.scene_id not in embedded_scene_ids:
                    continue
                valid_scores.append(score)
                if len(valid_scores) >= config.top_scenes_for_profile:
                    break

        if len(valid_scores) < min_scenes_required:
            self.log(
                f"Only {len(valid_scores)} scenes have embeddings - "
                f"need at least {min_scenes_required}",
                "warning",
            )
            return None

        self.log(f"Using {len(valid_scores)} scenes for profile", "info")

        # Step 3: Compute weighted average embedding
        embeddings: list[NDArray[np.float32]] = []
        weights_list: list[float] = []
        contributing_ids: list[int] = []
        total_score = 0.0

        for score in valid_scores:
            record = self.storage.get_embedding(score.scene_id)
            if not record:
                continue

            emb = np.array(record["composite_embedding"], dtype=np.float32)
            embeddings.append(emb)

            # Use time-decayed score for weighting
            weight = (
                score.time_decayed_score
                if config.scoring_method == EngagementScoringMethod.TIME_DECAYED
                else score.raw_score
            )
            weights_list.append(weight)
            contributing_ids.append(score.scene_id)
            total_score += weight

        if not embeddings:
            self.log("No embeddings found for top engaged scenes", "warning")
            return None

        # Normalize weights to sum to 1
        weights_arr = np.array(weights_list, dtype=np.float32)
        weights_arr = weights_arr / weights_arr.sum()

        # Weighted average
        stacked = np.stack(embeddings)  # Shape: (N, dims)
        profile_emb: NDArray[np.float32] = np.average(stacked, axis=0, weights=weights_arr)

        # Step 4: Normalize to unit vector
        norm = float(np.linalg.norm(profile_emb))
        if norm > 0:
            profile_emb = profile_emb / norm

        self.log(
            f"Profile built: {len(contributing_ids)} scenes, total engagement {total_score:.2f}",
            "info",
        )

        return UserPreferenceProfile(
            profile_embedding=profile_emb.tolist(),
            contributing_scenes=contributing_ids,
            total_engagement_score=total_score,
            created_at=datetime.now().isoformat(),
            scoring_method=config.scoring_method,
        )

    def build_profile_from_scene_ids(
        self,
        scene_ids: list[int],
    ) -> UserPreferenceProfile | None:
        """
        Build user preference profile from specific scene IDs.

        Used for session-based recommendations where we use scenes
        the user viewed this session rather than engagement history.

        All scenes are weighted equally (recency is implicit in session order).

        Args:
            scene_ids: List of scene IDs to build profile from

        Returns:
            UserPreferenceProfile or None if insufficient data
        """
        self.log(f"Building session profile from {len(scene_ids)} scenes", "info")

        if not scene_ids:
            self.log("No scene IDs provided for session profile", "warning")
            return None

        # Filter to scenes with embeddings that still exist in Stash
        valid_scene_ids = self._get_valid_scene_ids()
        embedded_scene_ids = set(self.storage.get_embedded_scene_ids_validated(valid_scene_ids))

        valid_ids: list[int] = []
        for scene_id in scene_ids:
            if scene_id in embedded_scene_ids:
                valid_ids.append(scene_id)

        if len(valid_ids) < 1:
            self.log(
                "No scenes have embeddings - need at least 1",
                "warning",
            )
            return None

        self.log(f"Using {len(valid_ids)} scenes with embeddings for profile", "info")

        # Compute average embedding (equal weights for session scenes)
        embeddings: list[NDArray[np.float32]] = []
        contributing_ids: list[int] = []

        for scene_id in valid_ids:
            record = self.storage.get_embedding(scene_id)
            if not record:
                continue

            emb = np.array(record["composite_embedding"], dtype=np.float32)
            embeddings.append(emb)
            contributing_ids.append(scene_id)

        if not embeddings:
            self.log("No embeddings found for session scenes", "warning")
            return None

        # Simple average (no weighting - all session scenes equally important)
        stacked = np.stack(embeddings)  # Shape: (N, dims)
        profile_emb: NDArray[np.float32] = np.mean(stacked, axis=0)

        # Normalize to unit vector
        norm = float(np.linalg.norm(profile_emb))
        if norm > 0:
            profile_emb = profile_emb / norm

        self.log(
            f"Session profile built: {len(contributing_ids)} scenes",
            "info",
        )

        return UserPreferenceProfile(
            profile_embedding=profile_emb.tolist(),
            contributing_scenes=contributing_ids,
            total_engagement_score=float(len(contributing_ids)),  # Use count as score
            created_at=datetime.now().isoformat(),
            scoring_method=EngagementScoringMethod.BASE_WEIGHTED,  # N/A for session
        )
