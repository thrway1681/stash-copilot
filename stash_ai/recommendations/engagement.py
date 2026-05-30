"""Engagement scoring and calculation logic."""

import math
from collections.abc import Callable
from datetime import datetime, timezone

from ..tools.database import get_readonly_connection, get_stash_db_path
from .types import (
    EngagementScore,
    EngagementScoringMethod,
    EngagementWeights,
    SceneEngagementData,
    TimeDecayConfig,
)


class EngagementCalculator:
    """
    Calculate engagement scores for scenes.

    Supports two scoring methods:
    1. Base Weighted: Canonical formula (ADR-0004): o_count*20 + replays*2 + stars*1.5
    2. Time Decayed: Base score multiplied by exponential recency decay (30-day half-life)
    """

    DEFAULT_WEIGHTS: EngagementWeights = {
        "o_count": 20.0,  # Median scene plays between o_counts
        "view_count": 2.0,  # Per replay (views beyond the first)
        "rating": 1.5,  # Per star on 5-star scale (rating100/20), only adds if rated
    }

    DEFAULT_TIME_DECAY: TimeDecayConfig = {
        "half_life_days": 30.0,
        "min_weight": 0.1,
        "min_score_threshold": 0.0,
    }

    def __init__(
        self,
        weights: EngagementWeights | None = None,
        time_decay: TimeDecayConfig | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.time_decay = time_decay or self.DEFAULT_TIME_DECAY
        self.log = log_callback or (lambda msg, level: None)

    def get_engagement(
        self,
        scene_ids: list[int] | None = None,
    ) -> dict[int, SceneEngagementData]:
        """
        Fetch engagement data for scenes.

        Args:
            scene_ids: Optional list of scene IDs to fetch. None = all engaged scenes.

        Returns:
            Dict mapping scene_id to SceneEngagementData
        """
        if scene_ids is not None and len(scene_ids) == 0:
            return {}

        db_path = get_stash_db_path()
        if not db_path.exists():
            self.log(f"Database not found at {db_path}", "warning")
            return {}

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        if scene_ids is not None:
            placeholders = ",".join("?" * len(scene_ids))
            where_clause = f"s.id IN ({placeholders})"
            params: tuple = tuple(scene_ids)
        else:
            where_clause = "(view_agg.view_count > 0 OR o_agg.o_count > 0)"
            params = ()

        cursor.execute(
            f"""
            SELECT
                s.id as scene_id,
                COALESCE(view_agg.view_count, 0) as view_count,
                COALESCE(o_agg.o_count, 0) as o_count,
                COALESCE(s.play_duration, 0) as play_duration,
                s.rating as rating,
                view_agg.last_view as last_played,
                view_agg.first_view as first_played
            FROM scenes s
            LEFT JOIN (
                SELECT
                    scene_id,
                    COUNT(*) as view_count,
                    MAX(view_date) as last_view,
                    MIN(view_date) as first_view
                FROM scenes_view_dates
                GROUP BY scene_id
            ) view_agg ON s.id = view_agg.scene_id
            LEFT JOIN (
                SELECT scene_id, COUNT(*) as o_count
                FROM scenes_o_dates
                GROUP BY scene_id
            ) o_agg ON s.id = o_agg.scene_id
            WHERE {where_clause}
            """,
            params,
        )

        results: dict[int, SceneEngagementData] = {}
        for row in cursor.fetchall():
            results[row["scene_id"]] = {
                "scene_id": row["scene_id"],
                "view_count": row["view_count"],
                "o_count": row["o_count"],
                "play_duration": row["play_duration"] or 0,
                "last_played": row["last_played"],
                "first_played": row["first_played"],
                "rating": row["rating"],  # Can be None if unrated
            }

        conn.close()
        self.log(f"Found engagement data for {len(results)} scenes", "debug")
        return results

    def get_all_scene_engagement(self) -> dict[int, SceneEngagementData]:
        """Thin alias for get_engagement(None). Fetches all engaged scenes."""
        return self.get_engagement(scene_ids=None)

    def calculate_base_score(self, data: SceneEngagementData) -> tuple[float, dict[str, float]]:
        """
        Calculate base weighted engagement score.

        Canonical formula (ADR-0004):
            score = (o_count * w_o) + (replay_count * w_v) + (stars * w_r)

        Where replay_count = max(view_count - 1, 0) (views beyond the first).
        Rating is converted to 5-star scale (rating100 / 20) and only contributes
        if the scene has been rated. Unrated scenes get 0 bonus (not penalty).
        play_duration is intentionally excluded to avoid duration bias.

        Returns:
            (score, components_dict)
        """
        o_component = data["o_count"] * self.weights["o_count"]
        # replay_count = views beyond the first one (replays indicate preference)
        replay_count = max(data["view_count"] - 1, 0)
        view_component = replay_count * self.weights["view_count"]

        # Rating: convert rating100 (0-100) to 5-star scale (0-5)
        # Only add rating bonus if scene is rated (not None and > 0)
        rating = data.get("rating")
        if rating is not None and rating > 0:
            stars = rating / 20.0  # Convert to 0-5 scale
            rating_component = stars * self.weights.get("rating", 1.5)
        else:
            rating_component = 0.0  # No penalty for unrated scenes

        score = o_component + view_component + rating_component

        components = {
            "o_count": o_component,
            "view_count": view_component,
            "rating": rating_component,
        }

        return score, components

    def calculate_time_decay_multiplier(self, last_played: str | None) -> float:
        """
        Calculate time-based decay multiplier.

        Uses exponential decay: weight = max(min_weight, 0.5^(days/half_life))
        """
        if not last_played:
            return self.time_decay["min_weight"]

        try:
            # Parse ISO datetime - handle various formats
            last_played_clean = last_played.replace("Z", "+00:00")
            # Handle datetime with or without timezone
            if "+" in last_played_clean or last_played_clean.endswith("00:00"):
                last_dt = datetime.fromisoformat(last_played_clean)
                now = datetime.now(timezone.utc)
            else:
                # No timezone info, assume local time
                last_dt = datetime.fromisoformat(last_played_clean)
                now = datetime.now()

            days_ago = (now - last_dt).total_seconds() / 86400.0

            # Exponential decay
            half_life = self.time_decay["half_life_days"]
            decay = math.pow(0.5, days_ago / half_life)

            return max(self.time_decay["min_weight"], decay)
        except (ValueError, TypeError) as e:
            self.log(f"Failed to parse last_played date '{last_played}': {e}", "debug")
            return self.time_decay["min_weight"]

    def calculate_score(
        self,
        data: SceneEngagementData,
        method: EngagementScoringMethod = EngagementScoringMethod.BASE_WEIGHTED,
    ) -> EngagementScore:
        """
        Calculate engagement score using specified method.
        """
        raw_score, components = self.calculate_base_score(data)

        if method == EngagementScoringMethod.TIME_DECAYED:
            decay_mult = self.calculate_time_decay_multiplier(data["last_played"])
            time_decayed = raw_score * decay_mult
        else:
            time_decayed = raw_score  # No decay for base method

        return EngagementScore(
            scene_id=data["scene_id"],
            raw_score=raw_score,
            time_decayed_score=time_decayed,
            components=components,
        )

    def rank(
        self,
        scene_ids: list[int] | None = None,
        method: EngagementScoringMethod = EngagementScoringMethod.BASE_WEIGHTED,
        limit: int | None = None,
    ) -> list[EngagementScore]:
        """
        Rank scenes by engagement score.

        Args:
            scene_ids: Optional list of scene IDs to rank; None = all engaged scenes.
            method: Scoring method to use.
            limit: Maximum number of results. None = return all.

        Returns:
            List of EngagementScore sorted by score descending.
        """
        all_engagement = self.get_engagement(scene_ids=scene_ids)

        if not all_engagement:
            self.log("No engagement data found", "warning")
            return []

        scores = [self.calculate_score(data, method) for data in all_engagement.values()]

        if method == EngagementScoringMethod.TIME_DECAYED:
            scores.sort(key=lambda x: x.time_decayed_score, reverse=True)
        else:
            scores.sort(key=lambda x: x.raw_score, reverse=True)

        if limit is not None:
            scores = scores[:limit]

        self.log(f"Returning {len(scores)} ranked scenes", "debug")
        return scores

    def get_top_engaged_scenes(
        self,
        limit: int = 20,
        method: EngagementScoringMethod = EngagementScoringMethod.BASE_WEIGHTED,
    ) -> list[EngagementScore]:
        """Get top N scenes by engagement score. Delegates to rank()."""
        return self.rank(scene_ids=None, method=method, limit=limit)
