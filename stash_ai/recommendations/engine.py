"""Core recommendation engine."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np

from ..embeddings.storage import EmbeddingStorage
from ..tools.database import get_readonly_connection, get_stash_db_path
from .engagement import EngagementCalculator
from .performer_profile import PerformerProfileBuilder
from .profile import UserProfileBuilder
from .types import (
    EngagementScoringMethod,
    OMomentProfileInfo,
    RecommendationConfig,
    RecommendationMode,
    RecommendationResult,
    SceneDetails,
    UserPreferenceProfile,
)


def _empty_scene_details(scene_id: int = 0) -> SceneDetails:
    """Return empty SceneDetails template for results without enrichment."""
    return {
        "id": scene_id,
        "title": None,
        "date": None,
        "rating100": None,
        "studio": None,
        "performers": [],
        "tags": [],
        "files": [],
        "play_count": 0,
        "o_counter": 0,
        "interactive": False,
    }


class RecommendationEngine:
    """
    Generate personalized scene recommendations.

    Supports two modes:
    - DISCOVER_NEW: Find unwatched scenes similar to user profile
    - REWATCH_FAVORITES: Rank watched scenes by similarity + engagement
    """

    def __init__(
        self,
        storage: EmbeddingStorage | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ):
        self.storage = storage or EmbeddingStorage()
        self.log = log_callback or (lambda msg, level: None)

    def _apply_preference_model(
        self,
        profile: UserPreferenceProfile,
    ) -> UserPreferenceProfile:
        """Blend engagement profile with learned preference model if available.

        If the user has completed preference training sessions (swipe
        comparisons), the Bayesian preference model is loaded from the DB
        and blended with the engagement-based profile using a sigmoid
        schedule that transitions from engagement-dominated to
        comparison-dominated as the number of comparisons grows.

        Returns the original profile unchanged if no model exists.
        """
        try:
            from ..preferences.model import BayesianPreferenceModel

            conn = self.storage._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT preference_mean, preference_covariance_diag,
                           n_comparisons, noise_variance
                    FROM preference_model_state
                    WHERE model_key = ?
                    """,
                    (self.storage.model_key,),
                )
                row = cursor.fetchone()
            finally:
                conn.close()

            if row is None or row["n_comparisons"] == 0:
                return profile

            # Reconstruct model from persisted state
            mean = np.array(
                self.storage._unpack_embedding(row["preference_mean"]),
                dtype=np.float32,
            )
            cov_diag = np.array(
                self.storage._unpack_embedding(row["preference_covariance_diag"]),
                dtype=np.float32,
            )
            model = BayesianPreferenceModel(
                dims=len(mean),
                noise_variance=row["noise_variance"],
            )
            model.mu = mean
            model.sigma_sq = cov_diag
            model.n_comparisons = row["n_comparisons"]

            # Blend learned preferences with engagement profile
            engagement_emb = np.array(profile.profile_embedding, dtype=np.float32)
            blended = model.combine_with_engagement_profile(engagement_emb)

            self.log(
                f"Blended preference model ({model.n_comparisons} comparisons) "
                f"with engagement profile",
                "info",
            )

            return UserPreferenceProfile(
                profile_embedding=blended.tolist(),
                contributing_scenes=profile.contributing_scenes,
                total_engagement_score=profile.total_engagement_score,
                created_at=profile.created_at,
                scoring_method=profile.scoring_method,
            )
        except ImportError:
            return profile
        except Exception as e:
            self.log(f"Failed to apply preference model: {e}", "warning")
            return profile

    def generate_recommendations(
        self,
        config: RecommendationConfig,
        profile: UserPreferenceProfile | None = None,
    ) -> list[RecommendationResult]:
        """
        Generate recommendations based on config and optional profile.

        If profile is not provided, builds one from config.
        """
        # Build profile if not provided
        if profile is None:
            builder = UserProfileBuilder(
                storage=self.storage,
                weights=config.weights,
                time_decay=config.time_decay,
                log_callback=self.log,
            )
            profile = builder.build_profile(config)

            if profile is None:
                self.log("Could not build user profile", "error")
                return []

        # Blend with learned preference model if available
        profile = self._apply_preference_model(profile)

        if config.mode == RecommendationMode.DISCOVER_NEW:
            # Use cluster engine if available (and no seed scene - seeds bypass clusters)
            if config.seed_scene_id is None and self._has_taste_clusters():
                self.log("Using cluster-based recommendations (discover)", "info")
                return self._cluster_discover_new(config)
            return self._discover_new(config, profile)
        elif config.mode == RecommendationMode.O_MOMENTS:
            return self._recommend_by_o_moments(config)
        elif config.mode == RecommendationMode.PERFORMER_PREFERENCE:
            return self._recommend_by_performer_preference(config)
        else:
            if self._has_taste_clusters():
                self.log("Using cluster-based recommendations (rewatch)", "info")
                return self._cluster_rewatch(config)
            return self._rewatch_favorites(config, profile)

    def _discover_new(
        self,
        config: RecommendationConfig,
        profile: UserPreferenceProfile,
    ) -> list[RecommendationResult]:
        """
        Find unwatched scenes similar to user profile.

        If seed_scene_id is provided, boosts results that are also
        similar to the seed scene.
        """
        # Dispatch to seed-aware method if seed is provided
        if config.seed_scene_id is not None:
            return self._discover_with_seed(config, profile)

        self.log("Generating 'Discover New' recommendations", "info")

        # Exclude ALL watched scenes (not just profile scenes)
        watched_ids = self._get_watched_scene_ids()
        exclude_ids = set(profile.contributing_scenes) | watched_ids

        self.log(f"Excluding {len(exclude_ids)} scenes ({len(watched_ids)} watched)", "debug")

        # Find similar to profile embedding
        similar = self.storage.find_similar(
            query_embedding=profile.profile_embedding,
            limit=config.limit * 2,  # Over-fetch for filtering
            exclude_scene_ids=list(exclude_ids),
            min_similarity=config.min_similarity,
        )

        results: list[RecommendationResult] = []

        for sim in similar[: config.limit]:
            result: RecommendationResult = {
                "scene_id": sim.scene_id,
                "similarity_score": sim.similarity,
                "engagement_score": 0.0,  # N/A for discover mode
                "combined_score": sim.similarity,
                "scene": _empty_scene_details(sim.scene_id),
            }
            results.append(result)

        self.log(f"Found {len(results)} new scene recommendations", "info")
        return results

    def _discover_with_seed(
        self,
        config: RecommendationConfig,
        profile: UserPreferenceProfile,
    ) -> list[RecommendationResult]:
        """
        Find unwatched scenes similar to user profile, boosted by seed scene similarity.

        Combined score = (profile_similarity * (1 - seed_weight)) + (seed_similarity * seed_weight)

        This allows scene-specific recommendations that still respect user preferences.
        """
        self.log(
            f"Generating 'Discover New' with seed scene {config.seed_scene_id}",
            "info",
        )

        # Get seed scene embedding
        seed_record = self.storage.get_embedding(config.seed_scene_id)  # type: ignore[arg-type]
        if not seed_record:
            self.log(
                f"Seed scene {config.seed_scene_id} has no embedding, falling back to profile-only",
                "warning",
            )
            # Fall back to regular discover (without seed)
            config_no_seed = RecommendationConfig(
                mode=config.mode,
                scoring_method=config.scoring_method,
                top_scenes_for_profile=config.top_scenes_for_profile,
                weights=config.weights,
                time_decay=config.time_decay,
                limit=config.limit,
                min_similarity=config.min_similarity,
                seed_scene_id=None,
                seed_weight=0.0,
            )
            return self._discover_new(config_no_seed, profile)

        seed_embedding = np.array(seed_record["composite_embedding"], dtype=np.float32)
        profile_embedding = np.array(profile.profile_embedding, dtype=np.float32)

        # Exclude ALL watched scenes plus seed scene
        watched_ids = self._get_watched_scene_ids()
        exclude_ids = set(profile.contributing_scenes) | watched_ids | {config.seed_scene_id}

        self.log(
            f"Excluding {len(exclude_ids)} scenes ({len(watched_ids)} watched + seed)", "debug"
        )

        # Get all candidate scene embeddings, filtering out deleted scenes
        valid_scene_ids = self._get_valid_scene_ids()
        all_embeddings = self.storage.get_all_embeddings_validated(valid_scene_ids)

        scored_results: list[dict[str, Any]] = []
        for scene_id, embedding in all_embeddings:
            if scene_id in exclude_ids:
                continue

            emb_arr = np.array(embedding, dtype=np.float32)

            # Calculate similarities
            profile_sim = float(np.dot(profile_embedding, emb_arr))
            seed_sim = float(np.dot(seed_embedding, emb_arr))

            # Combined score: weighted blend of profile and seed similarity
            profile_weight = 1.0 - config.seed_weight
            combined = (profile_sim * profile_weight) + (seed_sim * config.seed_weight)

            if combined >= config.min_similarity:
                scored_results.append(
                    {
                        "scene_id": scene_id,
                        "profile_similarity": profile_sim,
                        "seed_similarity": seed_sim,
                        "combined": combined,
                    }
                )

        # Sort by combined score
        scored_results.sort(key=lambda x: x["combined"], reverse=True)

        results: list[RecommendationResult] = []
        for r in scored_results[: config.limit]:
            result: RecommendationResult = {
                "scene_id": r["scene_id"],
                "similarity_score": r["profile_similarity"],
                "engagement_score": r["seed_similarity"],  # Repurpose for seed similarity
                "combined_score": r["combined"],
                "scene": _empty_scene_details(r["scene_id"]),
            }
            results.append(result)

        self.log(
            f"Found {len(results)} seed-boosted recommendations (seed_weight={config.seed_weight})",
            "info",
        )
        return results

    def _rewatch_favorites(
        self,
        config: RecommendationConfig,
        profile: UserPreferenceProfile,
    ) -> list[RecommendationResult]:
        """
        Rank watched scenes by combination of engagement and similarity.

        Combined score formula:
            combined = (similarity * (1 - engagement_weight)) + (normalized_engagement * engagement_weight)

        Where engagement_weight is configurable (default 0.6 = 60% engagement, 40% similarity)
        """
        eng_pct = int(config.engagement_weight * 100)
        sim_pct = 100 - eng_pct
        self.log(
            f"Generating 'Re-watch Favorites' recommendations "
            f"(engagement: {eng_pct}%, similarity: {sim_pct}%)",
            "info",
        )

        # Get engagement calculator
        calculator = EngagementCalculator(
            weights=config.weights,
            time_decay=config.time_decay,
            log_callback=self.log,
        )

        # Get all engagement data
        all_engagement = calculator.get_all_scene_engagement()

        if not all_engagement:
            self.log("No engagement data found", "warning")
            return []

        # Get embeddings for watched scenes
        watched_with_embeddings: list[tuple[int, dict[str, Any], Any]] = []
        for scene_id, eng_data in all_engagement.items():
            record = self.storage.get_embedding(scene_id)
            if record:
                score = calculator.calculate_score(eng_data, config.scoring_method)
                watched_with_embeddings.append((scene_id, record, score))  # type: ignore[arg-type]

        if not watched_with_embeddings:
            self.log("No watched scenes have embeddings", "warning")
            return []

        self.log(
            f"Found {len(watched_with_embeddings)} watched scenes with embeddings",
            "debug",
        )

        # Calculate similarities to profile
        profile_arr = np.array(profile.profile_embedding, dtype=np.float32)

        scored_results: list[dict[str, Any]] = []
        for scene_id, record, eng_score in watched_with_embeddings:  # type: ignore[assignment]
            emb = np.array(record["composite_embedding"], dtype=np.float32)  # type: ignore[index]
            similarity = float(np.dot(profile_arr, emb))  # Cosine sim (normalized)

            # Use appropriate score based on method
            engagement = (
                eng_score.time_decayed_score
                if config.scoring_method == EngagementScoringMethod.TIME_DECAYED
                else eng_score.raw_score
            )

            scored_results.append(
                {
                    "scene_id": scene_id,
                    "similarity": similarity,
                    "engagement": engagement,
                }
            )

        # Normalize engagement scores for combination
        max_eng = max(r["engagement"] for r in scored_results)
        if max_eng > 0:
            for r in scored_results:
                r["engagement_normalized"] = r["engagement"] / max_eng
        else:
            for r in scored_results:
                r["engagement_normalized"] = 0.0

        # Compute combined score using configurable engagement_weight
        eng_weight = config.engagement_weight
        sim_weight = 1.0 - eng_weight
        for r in scored_results:
            r["combined"] = r["similarity"] * sim_weight + r["engagement_normalized"] * eng_weight

        # Sort by combined score
        scored_results.sort(key=lambda x: x["combined"], reverse=True)

        results: list[RecommendationResult] = []
        for r in scored_results[: config.limit]:
            result: RecommendationResult = {
                "scene_id": r["scene_id"],
                "similarity_score": r["similarity"],
                "engagement_score": r["engagement"],
                "combined_score": r["combined"],
                "scene": _empty_scene_details(r["scene_id"]),
            }
            results.append(result)

        self.log(f"Found {len(results)} re-watch recommendations", "info")
        return results

    def _has_taste_clusters(self) -> bool:
        """Check if taste clusters have been built."""
        clusters = self.storage.get_taste_clusters(self.storage.model_key)
        return len(clusters) > 0

    def _cluster_discover_new(self, config: RecommendationConfig) -> list[RecommendationResult]:
        """Discover new scenes using cluster-based querying."""
        from stash_ai.recommendations.cluster_engine import ClusterRecommendationEngine

        cluster_engine = ClusterRecommendationEngine(
            storage=self.storage,
            log_callback=self.log,
        )

        watched = self._get_watched_scene_ids()
        return cluster_engine.get_cluster_recommendations(
            mode="discover_new",
            limit=config.limit,
            min_similarity=config.min_similarity,
            watched_scene_ids=watched,
        )

    def _cluster_rewatch(self, config: RecommendationConfig) -> list[RecommendationResult]:
        """Rewatch recommendations using cluster-based querying."""
        from stash_ai.recommendations.cluster_engine import ClusterRecommendationEngine

        cluster_engine = ClusterRecommendationEngine(
            storage=self.storage,
            log_callback=self.log,
        )

        watched = self._get_watched_scene_ids()
        return cluster_engine.get_cluster_recommendations(
            mode="rewatch",
            limit=config.limit,
            min_similarity=max(config.min_similarity, 0.3),
            watched_scene_ids=watched,
        )

    def _get_watched_scene_ids(self) -> set[int]:
        """Get set of all watched scene IDs."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return set()

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT scene_id FROM scenes_view_dates")
        ids = {row["scene_id"] for row in cursor.fetchall()}

        conn.close()
        return ids

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

    def _recommend_by_o_moments(
        self,
        config: RecommendationConfig,
    ) -> list[RecommendationResult]:
        """
        Find scenes with similar O-moment embeddings.

        This mode builds a profile from the user's O-moment embeddings
        (scenes with O markers they've engaged with) and finds other
        scenes with similar "peak moments".

        Steps:
        1. Get all O-moment embeddings from storage
        2. Weight them by engagement scores
        3. Build averaged profile embedding
        4. Find scenes with similar O-moment embeddings
        """
        self.log("Generating 'Peak Moments' recommendations", "info")

        # Get all O-moment embeddings
        all_o_moments = self.storage.get_all_o_moment_embeddings()
        if not all_o_moments:
            self.log("No O-moment embeddings found. Run 'Embed O-Moments' task first.", "warning")
            return []

        # Get scenes with O-moments
        scenes_with_o_moments = {m[0] for m in all_o_moments}

        # Get engagement calculator
        calculator = EngagementCalculator(
            weights=config.weights,
            time_decay=config.time_decay,
            log_callback=self.log,
        )

        # Get engagement data for scenes with O-moments
        all_engagement = calculator.get_all_scene_engagement()

        # Build weighted profile from O-moments
        profile_embedding, profile_info = self._build_o_moment_profile(
            all_o_moments,
            all_engagement,  # type: ignore[arg-type]
            calculator,
            config,
        )

        if profile_embedding is None:
            self.log("Could not build O-moment profile", "error")
            return []

        self.log(
            f"Built O-moment profile from {profile_info.total_moments} moments "
            f"across {len(profile_info.contributing_scenes)} scenes",
            "info",
        )

        # Exclude watched scenes if in discover mode-like behavior
        watched_ids = self._get_watched_scene_ids()
        exclude_ids = watched_ids | scenes_with_o_moments

        self.log(f"Excluding {len(exclude_ids)} scenes (watched + profile)", "debug")

        # Find scenes with similar visual content to our O-moment profile
        # Use find_similar (searches ALL scene embeddings) not find_similar_o_moments
        # This way we can recommend unwatched scenes that look similar to peak moments
        similar = self.storage.find_similar(
            query_embedding=profile_embedding,
            limit=config.limit * 2,  # Over-fetch for filtering
            exclude_scene_ids=list(exclude_ids),
            min_similarity=config.min_similarity,
        )

        results: list[RecommendationResult] = []
        for sim in similar[: config.limit]:
            result: RecommendationResult = {
                "scene_id": sim.scene_id,
                "similarity_score": sim.similarity,
                "engagement_score": 0.0,  # N/A for O-moments mode
                "combined_score": sim.similarity,
                "scene": _empty_scene_details(sim.scene_id),
            }
            results.append(result)

        self.log(f"Found {len(results)} peak moment recommendations", "info")
        return results

    def _build_o_moment_profile(
        self,
        all_o_moments: list[tuple[int, int, list[float]]],
        all_engagement: dict[int, dict[str, Any]],
        calculator: EngagementCalculator,
        config: RecommendationConfig,
    ) -> tuple[list[float] | None, OMomentProfileInfo]:
        """
        Build weighted profile from O-moment embeddings.

        Args:
            all_o_moments: List of (scene_id, marker_id, embedding) tuples
            all_engagement: Engagement data by scene_id
            calculator: EngagementCalculator for scoring
            config: Recommendation config

        Returns:
            Tuple of (profile_embedding, profile_info)
        """
        # Calculate engagement scores for scenes with O-moments
        scene_scores: dict[int, float] = {}
        for scene_id, _marker_id, _embedding in all_o_moments:
            if scene_id not in scene_scores and scene_id in all_engagement:
                eng_data = all_engagement[scene_id]
                score = calculator.calculate_score(eng_data, config.scoring_method)  # type: ignore[arg-type]
                # Use appropriate score based on method
                scene_scores[scene_id] = (
                    score.time_decayed_score
                    if config.scoring_method == EngagementScoringMethod.TIME_DECAYED
                    else score.raw_score
                )

        if not scene_scores:
            # No engagement data for O-moment scenes, use equal weighting
            self.log("No engagement data for O-moment scenes, using equal weights", "debug")
            scene_scores = {m[0]: 1.0 for m in all_o_moments}

        # Weight and average embeddings
        weighted_sum = None
        total_weight = 0.0
        contributing_markers: list[int] = []
        contributing_scenes: set[int] = set()

        for scene_id, marker_id, embedding in all_o_moments:
            weight = scene_scores.get(scene_id, 0.0)
            if weight <= 0:
                continue

            emb_arr = np.array(embedding, dtype=np.float32)
            if weighted_sum is None:
                weighted_sum = weight * emb_arr
            else:
                weighted_sum += weight * emb_arr

            total_weight += weight
            contributing_markers.append(marker_id)
            contributing_scenes.add(scene_id)

        if weighted_sum is None or total_weight == 0:
            return None, OMomentProfileInfo(
                contributing_moments=[],
                contributing_scenes=[],
                total_moments=0,
                created_at=datetime.now().isoformat(),
            )

        # Average and normalize
        profile = weighted_sum / total_weight
        norm = float(np.linalg.norm(profile))
        if norm > 0:
            profile = profile / norm

        return profile.tolist(), OMomentProfileInfo(
            contributing_moments=contributing_markers,
            contributing_scenes=list(contributing_scenes),
            total_moments=len(contributing_markers),
            created_at=datetime.now().isoformat(),
        )

    def _recommend_by_performer_preference(
        self,
        config: RecommendationConfig,
    ) -> list[RecommendationResult]:
        """
        Find scenes similar to favorite performer embeddings.

        This mode builds a profile from the user's favorite performers'
        embeddings and finds scenes visually similar to those performers.

        Steps:
        1. Get favorite performers with embeddings
        2. Weight by user engagement with each performer
        3. Build averaged performer profile embedding
        4. Find unwatched scenes similar to that profile
        """
        self.log("Generating 'Performer Preference' recommendations", "info")

        # Get performer profile builder
        performer_builder = PerformerProfileBuilder(
            storage=self.storage,
            log_callback=self.log,
        )

        # Get favorite performer embedding
        favorite_embedding = performer_builder.get_favorite_performer_embedding(top_n=5)

        if favorite_embedding is None:
            self.log(
                "No favorite performers with embeddings found. "
                "Run 'Embed All Performers' task first and mark performers as favorites.",
                "warning",
            )
            return []

        profile_arr = np.array(favorite_embedding, dtype=np.float32)

        # Get seed embedding if provided (to blend with performer preference)
        seed_embedding = None
        if config.seed_scene_id is not None:
            seed_record = self.storage.get_embedding(config.seed_scene_id)
            if seed_record:
                seed_embedding = np.array(seed_record["composite_embedding"], dtype=np.float32)
                self.log(
                    f"Blending with seed scene {config.seed_scene_id} "
                    f"(weight: {config.seed_weight})",
                    "debug",
                )

        # Exclude watched scenes
        watched_ids = self._get_watched_scene_ids()
        exclude_ids = watched_ids

        if config.seed_scene_id:
            exclude_ids = exclude_ids | {config.seed_scene_id}

        self.log(f"Excluding {len(exclude_ids)} watched scenes", "debug")

        # Get all candidate embeddings
        valid_scene_ids = self._get_valid_scene_ids()
        all_embeddings = self.storage.get_all_embeddings_validated(valid_scene_ids)

        scored_results: list[dict[str, Any]] = []

        for scene_id, embedding in all_embeddings:
            if scene_id in exclude_ids:
                continue

            emb_arr = np.array(embedding, dtype=np.float32)

            # Calculate similarity to performer profile
            performer_sim = float(np.dot(profile_arr, emb_arr))

            # Blend with seed if provided
            if seed_embedding is not None:
                seed_sim = float(np.dot(seed_embedding, emb_arr))
                profile_weight = 1.0 - config.seed_weight
                combined = (performer_sim * profile_weight) + (seed_sim * config.seed_weight)
            else:
                combined = performer_sim

            if combined >= config.min_similarity:
                scored_results.append(
                    {
                        "scene_id": scene_id,
                        "performer_similarity": performer_sim,
                        "combined": combined,
                    }
                )

        # Sort by combined score
        scored_results.sort(key=lambda x: x["combined"], reverse=True)

        results: list[RecommendationResult] = []
        for r in scored_results[: config.limit]:
            result: RecommendationResult = {
                "scene_id": r["scene_id"],
                "similarity_score": r["performer_similarity"],
                "engagement_score": 0.0,  # N/A for performer mode
                "combined_score": r["combined"],
                "scene": _empty_scene_details(r["scene_id"]),
            }
            results.append(result)

        self.log(
            f"Found {len(results)} performer-based recommendations",
            "info",
        )
        return results
