"""Cluster-based recommendation engine with proportional sampling.

Replaces single-profile cosine similarity with per-cluster querying
and weighted round-robin merging for diverse recommendations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from stash_ai.embeddings.storage import EmbeddingStorage
    from stash_ai.recommendations.types import RecommendationResult, SceneDetails


def _empty_scene_details(scene_id: int = 0) -> SceneDetails:
    """Return empty SceneDetails template."""
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


class ClusterRecommendationEngine:
    """Query recommendations per-cluster and merge with proportional sampling."""

    def __init__(
        self,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)

    def get_cluster_recommendations(
        self,
        mode: str,  # 'discover_new' | 'rewatch'
        limit: int = 120,
        min_similarity: float = 0.5,
        exclude_scene_ids: set[int] | None = None,
        watched_scene_ids: set[int] | None = None,
    ) -> list[RecommendationResult]:
        """Generate recommendations using cluster-based querying.

        Args:
            mode: 'discover_new' (unwatched) or 'rewatch' (watched only).
            limit: Max total results.
            min_similarity: Minimum cosine similarity threshold.
            exclude_scene_ids: Scene IDs to exclude from results.
            watched_scene_ids: Set of watched scene IDs (for mode filtering).

        Returns:
            Merged, deduplicated recommendation results.
        """
        model_key = self.storage.model_key
        clusters = self.storage.get_taste_clusters(model_key)

        if not clusters:
            self.log("No taste clusters found - run 'Build Taste Map' first", "warning")
            return []

        # Filter to active (non-excluded) clusters
        active_clusters = [c for c in clusters if not c["excluded"]]
        if not active_clusters:
            self.log("All clusters are excluded", "warning")
            return []

        # Calculate effective weights
        total_weight = sum(
            c.get("weight_override") or c["engagement_share"] for c in active_clusters
        )
        if total_weight <= 0:
            total_weight = 1.0

        cluster_weights: list[tuple[dict[str, Any], float]] = []
        for c in active_clusters:
            weight = (c.get("weight_override") or c["engagement_share"]) / total_weight
            cluster_weights.append((c, weight))

        self.log(
            f"Querying {len(active_clusters)} clusters "
            f"(weights: {[f'{w:.0%}' for _, w in cluster_weights]})",
            "info",
        )

        # Query each cluster
        per_cluster_results: list[tuple[dict[str, Any], float, list[RecommendationResult]]] = []
        for cluster, weight in cluster_weights:
            centroid = np.array(cluster["centroid"], dtype=np.float32)
            cluster_limit = max(10, int(limit * weight * 2))  # Over-fetch for dedup

            results = self._query_single_cluster(
                centroid=centroid,
                limit=cluster_limit,
                min_similarity=min_similarity,
                mode=mode,
                exclude_scene_ids=exclude_scene_ids or set(),
                profile_scene_ids=set(cluster["scene_ids"]),
                watched_scene_ids=watched_scene_ids or set(),
            )

            per_cluster_results.append((cluster, weight, results))
            self.log(
                f"  Cluster '{cluster['auto_label']}': {len(results)} results",
                "debug",
            )

        # Proportional merge
        merged = self._proportional_merge(per_cluster_results, limit)
        self.log(
            f"Merged {len(merged)} recommendations from {len(active_clusters)} clusters", "info"
        )

        return merged

    def _query_single_cluster(
        self,
        centroid: np.ndarray,
        limit: int,
        min_similarity: float,
        mode: str,
        exclude_scene_ids: set[int],
        profile_scene_ids: set[int],
        watched_scene_ids: set[int],
    ) -> list[RecommendationResult]:
        """Query similar scenes for a single cluster centroid."""
        # Get all similar scenes
        raw_results = self.storage.find_similar(
            query_embedding=centroid.tolist(),
            limit=limit * 3,  # Over-fetch to filter
            min_similarity=min_similarity,
        )

        results: list[RecommendationResult] = []
        for sim in raw_results:
            scene_id = sim.scene_id
            similarity = sim.similarity

            # Apply mode filter
            if mode == "discover_new":
                if scene_id in watched_scene_ids or scene_id in profile_scene_ids:
                    continue
            elif mode == "rewatch":
                if scene_id not in watched_scene_ids:
                    continue

            # Apply exclusions
            if scene_id in exclude_scene_ids:
                continue

            results.append(
                {
                    "scene_id": scene_id,
                    "similarity_score": similarity,
                    "engagement_score": 0.0,
                    "combined_score": similarity,
                    "scene": _empty_scene_details(scene_id),
                }
            )

            if len(results) >= limit:
                break

        return results

    def _proportional_merge(
        self,
        per_cluster_results: list[tuple[dict[str, Any], float, list[RecommendationResult]]],
        limit: int,
    ) -> list[RecommendationResult]:
        """Merge results from multiple clusters with proportional sampling.

        Uses weighted round-robin: if cluster A has 50% weight and cluster B has 30%
        and cluster C has 20%, every 10 results will have ~5 from A, ~3 from B, ~2 from C.
        """
        seen_ids: set[int] = set()
        merged: list[RecommendationResult] = []

        # Track position in each cluster's results
        positions = [0] * len(per_cluster_results)
        weights = [w for _, w, _ in per_cluster_results]

        # Accumulator-based round-robin
        accumulators = [0.0] * len(per_cluster_results)

        max_iterations = limit * 3  # Safety cap
        iteration = 0

        while len(merged) < limit and iteration < max_iterations:
            iteration += 1

            # Add weights to accumulators
            for i in range(len(accumulators)):
                accumulators[i] += weights[i]

            # Pick the cluster with highest accumulator
            best_idx = max(range(len(accumulators)), key=lambda i: accumulators[i])

            _cluster, _weight, results = per_cluster_results[best_idx]

            # Try to get next unseen result from this cluster
            added = False
            while positions[best_idx] < len(results):
                result = results[positions[best_idx]]
                positions[best_idx] += 1

                if result["scene_id"] not in seen_ids:
                    seen_ids.add(result["scene_id"])
                    merged.append(result)
                    added = True
                    break

            if added:
                accumulators[best_idx] -= 1.0
            else:
                # This cluster is exhausted, remove its weight
                accumulators[best_idx] = -float("inf")
                weights[best_idx] = 0.0

                # Check if all clusters are exhausted
                if all(w == 0.0 for w in weights):
                    break

        return merged
