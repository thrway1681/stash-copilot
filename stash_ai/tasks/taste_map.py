"""Build Taste Map task - orchestrates clustering, UMAP projection, and auto-labeling."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..stash_client import StashClient

from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.embeddings.tag_vocabulary import TagVocabulary
from stash_ai.recommendations.clusters import (
    build_taste_profile,
    compute_umap_projection,
)
from stash_ai.recommendations.engagement import EngagementCalculator
from stash_ai.recommendations.types import (
    EngagementScoringMethod,
    TasteMapResponse,
    TasteMapSceneData,
)


class TasteMapTask:
    """Task for building the taste map visualization data."""

    def __init__(
        self,
        stash: StashClient,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        model_key: str = "siglip",
    ) -> None:
        self.stash = stash
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)
        self.model_key = model_key
        self.storage = EmbeddingStorage(model_key=model_key)

    def run(
        self,
        request_id: str = "",
        weights: dict[str, float] | None = None,
        time_decay: dict[str, float] | None = None,
        scoring_method: str = "base_weighted",
        num_clusters: int | None = None,
    ) -> TasteMapResponse:
        """Run the taste map pipeline.

        Steps:
            1. Load ALL embedded scenes + engagement scores
            2. Ensure tag embeddings exist
            3. Cluster ALL scenes (K-Means + silhouette)
            4. UMAP projection
            5. Save results to JSON

        Args:
            request_id: Unique ID for result file.
            weights: Engagement weight overrides.
            time_decay: Time decay config.
            scoring_method: 'base_weighted' or 'time_decayed'.
            num_clusters: Fixed number of clusters (None = auto-detect).

        Returns:
            Complete taste map response.
        """
        try:
            total_steps = 5
            self.progress(0, total_steps)

            # Step 1: Load ALL embedded scenes + engagement scores
            self.log("Step 1/5: Loading all scenes and engagement data...", "info")
            scene_ids, embeddings, engagement_scores = self._load_scene_data(
                weights, time_decay, scoring_method
            )

            if len(scene_ids) == 0:
                self.log("No embedded scenes found", "error")
                response: TasteMapResponse = {
                    "status": "error",
                    "optimal_k": 0,
                    "silhouette_score": 0.0,
                    "clusters": [],
                    "scenes": [],
                    "error": "No embedded scenes found. Run 'Embed All Scenes' first.",
                }
                self._save_results(response, request_id)
                return response

            engaged_count = len(engagement_scores)
            self.log(
                f"Loaded {len(scene_ids)} scenes ({engaged_count} with engagement data)", "info"
            )
            self.progress(1, total_steps)

            # Step 2: Ensure tag embeddings
            self.log("Step 2/5: Preparing tag vocabulary embeddings...", "info")
            tag_vocab = TagVocabulary(
                storage=self.storage,
                model_key=self.model_key,
                log_callback=self.log,
            )
            stash_tags = self._get_stash_tags()
            tag_vocab.ensure_embeddings(stash_tags=stash_tags)
            self.progress(2, total_steps)

            # Step 3: Cluster ALL scenes
            self.log("Step 3/5: Clustering scenes...", "info")
            profile = build_taste_profile(
                scene_ids=scene_ids,
                embeddings=embeddings,
                engagement_scores=engagement_scores,
                tag_vocabulary=tag_vocab,
                model_key=self.model_key,
                log=self.log,
                num_clusters=num_clusters,
            )
            self.progress(3, total_steps)

            # Step 4: UMAP projection (supervised — uses cluster labels)
            self.log("Step 4/5: Computing UMAP projection...", "info")

            # Build cluster assignment map (all scenes have assignments from K-Means)
            cluster_map: dict[int, int] = {}
            for cluster in profile.clusters:
                for sid in cluster.scene_ids:
                    cluster_map[sid] = cluster.cluster_id

            # Build per-scene label array aligned with embeddings for supervised UMAP
            umap_labels = [cluster_map[sid] for sid in scene_ids]
            umap_coords = compute_umap_projection(embeddings, labels=umap_labels, log=self.log)

            # Save coords to storage
            coords_dict: dict[int, tuple[float, float, float]] = {}
            for i, sid in enumerate(scene_ids):
                coords_dict[sid] = (
                    float(umap_coords[i][0]),
                    float(umap_coords[i][1]),
                    float(umap_coords[i][2]),
                )
            self.storage.save_umap_coords(coords_dict, cluster_map, self.model_key)
            self.progress(4, total_steps)

            # Step 5: Build response and save
            self.log("Step 5/5: Building response...", "info")

            # Save clusters to storage
            self.storage.save_taste_clusters(profile.clusters, self.model_key)

            # Build scene data for frontend
            scene_details = self._get_scene_details(scene_ids)
            scenes_data: list[TasteMapSceneData] = []
            for i, sid in enumerate(scene_ids):
                details = scene_details.get(sid, {})
                scenes_data.append(
                    {
                        "scene_id": sid,
                        "x": float(umap_coords[i][0]),
                        "y": float(umap_coords[i][1]),
                        "z": float(umap_coords[i][2]),
                        "cluster_id": cluster_map.get(sid),
                        "engagement_score": engagement_scores.get(sid, 0.0),
                        "is_profile": sid in engagement_scores,
                        "title": details.get("title"),
                        "thumbnail": self._get_thumbnail_url(sid),
                        "play_count": details.get("play_count", 0),
                        "o_counter": details.get("o_counter", 0),
                    }
                )

            # Build cluster data
            clusters_data = []
            for cluster in profile.clusters:
                # Find representative scenes (closest to centroid)
                rep_scenes = self._find_representative_scenes(cluster, embeddings, scene_ids, n=3)
                clusters_data.append(
                    {
                        "cluster_id": cluster.cluster_id,
                        "auto_label": cluster.auto_label,
                        "scene_ids": cluster.scene_ids,
                        "engagement_total": round(cluster.engagement_total, 2),
                        "engagement_share": round(cluster.engagement_share, 4),
                        "representative_scenes": rep_scenes,
                        "tag_matches": cluster.tag_matches,
                    }
                )

            response = TasteMapResponse(
                status="complete",
                optimal_k=profile.optimal_k,
                silhouette_score=round(profile.silhouette_score, 4),
                clusters=clusters_data,  # type: ignore[typeddict-item]
                scenes=scenes_data,
                error=None,
            )

            self._save_results(response, request_id)
            self.progress(5, total_steps)

            self.log(
                f"Taste map complete: {profile.optimal_k} clusters, "
                f"{len(scenes_data)} scenes, silhouette={profile.silhouette_score:.4f}",
                "info",
            )
            return response

        except Exception as e:
            self.log(f"Taste map failed: {e}", "error")
            import traceback

            self.log(traceback.format_exc(), "error")
            error_response: TasteMapResponse = {
                "status": "error",
                "optimal_k": 0,
                "silhouette_score": 0.0,
                "clusters": [],
                "scenes": [],
                "error": str(e),
            }
            self._save_results(error_response, request_id)
            return error_response

    def _load_scene_data(
        self,
        weights: dict[str, float] | None,
        time_decay: dict[str, float] | None,
        scoring_method: str,
    ) -> tuple[list[int], np.ndarray, dict[int, float]]:
        """Load ALL embedded scenes and compute engagement scores.

        Returns:
            Tuple of (scene_ids, embeddings, engagement_scores) where
            engagement_scores only contains scenes with engagement data.
        """
        # Load ALL embeddings
        all_ids = self.storage.get_embedded_scene_ids()

        embeddings_list: list[list[float]] = []
        valid_ids: list[int] = []
        for sid in all_ids:
            emb = self.storage.get_embedding(sid)
            if emb and emb.get("visual_embedding"):
                embeddings_list.append(emb["visual_embedding"])  # type: ignore[arg-type]
                valid_ids.append(sid)

        if not valid_ids:
            return [], np.array([]), {}

        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Compute engagement scores for scenes that have engagement data
        calculator = EngagementCalculator(
            weights=weights,  # type: ignore[arg-type]
            time_decay=time_decay,  # type: ignore[arg-type]
            log_callback=self.log,
        )
        method = (
            EngagementScoringMethod.TIME_DECAYED
            if scoring_method == "time_decayed"
            else EngagementScoringMethod.BASE_WEIGHTED
        )

        # Get all engaged scenes (high limit to get everything)
        scores = calculator.get_top_engaged_scenes(limit=100000, method=method)
        engagement_map: dict[int, float] = {}
        embedded_set = set(valid_ids)
        for score in scores:
            if score.scene_id in embedded_set:
                eng = (
                    score.time_decayed_score
                    if method == EngagementScoringMethod.TIME_DECAYED
                    else score.raw_score
                )
                engagement_map[score.scene_id] = eng

        return valid_ids, embeddings, engagement_map

    def _get_stash_tags(self) -> list[str]:
        """Fetch all tag names from Stash database."""
        try:
            result = self.stash.find_tags(filter={"per_page": -1}, fragment="name")
            return [t["name"] for t in result if t.get("name")]
        except Exception as e:
            self.log(f"Failed to fetch tags: {e}", "warning")
            return []

    def _get_scene_details(self, scene_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch scene details (title, play_count, o_counter) from Stash."""
        details: dict[int, dict[str, Any]] = {}
        try:
            for sid in scene_ids:
                scene = self.stash.find_scene(sid)
                if scene:
                    details[sid] = {
                        "title": scene.get("title")
                        or (scene.get("files") or [{}])[0].get("basename", f"Scene {sid}"),
                        "play_count": scene.get("play_count", 0),
                        "o_counter": scene.get("o_counter", 0),
                    }
        except Exception as e:
            self.log(f"Failed to fetch scene details: {e}", "warning")
        return details

    def _get_thumbnail_url(self, scene_id: int) -> str:
        """Get the thumbnail URL for a scene."""
        return f"/scene/{scene_id}/screenshot"

    def _find_representative_scenes(
        self,
        cluster: Any,
        all_embeddings: np.ndarray,
        all_scene_ids: list[int],
        n: int = 3,
    ) -> list[int]:
        """Find n scenes closest to the cluster centroid."""
        scene_id_to_idx = {sid: i for i, sid in enumerate(all_scene_ids)}
        distances: list[tuple[int, float]] = []

        for sid in cluster.scene_ids:
            idx = scene_id_to_idx.get(sid)
            if idx is not None:
                emb = all_embeddings[idx]
                dist = float(np.dot(cluster.centroid, emb / (np.linalg.norm(emb) + 1e-8)))
                distances.append((sid, dist))

        distances.sort(key=lambda x: x[1], reverse=True)
        return [sid for sid, _ in distances[:n]]

    def _save_results(self, response: TasteMapResponse, request_id: str) -> None:
        """Save results to JSON file for frontend polling and persistence."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Request-specific file (for polling during build)
        if request_id:
            req_filepath = os.path.join(assets_dir, f"taste_map_{request_id}.json")
            with open(req_filepath, "w") as f:
                json.dump(response, f)
            self.log(f"Results saved to taste_map_{request_id}.json", "debug")

        # Always overwrite the latest file (for persistence / auto-load)
        latest_filepath = os.path.join(assets_dir, "taste_map_latest.json")
        with open(latest_filepath, "w") as f:
            json.dump(response, f)
        self.log("Results saved to taste_map_latest.json", "debug")
