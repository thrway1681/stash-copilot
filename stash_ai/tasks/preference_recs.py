"""Preference-based recommendations task.

Generates recommendations purely from the Bayesian preference model
(trained via the swipe-based preference trainer), with no engagement blending.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..embeddings.storage import EmbeddingStorage
from ..preferences.model import BayesianPreferenceModel
from ..preferences.types import ConvergenceMetrics
from ..recommendations.types import FileDetails, SceneDetails
from ..tools.database import get_readonly_connection, get_stash_db_path

if TYPE_CHECKING:
    from ..stash_client import StashClient


# ---------------------------------------------------------------------------
# Response type definitions
# ---------------------------------------------------------------------------


class PreferenceRecResult:
    """A single preference-based recommendation result."""

    __slots__ = ("preference_score", "scene", "scene_id", "uncertainty")

    def __init__(
        self,
        *,
        scene_id: int,
        preference_score: float,
        uncertainty: float,
        scene: SceneDetails,
    ) -> None:
        self.scene_id = scene_id
        self.preference_score = preference_score
        self.uncertainty = uncertainty
        self.scene = scene

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "preference_score": self.preference_score,
            "uncertainty": self.uncertainty,
            "scene": dict(self.scene),
        }


class PreferenceRecsResponse:
    """Full response from the preference recommendations task."""

    __slots__ = ("model_stats", "request_id", "results", "status")

    def __init__(
        self,
        *,
        status: str,
        results: list[dict[str, Any]],
        model_stats: dict[str, Any],
        request_id: str,
    ) -> None:
        self.status = status
        self.results = results
        self.model_stats = model_stats
        self.request_id = request_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "results": self.results,
            "model_stats": self.model_stats,
            "request_id": self.request_id,
        }


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------


class PreferenceRecsTask:
    """Generate recommendations from the trained Bayesian preference model.

    This task is completely separate from engagement-based recommendations.
    It ranks scenes purely by the learned preference vector ``mu``.
    """

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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        limit: int = 24,
        mode: str = "discover",
        request_id: str = "",
    ) -> PreferenceRecsResponse:
        """Generate preference-based recommendations.

        Args:
            limit: Maximum number of results.
            mode: ``"discover"`` (exclude watched) or ``"all"``.
            request_id: Frontend polling identifier.

        Returns:
            PreferenceRecsResponse with ranked scene results.
        """
        self.log("Loading preference model...", "info")
        model = self._load_model()

        if model.n_comparisons == 0:
            self.log("No comparisons recorded yet — cannot generate recommendations", "warning")
            response = PreferenceRecsResponse(
                status="no_model",
                results=[],
                model_stats={"n_comparisons": 0, "phase": "broad", "confidence_pct": 0.0},
                request_id=request_id,
            )
            self._save_results(response, request_id)
            return response

        self.log(f"Model loaded: {model.n_comparisons} comparisons", "info")

        # Score all frame embeddings via numpy memmap (fast, uses FAISS data).
        self.log("Scoring frame embeddings...", "info")
        embeddings = self.storage.score_all_frames(model.mu)

        if embeddings is None:
            # No FAISS index — fall back to subsampled frame loading.
            self.log("No frame index found, falling back to subsampled frames...", "info")
            scene_frames = self.storage.get_all_frame_representatives()
            embeddings = {}
            for sid, frames in scene_frames.items():
                if model.n_comparisons > 0:
                    scores = frames @ model.mu
                    embeddings[sid] = frames[int(np.argmax(scores))]
                else:
                    embeddings[sid] = frames[len(frames) // 2]

        # Fall back to composite for scenes without frame embeddings.
        composite_raw = self.storage.get_all_embeddings()
        fallback_count = 0
        for sid, emb in composite_raw:
            if sid not in embeddings:
                embeddings[sid] = np.array(emb, dtype=np.float32)
                fallback_count += 1

        self.log(
            f"Loaded {len(embeddings)} scene embeddings "
            f"({len(embeddings) - fallback_count} frame-level, "
            f"{fallback_count} composite fallback)",
            "info",
        )

        if not embeddings:
            response = PreferenceRecsResponse(
                status="no_embeddings",
                results=[],
                model_stats=self._build_model_stats(model, len(embeddings)),
                request_id=request_id,
            )
            self._save_results(response, request_id)
            return response

        # Filter out watched scenes if in discover mode
        if mode == "discover":
            watched_ids = self._get_watched_scene_ids()
            candidate_embeddings = {
                sid: emb for sid, emb in embeddings.items() if sid not in watched_ids
            }
            self.log(
                f"Discover mode: {len(embeddings)} total, "
                f"{len(watched_ids)} watched, "
                f"{len(candidate_embeddings)} candidates",
                "info",
            )
        else:
            candidate_embeddings = embeddings

        if not candidate_embeddings:
            response = PreferenceRecsResponse(
                status="no_candidates",
                results=[],
                model_stats=self._build_model_stats(model, len(embeddings)),
                request_id=request_id,
            )
            self._save_results(response, request_id)
            return response

        # Score all candidates via the preference model
        self.log(f"Scoring {len(candidate_embeddings)} scenes...", "info")
        top_scenes = model.get_top_scenes(candidate_embeddings, limit=limit)

        # Normalize scores to [0, 1] via min-max across ALL scored scenes
        # (not just the top-N, for accurate normalization)
        all_scored = model.get_top_scenes(candidate_embeddings, limit=len(candidate_embeddings))
        normalized = self._normalize_scores(top_scenes, all_scored)

        # Enrich with scene details
        scene_ids = [sid for sid, _, _ in top_scenes]
        self.log(f"Enriching {len(scene_ids)} scenes with details...", "info")
        scene_details = self._get_scene_details_batch(scene_ids)

        # Build results
        results: list[dict[str, Any]] = []
        for scene_id, norm_score, uncertainty in normalized:
            if scene_id in scene_details:
                result = PreferenceRecResult(
                    scene_id=scene_id,
                    preference_score=round(norm_score, 4),
                    uncertainty=round(float(uncertainty), 4),
                    scene=scene_details[scene_id],
                )
                results.append(result.to_dict())

        model_stats = self._build_model_stats(model, len(embeddings))

        response = PreferenceRecsResponse(
            status="complete",
            results=results,
            model_stats=model_stats,
            request_id=request_id,
        )

        self._save_results(response, request_id)
        self.log(f"Generated {len(results)} preference recommendations", "info")
        return response

    # ------------------------------------------------------------------
    # Model loading (mirrors PreferenceSessionManager._load_model)
    # ------------------------------------------------------------------

    def _load_model(self) -> BayesianPreferenceModel:
        """Load preference model from DB or return empty model."""
        conn = self.storage._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT preference_mean, preference_covariance_diag,
                       n_comparisons, noise_variance, phase
                FROM preference_model_state
                WHERE model_key = ?
                """,
                (self.model_key,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is not None:
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
            return model

        # No persisted model — return empty (0 comparisons)
        return BayesianPreferenceModel(dims=768)

    # ------------------------------------------------------------------
    # Score normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(
        top_scenes: list[tuple[int, float, float]],
        all_scored: list[tuple[int, float, float]],
    ) -> list[tuple[int, float, float]]:
        """Min-max normalize preference scores to [0, 1].

        Uses the full range of all scored scenes for normalization,
        then returns only the top_scenes with normalized values.

        Args:
            top_scenes: Top-N results from model.get_top_scenes().
            all_scored: All scored scenes for computing min/max range.

        Returns:
            List of (scene_id, normalized_score, uncertainty) for top_scenes.
        """
        if not all_scored:
            return []

        all_scores = [score for _, score, _ in all_scored]
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score

        if score_range < 1e-8:
            # All scores identical — assign 0.5
            return [(sid, 0.5, unc) for sid, _, unc in top_scenes]

        return [(sid, (score - min_score) / score_range, unc) for sid, score, unc in top_scenes]

    # ------------------------------------------------------------------
    # Model stats
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model_stats(
        model: BayesianPreferenceModel,
        n_scenes: int = 0,
    ) -> dict[str, Any]:
        """Build model statistics dict for the response."""
        phase = model._infer_phase(n_scenes)
        avg_sigma = float(np.mean(np.sqrt(model.sigma_sq)))
        metrics = ConvergenceMetrics(
            avg_sigma=avg_sigma,
            max_sigma_top50=avg_sigma,  # Approximation without full ranking
            n_comparisons=model.n_comparisons,
            phase=phase,
        )
        return {
            "n_comparisons": model.n_comparisons,
            "phase": phase.value,
            "confidence_pct": metrics.confidence_pct,
        }

    # ------------------------------------------------------------------
    # Watched scene IDs
    # ------------------------------------------------------------------

    @staticmethod
    def _get_watched_scene_ids() -> set[int]:
        """Get set of all watched scene IDs from Stash DB."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return set()

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT scene_id FROM scenes_view_dates")
        ids = {row["scene_id"] for row in cursor.fetchall()}
        conn.close()
        return ids

    # ------------------------------------------------------------------
    # Scene details (reused from RecommendationsTask pattern)
    # ------------------------------------------------------------------

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

        file_ids: dict[int, int] = {}
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

        # Fetch oshash fingerprints
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
                scenes[row["scene_id"]]["performers"].append({"id": row["id"], "name": row["name"]})

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
                scenes[row["scene_id"]]["tags"].append({"id": row["id"], "name": row["name"]})

        conn.close()
        return scenes

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def _save_results(self, response: PreferenceRecsResponse, request_id: str) -> None:
        """Save results to JSON file for frontend polling."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        filename = f"preference_recs_{request_id}.json" if request_id else "preference_recs.json"
        filepath = os.path.join(assets_dir, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(response.to_dict(), f, indent=2, default=str)
            self.log(f"Saved preference recs to {filepath}", "debug")
        except (OSError, TypeError) as e:
            self.log(f"Failed to save preference recs: {e}", "warning")
