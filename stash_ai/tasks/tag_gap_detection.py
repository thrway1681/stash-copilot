"""Tag gap detection task -- finds visual content not covered by existing tags.

Compares per-frame OpenCLIP image embeddings against tag text embeddings
(both in CLIP shared space).  Frames whose best-matching tag similarity
falls below an adaptive threshold (5th percentile) are flagged as
"uncovered", indicating visual content the current tag vocabulary misses.
"""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..stash_client import StashClient

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
from typing import TypedDict

from stash_ai.embeddings.storage import (
    EmbeddingStorage,
    FrameTagCoverageRecord,
)
from stash_ai.embeddings.tag_vocabulary import TagVocabulary


class SceneCoverageSummary(TypedDict):
    """Per-scene summary of tag coverage."""

    scene_id: int
    total_frames: int
    uncovered_frames: int
    coverage_ratio: float
    top_uncovered_tags: list[dict[str, Any]]


class TagGapReport(TypedDict):
    """Full tag-gap detection report."""

    status: str
    threshold: float
    total_scenes: int
    avg_coverage: float
    flagged_scenes: int
    scenes: list[SceneCoverageSummary]
    error: str | None


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TagGapDetectionTask:
    """Detect visual content not covered by existing tags.

    For every embedded scene, compares each frame embedding against all tag
    text embeddings.  Frames whose best tag similarity falls below an
    adaptive threshold (5th percentile of all best-similarities) are
    marked as *uncovered*.
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
        self._cached_threshold: float | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        request_id: str = "",
        force: bool = False,
    ) -> TagGapReport:
        """Run the full tag-gap detection pipeline.

        Args:
            request_id: Unique ID for result file (enables frontend polling).
            force: If ``True``, re-process scenes that already have coverage
                data.  Otherwise only new scenes are processed.

        Returns:
            A :class:`TagGapReport` dict.
        """
        try:
            total_steps = 5
            self.progress(0, total_steps)

            # Step 1: Load tag embeddings
            self.log("Step 1/5: Loading tag embeddings...", "info")
            self._save_progress(request_id, 1, "Loading tag embeddings...")
            tag_matrix, tag_names = self._load_tag_embeddings()
            if tag_matrix.shape[0] == 0:
                msg = "No tag embeddings found. Ensure tags exist and embeddings are generated."
                self.log(msg, "error")
                error_report = self._error_report(msg)
                self._save_results(error_report, request_id)
                return error_report
            self.log(f"Loaded {len(tag_names)} tag embeddings", "info")
            self.progress(1, total_steps)

            # Step 2: Get scenes to process
            self.log("Step 2/5: Identifying scenes to process...", "info")
            self._save_progress(request_id, 3, "Identifying scenes to process...")
            scene_ids = self._get_scenes_to_process(force)
            if not scene_ids:
                self.log("No new scenes to process", "info")
                # Still build a report from existing data
                threshold = 0.0
                summary = self.storage.get_coverage_summary()
                if summary:
                    # Use existing threshold (re-derive from stored data)
                    threshold = self._derive_threshold_from_db()
                report = self._build_report(threshold)
                report["status"] = "complete"
                self._save_results(report, request_id)
                return report
            self.log(f"Processing {len(scene_ids)} scenes", "info")
            self.progress(2, total_steps)

            # Step 3: Compute per-frame similarities (bulk of the work: 5-95%)
            self.log("Step 3/5: Computing frame-tag similarities...", "info")
            self._save_progress(
                request_id,
                5,
                f"Computing similarities for {len(scene_ids)} scenes...",
                scenes_total=len(scene_ids),
            )
            all_best_sims = self._compute_all_similarities(
                scene_ids, tag_matrix, tag_names, request_id
            )
            self.log(
                f"Computed similarities for {len(all_best_sims)} frames "
                f"across {len(scene_ids)} scenes",
                "info",
            )
            self.progress(3, total_steps)

            # Step 4: Compute adaptive threshold and update coverage flags
            self.log("Step 4/5: Computing adaptive threshold...", "info")
            self._save_progress(request_id, 96, "Computing adaptive threshold...")
            threshold = self._compute_threshold(all_best_sims)
            updated = self.storage.update_coverage_threshold(threshold)
            self.log(
                f"Threshold: {threshold:.4f} (5th percentile), updated {updated} rows",
                "info",
            )
            self.progress(4, total_steps)

            # Step 5: Build report
            self.log("Step 5/5: Building report...", "info")
            self._save_progress(request_id, 98, "Building report...")
            report = self._build_report(threshold)
            report["status"] = "complete"
            self._save_results(report, request_id)
            self.progress(5, total_steps)

            self.log(
                f"Tag gap detection complete: {report['total_scenes']} scenes, "
                f"avg coverage {report['avg_coverage']:.1%}, "
                f"{report['flagged_scenes']} flagged",
                "info",
            )
            return report

        except Exception as e:
            self.log(f"Tag gap detection failed: {e}", "error")
            self.log(traceback.format_exc(), "error")
            error_report = self._error_report(str(e))
            self._save_results(error_report, request_id)
            return error_report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_tag_embeddings(self) -> tuple[NDArray[np.float32], list[str]]:
        """Load (and ensure) all tag embeddings.

        Returns:
            (tag_matrix, tag_names) where tag_matrix is an ``(N, dims)``
            L2-normalised matrix and tag_names is the corresponding list
            of tag texts.
        """
        # Ensure tag vocabulary embeddings exist
        tag_vocab = TagVocabulary(
            storage=self.storage,
            model_key=self.model_key,
            log_callback=self.log,
        )

        # Fetch stash tags to include user's own tags
        stash_tags: list[str] = []
        try:
            result = self.stash.call_GQL("""query { allTags { id name } }""")
            if result and "allTags" in result:
                stash_tags = [t["name"] for t in result["allTags"] if t.get("name")]
        except Exception as e:
            self.log(f"Failed to fetch stash tags: {e}", "warning")

        tag_vocab.ensure_embeddings(stash_tags=stash_tags)

        # Load all tag embeddings into a matrix
        raw_entries = self.storage.get_all_tag_embeddings(self.model_key)
        if not raw_entries:
            return np.array([], dtype=np.float32).reshape(0, 0), []

        tag_names: list[str] = []
        vectors: list[list[float]] = []
        for entry in raw_entries:
            tag_names.append(entry["text"])
            vectors.append(entry["embedding"])

        tag_matrix = np.array(vectors, dtype=np.float32)

        # L2 normalise rows
        norms = np.linalg.norm(tag_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        tag_matrix = tag_matrix / norms

        return tag_matrix, tag_names

    def _get_scenes_to_process(self, force: bool) -> list[int]:
        """Return scene IDs that have frame embeddings but need coverage analysis.

        Args:
            force: If ``True``, return all scenes with frame embeddings.
                Otherwise exclude scenes that already have coverage data.

        Returns:
            Sorted list of scene IDs to process.
        """
        conn = self.storage._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT scene_id FROM frame_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        all_scene_ids: set[int] = {row["scene_id"] for row in cursor.fetchall()}
        conn.close()

        if force:
            return sorted(all_scene_ids)

        already_done = set(self.storage.get_scenes_with_tag_coverage())
        return sorted(all_scene_ids - already_done)

    def _compute_all_similarities(
        self,
        scene_ids: list[int],
        tag_matrix: NDArray[np.float32],
        tag_names: list[str],
        request_id: str = "",
    ) -> list[float]:
        """Compute per-frame best-tag similarity for every scene.

        For each frame, records the best-matching tag and its similarity.
        Saves ``FrameTagCoverageRecord`` rows (with ``is_covered=False``
        as a placeholder -- the threshold step will update this).

        Args:
            scene_ids: Scenes to process.
            tag_matrix: ``(T, dims)`` normalised tag embedding matrix.
            tag_names: Tag names aligned with ``tag_matrix`` rows.
            request_id: For writing intermediate progress updates.

        Returns:
            Flat list of every frame's best similarity score
            (used for threshold computation).
        """
        all_best_sims: list[float] = []
        total = len(scene_ids)

        for idx, scene_id in enumerate(scene_ids):
            frame_matrix = self.storage.get_scene_frames(scene_id)
            if frame_matrix is None or frame_matrix.shape[0] == 0:
                continue

            # L2 normalise frame embeddings
            norms = np.linalg.norm(frame_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            frame_matrix = frame_matrix / norms

            # Cosine similarity: (num_frames, num_tags)
            sim_matrix: NDArray[np.float32] = frame_matrix @ tag_matrix.T

            best_indices = np.argmax(sim_matrix, axis=1)
            best_sims = sim_matrix[np.arange(sim_matrix.shape[0]), best_indices]

            # Build coverage records for this scene
            records: list[FrameTagCoverageRecord] = []
            for frame_idx in range(frame_matrix.shape[0]):
                best_sim = float(best_sims[frame_idx])
                best_tag = tag_names[int(best_indices[frame_idx])]
                all_best_sims.append(best_sim)
                records.append(
                    FrameTagCoverageRecord(
                        scene_id=scene_id,
                        frame_index=frame_idx,
                        model_key=self.model_key,
                        best_tag=best_tag,
                        best_similarity=best_sim,
                        is_covered=False,  # placeholder; threshold step updates
                    )
                )

            self.storage.save_frame_tag_coverage_batch(records)

            # Progress logging and frontend progress update every 10 scenes
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                self.log(
                    f"  Processed {idx + 1}/{total} scenes ({len(all_best_sims)} frames so far)",
                    "info",
                )
                # Map scene progress linearly into 5%-95% range
                scene_pct = 5 + int(((idx + 1) / total) * 90)
                self._save_progress(
                    request_id,
                    scene_pct,
                    f"Processing scene {idx + 1} / {total}...",
                    scenes_done=idx + 1,
                    scenes_total=total,
                )

        return all_best_sims

    def _compute_threshold(self, all_best_sims: list[float]) -> float:
        """Compute the adaptive coverage threshold.

        Returns:
            The 5th percentile of all best-similarity scores.
        """
        if not all_best_sims:
            return 0.0
        threshold = float(np.percentile(all_best_sims, 5))

        # Cache the threshold for fast retrieval
        self._cached_threshold = threshold
        conn = self.storage._get_connection()
        cache_key = f"tag_gap_threshold_{self.model_key}"
        conn.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            (cache_key, str(threshold)),
        )
        conn.commit()
        conn.close()

        return threshold

    def _derive_threshold_from_db(self) -> float:
        """Get the 5th-percentile threshold, using cache when available.

        Uses a three-tier caching strategy:
        1. In-memory cache (fastest)
        2. Database-stored threshold (fast)
        3. Compute from scratch (slow - loads all similarities)
        """
        # Check in-memory cache first
        if self._cached_threshold is not None:
            return self._cached_threshold

        conn = self.storage._get_connection()
        cursor = conn.cursor()

        # Check for stored threshold in schema_info
        cache_key = f"tag_gap_threshold_{self.model_key}"
        cursor.execute(
            "SELECT value FROM schema_info WHERE key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row:
            self._cached_threshold = float(row["value"])
            conn.close()
            return self._cached_threshold

        # Compute from scratch (slow)
        cursor.execute(
            "SELECT best_similarity FROM frame_tag_coverage WHERE model_key = ?",
            (self.model_key,),
        )
        sims = [row["best_similarity"] for row in cursor.fetchall()]
        if not sims:
            conn.close()
            return 0.0

        threshold = float(np.percentile(sims, 5))

        # Store for future use
        cursor.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            (cache_key, str(threshold)),
        )
        conn.commit()
        conn.close()

        self._cached_threshold = threshold
        return threshold

    def _build_report(self, threshold: float) -> TagGapReport:
        """Build the tag-gap report from stored coverage data.

        Args:
            threshold: The similarity threshold used for coverage.

        Returns:
            Populated :class:`TagGapReport` (without ``status`` set).
        """
        summary_rows = self.storage.get_coverage_summary()

        scenes: list[SceneCoverageSummary] = []
        total_coverage = 0.0

        for row in summary_rows:
            scene_id: int = row["scene_id"]
            total_frames: int = row["total_frames"]
            uncovered_frames: int = row["uncovered_frames"]
            coverage_ratio: float = row["coverage_ratio"]
            total_coverage += coverage_ratio

            # Get per-frame coverage to aggregate uncovered tags
            frame_rows = self.storage.get_scene_tag_coverage(scene_id)
            uncovered_tag_counts: dict[str, dict[str, Any]] = {}
            for fr in frame_rows:
                if not fr["is_covered"]:
                    tag = fr["best_tag"]
                    if tag not in uncovered_tag_counts:
                        uncovered_tag_counts[tag] = {
                            "tag": tag,
                            "frame_count": 0,
                            "total_similarity": 0.0,
                        }
                    uncovered_tag_counts[tag]["frame_count"] += 1
                    uncovered_tag_counts[tag]["total_similarity"] += fr["best_similarity"]

            # Compute average similarity and sort by frame count
            top_uncovered: list[dict[str, Any]] = []
            for info in uncovered_tag_counts.values():
                count: int = info["frame_count"]
                top_uncovered.append(
                    {
                        "tag": info["tag"],
                        "frame_count": count,
                        "avg_similarity": round(info["total_similarity"] / count, 4),
                    }
                )

            top_uncovered.sort(key=lambda x: x["frame_count"], reverse=True)
            top_uncovered = top_uncovered[:5]

            scenes.append(
                SceneCoverageSummary(
                    scene_id=scene_id,
                    total_frames=total_frames,
                    uncovered_frames=uncovered_frames,
                    coverage_ratio=round(coverage_ratio, 4),
                    top_uncovered_tags=top_uncovered,
                )
            )

        total_scenes = len(scenes)
        avg_coverage = round(total_coverage / total_scenes, 4) if total_scenes else 0.0
        flagged_scenes = sum(1 for s in scenes if s["coverage_ratio"] < 0.75)

        return TagGapReport(
            status="",
            threshold=round(threshold, 4),
            total_scenes=total_scenes,
            avg_coverage=avg_coverage,
            flagged_scenes=flagged_scenes,
            scenes=scenes,
            error=None,
        )

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def _save_results(self, report: TagGapReport, request_id: str) -> None:
        """Save report to assets/ as JSON for frontend polling."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        if request_id:
            with open(os.path.join(assets_dir, f"tag_gaps_{request_id}.json"), "w") as f:
                json.dump(report, f)

        with open(os.path.join(assets_dir, "tag_gaps_latest.json"), "w") as f:
            json.dump(report, f)

    def _save_progress(
        self,
        request_id: str,
        progress_pct: int,
        message: str,
        scenes_done: int = 0,
        scenes_total: int = 0,
    ) -> None:
        """Write an intermediate progress update for frontend polling.

        The frontend polls the same JSON file used for final results.
        While status is ``"processing"`` the UI shows a progress bar.

        Args:
            request_id: Unique request identifier for the result file.
            progress_pct: Direct percentage (0-100) for the progress bar.
            message: Human-readable status message.
            scenes_done: Number of scenes processed so far.
            scenes_total: Total scenes to process.
        """
        if not request_id:
            return

        data: dict[str, Any] = {
            "status": "processing",
            "progress": progress_pct,
            "status_message": message,
            "scenes_done": scenes_done,
            "scenes_total": scenes_total,
        }

        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        with open(os.path.join(assets_dir, f"tag_gaps_{request_id}.json"), "w") as f:
            json.dump(data, f)

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_report(message: str) -> TagGapReport:
        """Create an error report."""
        return TagGapReport(
            status="error",
            threshold=0.0,
            total_scenes=0,
            avg_coverage=0.0,
            flagged_scenes=0,
            scenes=[],
            error=message,
        )

    # ------------------------------------------------------------------
    # Per-scene detail & cross-scene similarity (for sidebar UI)
    # ------------------------------------------------------------------

    def find_similar_uncovered(
        self,
        scene_id: int,
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Find scenes with similar uncovered (untagged) content.

        Averages uncovered frame embeddings for the query scene into a
        single vector, then compares against the averaged uncovered
        embeddings of every other scene that has tag-coverage data.

        Args:
            scene_id: The query scene.
            limit: Maximum number of results to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of ``{"scene_id": int, "similarity": float}`` dicts
            sorted by descending similarity.  Empty if the query scene
            has no uncovered frames.
        """
        # Build query vector from uncovered frames
        query_embeddings: NDArray[np.float32] = self.storage.get_uncovered_frame_embeddings(
            scene_id
        )
        if query_embeddings.ndim < 2 or query_embeddings.shape[0] == 0:
            return []

        query_vec: NDArray[np.float32] = np.mean(query_embeddings, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(query_vec))
        if norm < 1e-8:
            return []
        query_vec = query_vec / norm

        # Compare against all other scenes with coverage data
        candidate_scene_ids = self.storage.get_scenes_with_tag_coverage()
        results: list[dict[str, Any]] = []

        for candidate_id in candidate_scene_ids:
            if candidate_id == scene_id:
                continue

            cand_embeddings: NDArray[np.float32] = self.storage.get_uncovered_frame_embeddings(
                candidate_id
            )
            if cand_embeddings.ndim < 2 or cand_embeddings.shape[0] == 0:
                continue

            cand_vec: NDArray[np.float32] = np.mean(cand_embeddings, axis=0).astype(np.float32)
            cand_norm = float(np.linalg.norm(cand_vec))
            if cand_norm < 1e-8:
                continue
            cand_vec = cand_vec / cand_norm

            similarity = float(np.dot(query_vec, cand_vec))
            if similarity >= min_similarity:
                results.append(
                    {
                        "scene_id": candidate_id,
                        "similarity": round(similarity, 4),
                    }
                )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def _get_scene_tag_names(self, scene_id: int) -> set[str]:
        """Fetch the current tag names for a scene from Stash.

        Args:
            scene_id: The scene ID to query.

        Returns:
            Set of lowercase tag names currently on the scene.
        """
        if not self.stash:
            return set()

        try:
            result = self.stash.call_GQL(
                """
                query FindScene($id: ID!) {
                    findScene(id: $id) {
                        tags {
                            name
                        }
                    }
                }
                """,
                {"id": str(scene_id)},
            )
            if result and "findScene" in result and result["findScene"]:
                return {t["name"].lower() for t in result["findScene"].get("tags", [])}
        except Exception:
            pass
        return set()

    def _calculate_scene_tag_coverage(
        self,
        scene_id: int,
        scene_tags: set[str],
        threshold: float,
    ) -> dict[str, Any]:
        """Calculate coverage using only the scene's assigned tags.

        Args:
            scene_id: The scene to analyze.
            scene_tags: Set of lowercase tag names assigned to the scene.
            threshold: The similarity threshold for "covered".

        Returns:
            Dict with scene-specific coverage stats:
            - covered_frames: Frames covered by scene's tags
            - coverage_ratio: Ratio of frames covered
            - tag_count: Number of tags on the scene
        """
        if not scene_tags:
            return {
                "covered_frames": 0,
                "coverage_ratio": 0.0,
                "tag_count": 0,
            }

        # Get tag embeddings for scene's tags
        scene_tag_embeddings: list[tuple[str, NDArray[np.float32]]] = []
        for tag in scene_tags:
            embedding = self.storage.get_tag_embedding(tag, self.model_key)
            if embedding is not None:
                scene_tag_embeddings.append((tag, np.array(embedding, dtype=np.float32)))

        if not scene_tag_embeddings:
            return {
                "covered_frames": 0,
                "coverage_ratio": 0.0,
                "tag_count": len(scene_tags),
            }

        # Build tag embedding matrix
        tag_matrix = np.stack([emb for _, emb in scene_tag_embeddings])
        # Normalize rows
        norms = np.linalg.norm(tag_matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        tag_matrix = tag_matrix / norms

        # Get frame embeddings for this scene
        frame_embeddings = self.storage.get_scene_frame_embeddings(scene_id)
        if frame_embeddings.ndim < 2 or frame_embeddings.shape[0] == 0:
            return {
                "covered_frames": 0,
                "coverage_ratio": 0.0,
                "tag_count": len(scene_tag_embeddings),
            }

        # Compute similarities: (num_frames, num_scene_tags)
        similarities = np.dot(frame_embeddings, tag_matrix.T)

        # For each frame, get best match among scene tags
        best_sims = np.max(similarities, axis=1)

        # Count frames above threshold
        covered_count = int((best_sims >= threshold).sum())
        total_frames = frame_embeddings.shape[0]

        return {
            "covered_frames": covered_count,
            "coverage_ratio": round(covered_count / total_frames, 4) if total_frames > 0 else 0.0,
            "tag_count": len(scene_tag_embeddings),
        }

    def get_scene_gaps_detail(self, scene_id: int) -> dict[str, Any]:
        """Return full tag-gap detail for a single scene (sidebar use).

        Combines coverage statistics, per-frame uncovered details, and
        aggregated nearest-tag information into one response.

        Fetches the scene's current tags from Stash and filters them out
        of the nearest_tags list, so only truly missing tags are shown.

        Args:
            scene_id: The scene to query.

        Returns:
            Dict with ``has_data: False`` if no coverage data exists,
            otherwise a dict containing frame-level and tag-level detail.
        """
        frame_rows: list[FrameTagCoverageRecord] = self.storage.get_scene_tag_coverage(scene_id)
        if not frame_rows:
            return {"has_data": False}

        # Get scene's current tags to filter from suggestions
        scene_tags: set[str] = self._get_scene_tag_names(scene_id)

        # Get threshold FIRST to ensure both Coverage and Scene Tags Only
        # use the exact same threshold (prevents threshold mismatch bugs)
        threshold = self._derive_threshold_from_db()

        total_frames = len(frame_rows)
        # Recalculate coverage using current threshold (not stored is_covered)
        # This ensures consistency with scene_tag_coverage calculation
        covered_frames = sum(1 for fr in frame_rows if fr["best_similarity"] >= threshold)
        uncovered_frames = total_frames - covered_frames
        coverage_ratio = round(covered_frames / total_frames if total_frames > 0 else 0.0, 4)

        # Per-frame uncovered detail
        uncovered_frame_list: list[dict[str, Any]] = []
        # Aggregate uncovered frames by best_tag
        tag_agg: dict[str, dict[str, Any]] = {}

        for fr in frame_rows:
            # Use threshold-based check (not stored is_covered flag)
            # to ensure consistency with coverage_ratio calculation
            if fr["best_similarity"] >= threshold:
                continue

            frame_index: int = fr["frame_index"]
            best_tag: str = fr["best_tag"]
            best_similarity: float = fr["best_similarity"]

            uncovered_frame_list.append(
                {
                    "frame_index": frame_index,
                    "timestamp": float(frame_index),  # 1 fps assumption
                    "best_tag": best_tag,
                    "best_similarity": round(best_similarity, 4),
                }
            )

            # Skip tags already on the scene
            if best_tag.lower() in scene_tags:
                continue

            if best_tag not in tag_agg:
                tag_agg[best_tag] = {
                    "tag": best_tag,
                    "total_similarity": 0.0,
                    "frame_count": 0,
                }
            tag_agg[best_tag]["frame_count"] += 1
            tag_agg[best_tag]["total_similarity"] += best_similarity

        # Get actual Stash tag names (filter out curated phrases)
        stash_tag_names: set[str] = self.storage.get_stash_tag_names(self.model_key)

        # Build nearest_tags: top 10 by frame count (only real Stash tags)
        nearest_tags: list[dict[str, Any]] = []
        for info in tag_agg.values():
            tag_name: str = info["tag"]
            # Skip curated phrases - only show actual Stash tags
            if tag_name.lower() not in stash_tag_names:
                continue
            count: int = info["frame_count"]
            nearest_tags.append(
                {
                    "tag": tag_name,
                    "avg_similarity": round(info["total_similarity"] / count, 4),
                    "frame_count": count,
                }
            )
        nearest_tags.sort(key=lambda x: x["frame_count"], reverse=True)
        nearest_tags = nearest_tags[:10]

        # Calculate scene-tag-specific coverage (uses same threshold as above)
        scene_tag_coverage = self._calculate_scene_tag_coverage(scene_id, scene_tags, threshold)

        # Build suggested_tags: Library tags that could improve scene coverage
        # These are tags covering frames that scene's tags don't cover
        suggested_tags: list[dict[str, Any]] = []
        if scene_tag_coverage["coverage_ratio"] < coverage_ratio:
            # There's a gap - find tags that could fill it
            suggested_agg: dict[str, dict[str, Any]] = {}

            # Get scene tag embeddings for comparison
            scene_tag_embeddings: list[tuple[str, NDArray[np.float32]]] = []
            for tag in scene_tags:
                emb = self.storage.get_tag_embedding(tag, self.model_key)
                if emb is not None:
                    scene_tag_embeddings.append((tag, np.array(emb, dtype=np.float32)))

            # Get frame embeddings
            frame_embeddings = self.storage.get_scene_frame_embeddings(scene_id)

            if scene_tag_embeddings and frame_embeddings.shape[0] > 0:
                # Build scene tag matrix
                scene_tag_matrix = np.stack([e for _, e in scene_tag_embeddings])
                norms = np.linalg.norm(scene_tag_matrix, axis=1, keepdims=True)
                norms[norms < 1e-8] = 1.0
                scene_tag_matrix = scene_tag_matrix / norms

                # Compute frame-to-scene-tag similarities
                scene_sims = np.dot(frame_embeddings, scene_tag_matrix.T)
                scene_best_sims = np.max(scene_sims, axis=1)

                # For each frame covered by library but not by scene tags
                for i, fr in enumerate(frame_rows):
                    lib_sim = fr["best_similarity"]
                    scene_sim = scene_best_sims[i] if i < len(scene_best_sims) else 0.0

                    # Frame is covered by library but NOT by scene tags
                    if lib_sim >= threshold and scene_sim < threshold:
                        best_tag = fr["best_tag"]
                        # Skip if already on scene or not a real stash tag
                        if best_tag.lower() in scene_tags:
                            continue
                        if best_tag.lower() not in stash_tag_names:
                            continue

                        if best_tag not in suggested_agg:
                            suggested_agg[best_tag] = {
                                "tag": best_tag,
                                "total_similarity": 0.0,
                                "frame_count": 0,
                            }
                        suggested_agg[best_tag]["frame_count"] += 1
                        suggested_agg[best_tag]["total_similarity"] += lib_sim

            # Build sorted list
            for info in suggested_agg.values():
                count = info["frame_count"]
                suggested_tags.append(
                    {
                        "tag": info["tag"],
                        "avg_similarity": round(info["total_similarity"] / count, 4),
                        "frame_count": count,
                    }
                )
            suggested_tags.sort(key=lambda x: x["frame_count"], reverse=True)
            suggested_tags = suggested_tags[:10]

        return {
            "has_data": True,
            "total_frames": total_frames,
            "covered_frames": covered_frames,
            "uncovered_frames": uncovered_frames,
            "coverage_ratio": coverage_ratio,
            "scene_tag_coverage": scene_tag_coverage,
            "threshold": round(threshold, 4),
            "uncovered_frame_list": uncovered_frame_list,
            "nearest_tags": nearest_tags,
            "suggested_tags": suggested_tags,
        }

    def preview_tag_impact(self, scene_id: int, tag_name: str) -> dict[str, Any]:
        """Preview the coverage impact of adding a hypothetical tag.

        Generates a text embedding for the tag name and computes how many
        currently-uncovered frames would become covered if this tag existed.

        Uses cached embeddings when available to avoid slow model loading.

        Args:
            scene_id: The scene to analyze.
            tag_name: The potential tag name to test.

        Returns:
            Dict with coverage preview data:
            - tag_name: The tested tag
            - current_coverage: Current coverage ratio
            - new_coverage: Projected coverage with this tag
            - frames_covered: Number of uncovered frames this tag would cover
            - avg_similarity: Average similarity of uncovered frames to this tag
            - max_similarity: Highest similarity frame
        """
        # Get current coverage data
        frame_rows: list[FrameTagCoverageRecord] = self.storage.get_scene_tag_coverage(scene_id)
        if not frame_rows:
            return {"error": "No coverage data for this scene"}

        total_frames = len(frame_rows)
        current_covered = sum(1 for fr in frame_rows if fr["is_covered"])
        current_ratio = current_covered / total_frames if total_frames > 0 else 0.0

        # Get uncovered frame embeddings
        uncovered_embeddings = self.storage.get_uncovered_frame_embeddings(scene_id)
        if uncovered_embeddings.ndim < 2 or uncovered_embeddings.shape[0] == 0:
            return {
                "tag_name": tag_name,
                "current_coverage": round(current_ratio, 4),
                "new_coverage": round(current_ratio, 4),
                "frames_covered": 0,
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "message": "No uncovered frames to analyze",
            }

        # Try to get cached embedding first (fast path)
        tag_lower = tag_name.lower()
        cached_embedding = self.storage.get_tag_embedding(tag_lower, self.model_key)

        if cached_embedding is not None:
            tag_embedding = np.array(cached_embedding, dtype=np.float32)
        else:
            # Generate text embedding for the tag (slow - requires model load)
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.providers.openclip import OpenCLIPEmbeddingProvider

            model_parts = self.model_key.split(":")
            if len(model_parts) == 2 and model_parts[0] == "openclip":
                model_name = model_parts[1]
            else:
                model_name = "ViT-H-14"

            config = EmbeddingConfig(provider="openclip", model=model_name)
            provider = OpenCLIPEmbeddingProvider(config)

            try:
                result = provider.embed_text(tag_lower)
                tag_embedding = np.array(result["embedding"], dtype=np.float32)
                # Cache for future use
                self.storage.save_tag_embedding(
                    tag_lower, self.model_key, tag_embedding.tolist(), "preview"
                )
            finally:
                provider.cleanup()

        # Normalize tag embedding
        tag_norm = float(np.linalg.norm(tag_embedding))
        if tag_norm < 1e-8:
            return {"error": "Failed to generate tag embedding"}
        tag_embedding = tag_embedding / tag_norm

        # Compute similarities to all uncovered frames
        # uncovered_embeddings shape: (N, 1024)
        similarities = np.dot(uncovered_embeddings, tag_embedding)

        # Get the current threshold
        threshold = self._derive_threshold_from_db()

        # Count how many frames would become covered
        frames_covered = int((similarities >= threshold).sum())
        avg_sim = float(np.mean(similarities))
        max_sim = float(np.max(similarities))

        # Calculate new coverage
        new_covered = current_covered + frames_covered
        new_ratio = new_covered / total_frames if total_frames > 0 else 0.0

        return {
            "tag_name": tag_name,
            "current_coverage": round(current_ratio, 4),
            "new_coverage": round(new_ratio, 4),
            "frames_covered": frames_covered,
            "total_uncovered": int(uncovered_embeddings.shape[0]),
            "avg_similarity": round(avg_sim, 4),
            "max_similarity": round(max_sim, 4),
            "threshold": round(threshold, 4),
        }
