"""Tag suggestions task - embedding-based tag recommendations with evidence frames.

Uses frame-centric voting: each frame votes for matching tags based on
cosine similarity between frame embeddings and tag embeddings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..stash_client import StashClient

from stash_ai.embeddings.storage import EmbeddingStorage


@dataclass
class EvidenceFrame:
    """A frame that matched a tag, used as visual evidence."""

    frame_index: int
    similarity: float
    timestamp: str  # "MM:SS" format
    thumbnail_path: str  # Relative path to frame image


@dataclass
class TagSuggestion:
    """A tag suggestion with supporting evidence."""

    tag_id: int
    tag_name: str
    max_similarity: float  # Highest single-frame match
    mean_similarity: float  # Average across matching frames
    frame_count: int  # Frames with similarity >= threshold
    evidence_frames: list[EvidenceFrame] = field(default_factory=list)


class TagSuggestionsResult(TypedDict):
    """Result from tag suggestion computation."""

    status: str  # "complete", "error", "no_embeddings"
    scene_id: int
    suggestions: list[dict[str, Any]]  # Serialized TagSuggestion objects
    error: str | None


class TagSuggestionsTask:
    """Compute embedding-based tag suggestions for a scene.

    Uses frame-centric voting: each frame votes for tags based on
    cosine similarity. Tags with many votes rank higher.
    """

    # Minimum similarity for a frame to vote for a tag
    SIMILARITY_THRESHOLD = 0.30

    # Maximum suggestions to return
    MAX_SUGGESTIONS = 20

    # Number of evidence frames per suggestion
    EVIDENCE_FRAME_COUNT = 5

    def __init__(
        self,
        stash: StashClient,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        model_key: str = "siglip",
    ) -> None:
        """Initialize the tag suggestions task.

        Args:
            stash: StashClient instance for API calls
            storage: EmbeddingStorage instance for embeddings access
            log_callback: Optional callback for logging (message, level)
            model_key: Embedding model key (e.g., "siglip")
        """
        self.stash = stash
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.model_key = model_key

    def _compute_similarities(
        self,
        frame_embeddings: NDArray[np.float32],
        tag_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between all frames and all tags.

        Args:
            frame_embeddings: (N, D) array of frame embeddings
            tag_embeddings: (T, D) array of tag embeddings

        Returns:
            (N, T) similarity matrix
        """
        # Normalize embeddings for cosine similarity
        frame_norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
        tag_norms = np.linalg.norm(tag_embeddings, axis=1, keepdims=True)

        frame_normalized = frame_embeddings / (frame_norms + 1e-8)
        tag_normalized = tag_embeddings / (tag_norms + 1e-8)

        # Compute dot product (cosine similarity for normalized vectors)
        return cast("NDArray[np.float32]", np.dot(frame_normalized, tag_normalized.T))

    def _aggregate_votes(
        self,
        similarities: NDArray[np.float32],
        tag_info: list[dict[str, Any]],
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate frame votes into tag suggestions.

        Args:
            similarities: (N, T) similarity matrix
            tag_info: List of {"id": int, "name": str} for each tag
            threshold: Minimum similarity for a vote (defaults to SIMILARITY_THRESHOLD)

        Returns:
            List of vote aggregations per tag
        """
        if threshold is None:
            threshold = self.SIMILARITY_THRESHOLD

        results = []

        for tag_idx, tag in enumerate(tag_info):
            tag_similarities = similarities[:, tag_idx]

            # Find frames that vote for this tag
            voting_mask = tag_similarities >= threshold
            voting_frames = np.where(voting_mask)[0]

            if len(voting_frames) == 0:
                continue

            voting_sims = tag_similarities[voting_mask]

            results.append(
                {
                    "tag_id": tag["id"],
                    "tag_name": tag["name"],
                    "frame_count": len(voting_frames),
                    "max_similarity": float(np.max(voting_sims)),
                    "mean_similarity": float(np.mean(voting_sims)),
                    "voting_frames": voting_frames.tolist(),
                    "voting_similarities": voting_sims.tolist(),
                }
            )

        return results

    def run(self, scene_id: int) -> TagSuggestionsResult:
        """Compute tag suggestions for a scene.

        Args:
            scene_id: The scene to analyze

        Returns:
            TagSuggestionsResult with suggestions or error
        """
        try:
            # 1. Load frame embeddings
            frame_data = self.storage._load_all_frames_for_scene(scene_id)
            if not frame_data:
                return TagSuggestionsResult(
                    status="no_embeddings",
                    scene_id=scene_id,
                    suggestions=[],
                    error="No frame embeddings found for this scene",
                )

            frame_embeddings = np.array([f["embedding"] for f in frame_data], dtype=np.float32)
            frame_timestamps = [f["timestamp"] for f in frame_data]

            # 2. Load tag embeddings
            tag_embeddings_data = self.storage.get_all_tag_embeddings(self.model_key)
            if not tag_embeddings_data:
                return TagSuggestionsResult(
                    status="no_embeddings",
                    scene_id=scene_id,
                    suggestions=[],
                    error="No tag embeddings found. Run tag embedding first.",
                )

            # 3. Get tag info from Stash (for IDs and names)
            tag_info = self._get_stash_tags()
            if not tag_info:
                return TagSuggestionsResult(
                    status="error",
                    scene_id=scene_id,
                    suggestions=[],
                    error="Failed to load tags from Stash",
                )

            # 4. Build tag embedding matrix (only for tags that exist in Stash)
            tag_name_to_embedding: dict[str, list[float]] = {
                t["text"].lower(): t["embedding"] for t in tag_embeddings_data
            }
            valid_tags: list[dict[str, Any]] = []
            tag_embeddings_list: list[list[float]] = []
            for tag in tag_info:
                name_lower = tag["name"].lower()
                if name_lower in tag_name_to_embedding:
                    valid_tags.append(tag)
                    tag_embeddings_list.append(tag_name_to_embedding[name_lower])

            if not valid_tags:
                return TagSuggestionsResult(
                    status="error",
                    scene_id=scene_id,
                    suggestions=[],
                    error="No tags have embeddings",
                )

            tag_embeddings = np.array(tag_embeddings_list, dtype=np.float32)

            # 5. Compute similarities
            similarities = self._compute_similarities(frame_embeddings, tag_embeddings)

            # 6. Aggregate votes
            votes = self._aggregate_votes(similarities, valid_tags)

            # 7. Get exclusions
            existing_tag_ids = self._get_scene_tag_ids(scene_id)
            dismissed_tag_ids = self.storage.get_dismissed_tags(scene_id)
            excluded_ids = existing_tag_ids | dismissed_tag_ids

            # 8. Filter and rank
            filtered_votes = [v for v in votes if v["tag_id"] not in excluded_ids]
            filtered_votes.sort(
                key=lambda v: (v["frame_count"], v["max_similarity"]),
                reverse=True,
            )

            # 9. Build suggestions with evidence
            suggestions: list[dict[str, Any]] = []
            for vote in filtered_votes[: self.MAX_SUGGESTIONS]:
                evidence = self._build_evidence_frames(
                    scene_id,
                    vote["voting_frames"],
                    vote["voting_similarities"],
                    frame_timestamps,
                )
                suggestions.append(
                    {
                        "tag_id": vote["tag_id"],
                        "tag_name": vote["tag_name"],
                        "max_similarity": vote["max_similarity"],
                        "mean_similarity": vote["mean_similarity"],
                        "frame_count": vote["frame_count"],
                        "evidence_frames": evidence,
                    }
                )

            return TagSuggestionsResult(
                status="complete",
                scene_id=scene_id,
                suggestions=suggestions,
                error=None,
            )

        except Exception as e:
            self.log(f"Tag suggestion error: {e}", "error")
            return TagSuggestionsResult(
                status="error",
                scene_id=scene_id,
                suggestions=[],
                error=str(e),
            )

    def _get_stash_tags(self) -> list[dict[str, Any]]:
        """Get all tags from Stash, excluding bracketed system tags."""
        try:
            result = self.stash.call_GQL(
                """
                query FindTags {
                    findTags(filter: { per_page: -1 }) {
                        tags { id name }
                    }
                }
                """
            )
            if not result or "findTags" not in result:
                return []

            tags: list[dict[str, Any]] = []
            for t in result["findTags"]["tags"]:
                name = t["name"]
                if not any(c in name for c in "[](){}<>"):
                    tags.append({"id": int(t["id"]), "name": name})
            return tags
        except Exception as e:
            self.log(f"Failed to get tags: {e}", "warning")
            return []

    def _get_scene_tag_ids(self, scene_id: int) -> set[int]:
        """Get tag IDs already on the scene."""
        try:
            result = self.stash.call_GQL(
                """
                query FindScene($id: ID!) {
                    findScene(id: $id) { tags { id } }
                }
                """,
                {"id": str(scene_id)},
            )
            if not result or "findScene" not in result:
                return set()
            return {int(t["id"]) for t in result["findScene"]["tags"]}
        except Exception as e:
            self.log(f"Failed to get scene tags: {e}", "warning")
            return set()

    def _build_evidence_frames(
        self,
        scene_id: int,
        frame_indices: list[int],
        similarities: list[float],
        timestamps: list[float],
    ) -> list[dict[str, Any]]:
        """Build evidence frame list for a suggestion."""
        paired = list(zip(frame_indices, similarities))
        paired.sort(key=lambda x: x[1], reverse=True)

        evidence: list[dict[str, Any]] = []
        for frame_idx, sim in paired[: self.EVIDENCE_FRAME_COUNT]:
            ts = timestamps[frame_idx] if frame_idx < len(timestamps) else 0
            minutes = int(ts // 60)
            seconds = int(ts % 60)
            ts_str = f"{minutes}:{seconds:02d}"

            frame_path = f"assets/embedded_frames/scene_{scene_id}/frame_{frame_idx:04d}.jpg"

            evidence.append(
                {
                    "frame_index": frame_idx,
                    "similarity": round(sim, 3),
                    "timestamp": ts_str,
                    "thumbnail_path": frame_path,
                }
            )

        return evidence
