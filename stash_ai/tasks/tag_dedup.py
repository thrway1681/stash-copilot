"""Tag deduplication task - find and merge duplicate tags via embedding similarity."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..stash_client import StashClient

from stash_ai.embeddings.storage import EmbeddingStorage


class TagInfo(TypedDict):
    """Minimal tag information for dedup candidates."""

    id: int
    name: str
    scene_count: int


class TagDedupCandidate(TypedDict):
    """A pair of tags that may be duplicates."""

    tag_a: TagInfo
    tag_b: TagInfo
    similarity: float
    suggested_keep: str  # "a" or "b"


class FindDuplicateTagsResult(TypedDict):
    """Result from duplicate tag detection."""

    status: str  # "complete", "error", "no_embeddings"
    candidates: list[TagDedupCandidate]
    error: str | None


class MergeTagsResult(TypedDict):
    """Result from merging two tags."""

    status: str  # "complete", "error"
    scenes_updated: int
    error: str | None


class FindDuplicateTagsTask:
    """Find duplicate tags using embedding cosine similarity.

    Computes all-pairs similarity on cached tag embeddings,
    filters above threshold, and returns candidates sorted
    by descending similarity with auto-suggested keep targets.
    """

    SIMILARITY_THRESHOLD = 0.75

    def __init__(
        self,
        stash: StashClient,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        model_key: str = "openclip:ViT-H-14",
    ) -> None:
        self.stash = stash
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.model_key = model_key

    def run(self) -> FindDuplicateTagsResult:
        """Find duplicate tag candidates."""
        try:
            # 1. Load tag embeddings (only stash_tag source)
            tag_embeddings_data = self.storage.get_all_tag_embeddings(self.model_key)
            stash_tags_only = [
                t for t in tag_embeddings_data if t["source"] == "stash_tag"
            ]
            if not stash_tags_only:
                return FindDuplicateTagsResult(
                    status="no_embeddings",
                    candidates=[],
                    error="No tag embeddings found. Run 'Build Tag Vocabulary' first.",
                )

            self.log(f"Loaded {len(stash_tags_only)} stash tag embeddings", "info")

            # 2. Build embedding matrix
            tag_names = [t["text"] for t in stash_tags_only]
            embeddings = np.array(
                [t["embedding"] for t in stash_tags_only], dtype=np.float32
            )

            # 3. Compute all-pairs cosine similarity
            similarities = self._compute_all_pairs(embeddings)

            # 4. Extract pairs above threshold
            pairs = self._extract_candidate_pairs(similarities, tag_names)
            self.log(f"Found {len(pairs)} pairs above {self.SIMILARITY_THRESHOLD} threshold", "info")

            if not pairs:
                return FindDuplicateTagsResult(
                    status="complete",
                    candidates=[],
                    error=None,
                )

            # 5. Filter out previously dismissed pairs
            dismissed = self.storage.get_dismissed_tag_merges()
            pairs = [
                (a, b, sim) for a, b, sim in pairs
                if (min(a.lower(), b.lower()), max(a.lower(), b.lower())) not in dismissed
            ]
            self.log(f"{len(pairs)} pairs after excluding dismissed", "info")

            # 6. Get tag IDs and scene counts from Stash
            tag_info_map = self._get_tag_info_with_scene_counts()
            if not tag_info_map:
                return FindDuplicateTagsResult(
                    status="error",
                    candidates=[],
                    error="Failed to load tag info from Stash",
                )

            # 7. Build candidate list with scene counts
            candidates: list[TagDedupCandidate] = []
            for name_a, name_b, sim in pairs:
                info_a = tag_info_map.get(name_a.lower())
                info_b = tag_info_map.get(name_b.lower())
                if not info_a or not info_b:
                    continue

                suggested_keep = "a" if info_a["scene_count"] >= info_b["scene_count"] else "b"
                candidates.append(TagDedupCandidate(
                    tag_a=info_a,
                    tag_b=info_b,
                    similarity=round(sim, 4),
                    suggested_keep=suggested_keep,
                ))

            # Sort by descending similarity
            candidates.sort(key=lambda c: c["similarity"], reverse=True)

            self.log(f"Returning {len(candidates)} dedup candidates", "info")
            return FindDuplicateTagsResult(
                status="complete",
                candidates=candidates,
                error=None,
            )

        except Exception as e:
            self.log(f"Tag dedup error: {e}", "error")
            return FindDuplicateTagsResult(
                status="error",
                candidates=[],
                error=str(e),
            )

    def _compute_all_pairs(
        self, embeddings: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute all-pairs cosine similarity matrix.

        Args:
            embeddings: (N, D) array of tag embeddings

        Returns:
            (N, N) similarity matrix
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        return np.dot(normalized, normalized.T)

    def _extract_candidate_pairs(
        self,
        similarities: NDArray[np.float32],
        tag_names: list[str],
    ) -> list[tuple[str, str, float]]:
        """Extract tag pairs above similarity threshold.

        Only checks upper triangle to avoid duplicates.
        Uses vectorized NumPy ops instead of Python loops for O(N^2) scalability.

        Returns:
            List of (tag_a_name, tag_b_name, similarity) sorted descending.
        """
        n = len(tag_names)
        rows, cols = np.triu_indices(n, k=1)
        sims = similarities[rows, cols]
        mask = sims >= self.SIMILARITY_THRESHOLD
        ri, ci, si = rows[mask], cols[mask], sims[mask]
        # Sort descending by similarity
        order = np.argsort(si)[::-1]
        return [
            (tag_names[int(ri[k])], tag_names[int(ci[k])], float(si[k]))
            for k in order
        ]

    def _get_tag_info_with_scene_counts(self) -> dict[str, TagInfo]:
        """Get tag IDs and scene counts from Stash.

        Returns:
            Dict mapping lowercase tag name to TagInfo.
        """
        try:
            result = self.stash.call_GQL(
                """
                query FindTags {
                    findTags(filter: { per_page: -1 }) {
                        tags {
                            id
                            name
                            scene_count
                        }
                    }
                }
                """
            )
            if not result or "findTags" not in result:
                return {}

            info_map: dict[str, TagInfo] = {}
            for t in result["findTags"]["tags"]:
                name = t["name"]
                if not any(c in name for c in "[](){}<>"):
                    info_map[name.lower()] = TagInfo(
                        id=int(t["id"]),
                        name=name,
                        scene_count=t.get("scene_count", 0),
                    )
            return info_map
        except Exception as e:
            self.log(f"Failed to get tags with scene counts: {e}", "warning")
            return {}


class MergeTagsTask:
    """Merge one tag into another: reassign scenes, then delete the source tag.

    Moves all scene associations from remove_tag to keep_tag,
    then deletes remove_tag from Stash and cleans up its embedding.
    """

    def __init__(
        self,
        stash: StashClient,
        storage: EmbeddingStorage,
        log_callback: Callable[[str, str], None] | None = None,
        model_key: str = "openclip:ViT-H-14",
    ) -> None:
        self.stash = stash
        self.storage = storage
        self.log = log_callback or (lambda msg, level: None)
        self.model_key = model_key

    def run(self, keep_tag_id: int, remove_tag_id: int) -> MergeTagsResult:
        """Merge remove_tag into keep_tag.

        Args:
            keep_tag_id: ID of the tag to keep
            remove_tag_id: ID of the tag to delete after reassignment

        Returns:
            MergeTagsResult with status and scene count
        """
        try:
            # 1. Get the tag name BEFORE we delete it (needed for embedding cleanup)
            remove_tag_name = self._get_tag_name(remove_tag_id)
            if remove_tag_name is None:
                self.log(
                    f"Could not fetch name for tag {remove_tag_id}; "
                    "embedding will not be cleaned up",
                    "warning",
                )

            # 2. Find all scenes with the remove_tag
            scenes = self._find_scenes_with_tag(remove_tag_id)
            self.log(f"Found {len(scenes)} scenes with tag {remove_tag_id}", "info")

            scenes_updated = 0

            # 3. For each scene: add keep_tag, remove remove_tag
            for scene in scenes:
                scene_id = int(scene["id"])
                existing_tag_ids = {int(t["id"]) for t in scene["tags"]}

                new_tag_ids = existing_tag_ids.copy()
                new_tag_ids.add(keep_tag_id)
                new_tag_ids.discard(remove_tag_id)

                # Only update if tags actually changed
                if new_tag_ids != existing_tag_ids:
                    success = self._update_scene_tags(scene_id, list(new_tag_ids))
                    if not success:
                        return MergeTagsResult(
                            status="error",
                            scenes_updated=scenes_updated,
                            error=f"Failed to update scene {scene_id}. Stopping to prevent partial merge.",
                        )
                    scenes_updated += 1

            # 4. Delete the now-empty tag
            self._destroy_tag(remove_tag_id)
            self.log(f"Deleted tag {remove_tag_id}", "info")

            # 5. Clean up embedding from storage
            if remove_tag_name:
                self.storage.delete_tag_embedding(remove_tag_name, self.model_key)

            return MergeTagsResult(
                status="complete",
                scenes_updated=scenes_updated,
                error=None,
            )

        except Exception as e:
            self.log(f"Tag merge error: {e}", "error")
            return MergeTagsResult(
                status="error",
                scenes_updated=0,
                error=str(e),
            )

    def _find_scenes_with_tag(self, tag_id: int) -> list[dict[str, Any]]:
        """Find all scenes that have a given tag."""
        try:
            result = self.stash.call_GQL(
                """
                query FindScenes($tag_id: [ID!]) {
                    findScenes(
                        scene_filter: { tags: { value: $tag_id, modifier: INCLUDES } }
                        filter: { per_page: -1 }
                    ) {
                        scenes { id tags { id } }
                    }
                }
                """,
                {"tag_id": [str(tag_id)]},
            )
            if not result or "findScenes" not in result:
                return []
            return result["findScenes"]["scenes"]
        except Exception as e:
            self.log(f"Failed to find scenes with tag {tag_id}: {e}", "warning")
            return []

    def _update_scene_tags(self, scene_id: int, tag_ids: list[int]) -> bool:
        """Update a scene's tags to the given list of tag IDs."""
        try:
            result = self.stash.call_GQL(
                """
                mutation SceneUpdate($id: ID!, $tag_ids: [ID!]) {
                    sceneUpdate(input: { id: $id, tag_ids: $tag_ids }) { id }
                }
                """,
                {"id": str(scene_id), "tag_ids": [str(t) for t in tag_ids]},
            )
            return result is not None and "sceneUpdate" in result
        except Exception as e:
            self.log(f"Failed to update scene {scene_id}: {e}", "warning")
            return False

    def _destroy_tag(self, tag_id: int) -> bool:
        """Delete a tag from Stash."""
        try:
            result = self.stash.call_GQL(
                """
                mutation TagDestroy($id: ID!) {
                    tagDestroy(input: { id: $id })
                }
                """,
                {"id": str(tag_id)},
            )
            return result is not None
        except Exception as e:
            self.log(f"Failed to delete tag {tag_id}: {e}", "warning")
            return False

    def _get_tag_name(self, tag_id: int) -> str | None:
        """Get a tag's name by ID (for embedding cleanup)."""
        try:
            result = self.stash.call_GQL(
                """
                query FindTag($id: ID!) {
                    findTag(id: $id) { name }
                }
                """,
                {"id": str(tag_id)},
            )
            if result and "findTag" in result and result["findTag"]:
                return result["findTag"]["name"]
            return None
        except Exception:
            return None
