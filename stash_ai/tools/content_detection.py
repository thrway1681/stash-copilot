"""Content detection tool using text-to-image similarity on frame embeddings."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from .base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from ..embeddings.base import BaseImageEmbeddingProvider
    from ..embeddings.storage import EmbeddingStorage


class GatingConditions(TypedDict, total=False):
    """Optional conditions from classification that enable this detector.

    These conditions are matched against the scene classification results
    to determine if a detector should run. All specified conditions must
    match for the detector to be enabled.
    """

    scene_type: list[str]  # e.g., ["couple", "threesome"]
    genders_present: list[str]  # e.g., ["mixed"]
    activities: list[str]  # e.g., ["vaginal"] - boosts relevance


@dataclass
class ContentDetector:
    """Definition for a content type detector.

    Each detector defines:
    - Search phrases for text-to-image similarity matching
    - Threshold for triggering detection
    - Tag to suggest when content is detected
    - Optional gating conditions to filter when detector runs
    - Clustering parameters for grouping nearby detections
    """

    name: str  # Internal identifier
    phrases: list[str]  # 2-3 visually descriptive phrases
    threshold: float  # Min similarity (e.g., 0.30)
    suggested_tag: str  # Tag to suggest if detected
    requires: GatingConditions | None = None  # Optional gating conditions
    cluster_window: float = 30.0  # Seconds to group nearby frames


# Content detector definitions
CONTENT_DETECTORS: dict[str, ContentDetector] = {
    "creampie": ContentDetector(
        name="creampie",
        phrases=["creampie dripping out", "cum leaking from vagina"],
        threshold=0.30,
        suggested_tag="creampie",
        requires={
            "scene_type": ["couple", "threesome", "group"],
            "genders_present": ["mixed"],
            "activities": ["vaginal"],
        },
    ),
}


def get_available_detectors(classification: dict[str, Any] | None) -> list[str]:
    """
    Filter content detectors based on classification results.

    Args:
        classification: Classification dict from stage 1, or None

    Returns:
        List of detector names that pass gating conditions
    """
    if not classification:
        # Return detectors with no gating requirements
        return [name for name, detector in CONTENT_DETECTORS.items() if detector.requires is None]

    available: list[str] = []

    for name, detector in CONTENT_DETECTORS.items():
        if detector.requires is None:
            # No gating = always available
            available.append(name)
            continue

        reqs = detector.requires
        passes_gating = True

        # Check scene_type requirement
        if "scene_type" in reqs:
            if classification.get("scene_type") not in reqs["scene_type"]:
                passes_gating = False

        # Check genders_present requirement
        if "genders_present" in reqs:
            if classification.get("genders_present") not in reqs["genders_present"]:
                passes_gating = False

        if passes_gating:
            available.append(name)

    return available


def format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    if seconds < 0:
        return "0:00"

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class ContentEvent(TypedDict):
    """A detected content event (cluster of frames)."""

    start_timestamp: float
    end_timestamp: float
    start_formatted: str
    end_formatted: str
    peak_timestamp: float
    peak_formatted: str
    peak_similarity: float
    matched_phrase: str
    frame_count: int


def cluster_events(
    candidates: list[dict[str, Any]],
    window: float,
) -> list[ContentEvent]:
    """
    Group nearby candidate frames into events.

    Args:
        candidates: Frames above threshold with timestamp/similarity/matched_phrase
        window: Seconds to group (e.g., 30.0)

    Returns:
        List of content events
    """
    if not candidates:
        return []

    # Sort by timestamp
    sorted_candidates = sorted(candidates, key=lambda c: c["timestamp"])

    events: list[ContentEvent] = []
    current_cluster: list[dict[str, Any]] = [sorted_candidates[0]]

    for candidate in sorted_candidates[1:]:
        # If within window of last frame in current cluster, add to it
        if candidate["timestamp"] - current_cluster[-1]["timestamp"] <= window:
            current_cluster.append(candidate)
        else:
            # Finalize current cluster, start new one
            events.append(_finalize_event(current_cluster))
            current_cluster = [candidate]

    # Don't forget last cluster
    events.append(_finalize_event(current_cluster))

    return events


def _finalize_event(frames: list[dict[str, Any]]) -> ContentEvent:
    """Summarize a cluster of frames into an event."""
    peak_frame = max(frames, key=lambda f: f["similarity"])

    return {
        "start_timestamp": frames[0]["timestamp"],
        "end_timestamp": frames[-1]["timestamp"],
        "start_formatted": format_timestamp(frames[0]["timestamp"]),
        "end_formatted": format_timestamp(frames[-1]["timestamp"]),
        "peak_timestamp": peak_frame["timestamp"],
        "peak_formatted": format_timestamp(peak_frame["timestamp"]),
        "peak_similarity": round(peak_frame["similarity"], 3),
        "matched_phrase": peak_frame["matched_phrase"],
        "frame_count": len(frames),
    }


class FindContentTool(BaseTool):
    """
    Search scene frame embeddings to detect specific content types.

    Uses text-to-image similarity with CLIP-style embeddings.
    Returns candidate timestamps with similarity scores for LLM verification.
    """

    def __init__(
        self,
        stash: Any,  # StashClient, but we don't use it
        scene_id: int,
        storage: "EmbeddingStorage",
        image_embedder: "BaseImageEmbeddingProvider",
        available_detectors: list[str],
    ):
        """
        Initialize the content detection tool.

        Args:
            stash: StashClient (required by base class)
            scene_id: Scene being analyzed
            storage: EmbeddingStorage for loading frame embeddings
            image_embedder: Image embedding provider (must support text embedding)
            available_detectors: List of detector names available for this scene
        """
        super().__init__(stash)
        self._scene_id = scene_id
        self._storage = storage
        self._image_embedder = image_embedder
        self._available_detectors = available_detectors

    @property
    def name(self) -> str:
        return "find_content"

    @property
    def description(self) -> str:
        detector_list = ", ".join(self._available_detectors)
        return (
            f"Search for specific content types ({detector_list}) in the scene using "
            "frame embeddings. Returns candidate timestamps with similarity scores. "
            "You should verify candidates against the frames you can see and report "
            "both the embedding results AND your visual verification."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "content_type",
                "type": "string",
                "description": "Type of content to search for",
                "required": True,
                "enum": self._available_detectors,
            },
            {
                "name": "custom_threshold",
                "type": "number",
                "description": "Override default similarity threshold (0.0-1.0). Optional.",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Search for content type in scene frame embeddings.

        Args:
            **kwargs: Must contain 'content_type', optionally 'custom_threshold'

        Returns:
            ToolResult with detected events or error
        """
        content_type: str = kwargs.get("content_type", "")
        custom_threshold: float | None = kwargs.get("custom_threshold")

        # Validate content_type
        if content_type not in self._available_detectors:
            return {
                "success": False,
                "data": None,
                "error": f"Unknown content type: {content_type}. Available: {self._available_detectors}",
            }

        detector = CONTENT_DETECTORS[content_type]
        threshold = custom_threshold if custom_threshold is not None else detector.threshold

        try:
            # 1. Embed text phrases
            phrase_embeddings: list[np.ndarray] = []
            for phrase in detector.phrases:
                result = self._image_embedder.embed_text(phrase)
                phrase_embeddings.append(np.array(result["embedding"], dtype=np.float32))

            # 2. Load all frame embeddings for scene
            frames = self._storage._load_all_frames_for_scene(self._scene_id)

            if not frames:
                return {
                    "success": False,
                    "data": None,
                    "error": f"No frame embeddings found for scene {self._scene_id}",
                }

            # 3. Compute max similarity per frame across all phrases
            frame_scores: list[dict[str, Any]] = []
            for frame in frames:
                frame_emb = np.array(frame["embedding"], dtype=np.float32)

                # Normalize for cosine similarity (if not already normalized)
                frame_norm = frame_emb / (np.linalg.norm(frame_emb) + 1e-8)

                similarities = []
                for phrase_emb in phrase_embeddings:
                    phrase_norm = phrase_emb / (np.linalg.norm(phrase_emb) + 1e-8)
                    similarity = float(np.dot(frame_norm, phrase_norm))
                    similarities.append(similarity)

                best_idx = int(np.argmax(similarities))
                frame_scores.append(
                    {
                        "timestamp": frame["timestamp"],
                        "frame_index": frame["frame_index"],
                        "similarity": similarities[best_idx],
                        "matched_phrase": detector.phrases[best_idx],
                    }
                )

            # 4. Filter by threshold
            candidates = [f for f in frame_scores if f["similarity"] >= threshold]

            # 5. Cluster into events
            events = cluster_events(candidates, detector.cluster_window)

            # 6. Return top 5 events sorted by peak similarity
            events = sorted(events, key=lambda e: e["peak_similarity"], reverse=True)[:5]

            return {
                "success": True,
                "data": {
                    "detected": len(events) > 0,
                    "content_type": content_type,
                    "threshold_used": threshold,
                    "events": events,
                    "suggested_tag": detector.suggested_tag if events else None,
                    "note": "Please verify these candidates against the frames you can see.",
                },
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Content detection failed: {e!s}",
            }
