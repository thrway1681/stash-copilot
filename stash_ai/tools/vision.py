"""Vision-specific tools for accurate timestamp lookup during scene analysis."""

from typing import TYPE_CHECKING, Any

from .base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from ..embeddings.storage import EmbeddingStorage
    from ..stash_client import StashClient


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


class GetFrameTimestampTool(BaseTool):
    """
    Tool to convert displayed frame indices to exact timestamps.

    During vision analysis, frames are displayed numbered 1, 2, 3, etc.
    This tool lets the VLM look up the exact timestamp for any frame.
    """

    def __init__(
        self,
        stash: "StashClient",
        frame_timestamps: list[float],
    ):
        """
        Initialize with the frame timestamps from the current analysis.

        Args:
            stash: StashClient (required by base class but not used here)
            frame_timestamps: List of timestamps in seconds for each displayed frame
        """
        super().__init__(stash)
        self._frame_timestamps = frame_timestamps

    @property
    def name(self) -> str:
        return "get_frame_timestamp"

    @property
    def description(self) -> str:
        return (
            "Get the exact video timestamp for a displayed frame. "
            "Use this to verify timestamps before mentioning them in descriptions or tag suggestions. "
            f"Valid frame indices: 1 to {len(self._frame_timestamps)}."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "frame_index",
                "type": "integer",
                "description": "The frame number as displayed (1-indexed). Frame 1 is the first frame shown.",
                "required": True,
                "enum": None,
            }
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Get the timestamp for a specific frame index.

        Args:
            **kwargs: Must contain 'frame_index' (1-indexed frame number)

        Returns:
            ToolResult with timestamp data or error
        """
        frame_index = kwargs.get("frame_index")
        # Validate frame index
        if not isinstance(frame_index, int):
            try:
                frame_index = int(frame_index)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "data": None,
                    "error": f"frame_index must be an integer, got {type(frame_index).__name__ if frame_index else 'None'}",
                }

        if frame_index < 1 or frame_index > len(self._frame_timestamps):
            return {
                "success": False,
                "data": None,
                "error": f"frame_index {frame_index} out of range. Valid range: 1 to {len(self._frame_timestamps)}",
            }

        # Convert 1-indexed to 0-indexed
        timestamp_seconds = self._frame_timestamps[frame_index - 1]

        return {
            "success": True,
            "data": {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "timestamp_formatted": format_timestamp(timestamp_seconds),
            },
            "error": None,
        }


class FindSimilarFramesTool(BaseTool):
    """
    Tool to find all timestamps where similar visual content appears.

    Uses pre-computed 1fps frame embeddings to find visually similar moments
    throughout the video. Useful for finding when specific actions repeat
    or tracking visual themes across a scene.
    """

    def __init__(
        self,
        stash: "StashClient",
        scene_id: int,
        frame_timestamps: list[float],
        storage: "EmbeddingStorage",
    ):
        """
        Initialize with scene context and embedding storage.

        Args:
            stash: StashClient (required by base class)
            scene_id: Current scene being analyzed
            frame_timestamps: List of timestamps for displayed frames
            storage: EmbeddingStorage instance for querying frame embeddings
        """
        super().__init__(stash)
        self._scene_id = scene_id
        self._frame_timestamps = frame_timestamps
        self._storage = storage

    @property
    def name(self) -> str:
        return "find_similar_frames"

    @property
    def description(self) -> str:
        return (
            "Find other timestamps in the video where visually similar content appears. "
            "Use this to identify when specific actions, positions, or visual elements "
            "repeat throughout the scene. Returns timestamps sorted by visual similarity."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            {
                "name": "reference_frame",
                "type": "integer",
                "description": f"The frame number to use as reference (1 to {len(self._frame_timestamps)})",
                "required": True,
                "enum": None,
            },
            {
                "name": "min_similarity",
                "type": "number",
                "description": "Minimum similarity threshold (0.0 to 1.0). Default: 0.8. Higher = more similar.",
                "required": False,
                "enum": None,
            },
            {
                "name": "max_results",
                "type": "integer",
                "description": "Maximum number of similar frames to return. Default: 5.",
                "required": False,
                "enum": None,
            },
            {
                "name": "exclude_nearby_seconds",
                "type": "integer",
                "description": "Exclude frames within this many seconds of the reference. Default: 10.",
                "required": False,
                "enum": None,
            },
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Find timestamps with similar visual content to the reference frame.

        Args:
            **kwargs: Must contain 'reference_frame', optional: min_similarity,
                      max_results, exclude_nearby_seconds

        Returns:
            ToolResult with list of similar timestamps or error
        """
        reference_frame = kwargs.get("reference_frame")
        min_similarity = kwargs.get("min_similarity", 0.8)
        max_results = kwargs.get("max_results", 5)
        exclude_nearby_seconds = kwargs.get("exclude_nearby_seconds", 10)

        # Validate reference_frame
        if not isinstance(reference_frame, int):
            try:
                reference_frame = int(reference_frame)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "data": None,
                    "error": f"reference_frame must be an integer, got {type(reference_frame).__name__ if reference_frame else 'None'}",
                }

        if reference_frame < 1 or reference_frame > len(self._frame_timestamps):
            return {
                "success": False,
                "data": None,
                "error": f"reference_frame {reference_frame} out of range. Valid: 1 to {len(self._frame_timestamps)}",
            }

        # Validate other parameters
        min_similarity = max(0.0, min(1.0, float(min_similarity)))
        max_results = max(1, min(20, int(max_results)))
        exclude_nearby_seconds = max(0, int(exclude_nearby_seconds))

        # Get reference timestamp (convert 1-indexed to 0-indexed)
        reference_timestamp = self._frame_timestamps[reference_frame - 1]

        try:
            # Load reference frame embedding from storage
            # Frame embeddings are stored at 1fps with timestamp = frame_index (0-indexed)
            # We need to find the embedding closest to our reference timestamp
            reference_embedding = self._get_frame_embedding_at_timestamp(reference_timestamp)

            if reference_embedding is None:
                return {
                    "success": False,
                    "data": None,
                    "error": f"No frame embedding found near timestamp {reference_timestamp:.1f}s. "
                    "Frame embeddings may not be generated for this scene.",
                }

            # Search all frame embeddings for similar frames
            similar_frames = self._storage.find_frames_by_embedding(
                scene_id=self._scene_id,
                query_embedding=reference_embedding,
                min_similarity=min_similarity,
                max_results=max_results + 10,  # Get extra to account for filtering
            )

            # Filter out frames near the reference
            filtered_results = []
            for frame in similar_frames:
                frame_ts = frame["timestamp"]

                # Skip frames too close to reference
                if abs(frame_ts - reference_timestamp) < exclude_nearby_seconds:
                    continue

                filtered_results.append(
                    {
                        "timestamp_seconds": frame_ts,
                        "timestamp_formatted": format_timestamp(frame_ts),
                        "similarity": round(frame["similarity"], 3),
                    }
                )

                if len(filtered_results) >= max_results:
                    break

            return {
                "success": True,
                "data": {
                    "reference_frame": reference_frame,
                    "reference_timestamp": format_timestamp(reference_timestamp),
                    "similar_frames": filtered_results,
                    "total_found": len(filtered_results),
                },
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Error searching for similar frames: {e!s}",
            }

    def _get_frame_embedding_at_timestamp(
        self,
        target_timestamp: float,
    ) -> list[float] | None:
        """
        Get the frame embedding closest to the target timestamp.

        Since frame embeddings are at 1fps, we find the nearest second.

        Args:
            target_timestamp: Target timestamp in seconds

        Returns:
            Embedding vector or None if not found
        """
        # Round to nearest second for 1fps embeddings
        nearest_second = round(target_timestamp)

        # Load all frames for scene and find the closest one
        frames = self._storage._load_all_frames_for_scene(self._scene_id)

        if not frames:
            return None

        # Find frame with closest timestamp
        closest_frame = min(frames, key=lambda f: abs(f["timestamp"] - nearest_second))

        # Only return if reasonably close (within 2 seconds)
        if abs(closest_frame["timestamp"] - target_timestamp) <= 2.0:
            embedding: list[float] = closest_frame["embedding"]
            return embedding

        return None
