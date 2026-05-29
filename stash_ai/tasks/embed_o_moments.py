"""Task for generating O-moment embeddings from O markers."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

from ..embeddings.config import EmbeddingConfig
from ..embeddings.provider import get_embedding_provider
from ..embeddings.storage import EmbeddingStorage
from ..recommendations.types import OMomentData, OMomentExtractionConfig
from .frame_extractor import FrameExtractionConfig, FrameExtractor
from .o_moment_extractor import OMomentExtractor

if TYPE_CHECKING:
    from ..stash_client import StashClient


@dataclass
class EmbedOMomentsConfig:
    """Configuration for O-moment embedding generation."""

    # O-moment extraction settings
    window_seconds: float = 120.0  # Total window (+/- 60s)
    frames_per_window: int = 12  # Frames to extract per window
    o_tag_name: str = "O"  # Tag name for O markers

    # Frame extraction settings
    frame_width: int = 640  # Frame resolution


class EmbedOMomentsTask:
    """
    Task for generating embeddings from O-moment frames.

    Workflow:
    1. Find all O markers in the library
    2. For each marker, extract frames in window around position
    3. Generate image embeddings for frames
    4. Average embeddings and store
    """

    def __init__(
        self,
        stash: "StashClient",
        embedding_config: EmbeddingConfig,
        embed_config: EmbedOMomentsConfig | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Initialize the O-moment embedding task.

        Args:
            stash: StashClient instance
            embedding_config: Config for image embedding model
            embed_config: Optional O-moment-specific configuration
            log_callback: Optional logging callback
            progress_callback: Optional progress callback
        """
        self.stash = stash
        self.embedding_config = embedding_config
        self.config = embed_config or EmbedOMomentsConfig()
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

        # Initialize providers lazily
        self._embedder: Any | None = None

        # Lock for GPU operations
        self._gpu_lock = (
            RLock()
        )  # Reentrant lock - needed because embedder property also acquires this lock

        # Initialize storage with model_key
        model_key = embedding_config.model_key
        self.storage = EmbeddingStorage(model_key=model_key)
        self.log(f"Using embedding model key: {model_key}", "debug")

        # Setup frame extractor
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cache_dir = os.path.join(plugin_dir, "assets", "o_moment_cache")
        self.frame_extractor = FrameExtractor(
            config=FrameExtractionConfig(
                frame_width=self.config.frame_width,
            ),
            cache_dir=cache_dir,
            log_callback=self.log,
        )

        # Setup O-moment extractor
        self.o_moment_extractor = OMomentExtractor(
            frame_extractor=self.frame_extractor,
            config=OMomentExtractionConfig(
                window_seconds=self.config.window_seconds,
                frames_per_window=self.config.frames_per_window,
                o_tag_name=self.config.o_tag_name,
            ),
            log_callback=self.log,
            progress_callback=self.progress,
        )

    @property
    def embedder(self) -> Any:
        """Lazy-load image embedding provider with thread-safe initialization."""
        if self._embedder is None:
            with self._gpu_lock:
                # Double-check after acquiring lock to prevent race condition
                if self._embedder is None:
                    self._embedder = get_embedding_provider(self.embedding_config)
        return self._embedder

    def embed_scene_o_moments(
        self,
        scene_id: int,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Generate and store O-moment embeddings for a single scene.

        Args:
            scene_id: Stash scene ID
            force: If True, regenerate even if embeddings exist

        Returns:
            Dict with success status and embedding info
        """
        # Get O markers for this scene
        markers = self.o_moment_extractor.get_o_markers(scene_id)
        if not markers:
            return {
                "success": True,
                "scene_id": scene_id,
                "embedded": 0,
                "skipped": 0,
                "message": "No O markers found",
            }

        # Get scene info
        scene_info = self.o_moment_extractor.get_scene_info(scene_id)
        if not scene_info:
            return {"success": False, "error": f"Scene {scene_id} not found"}

        video_path = scene_info["file_path"]
        duration = scene_info["duration"]

        if not os.path.exists(video_path):
            return {"success": False, "error": f"Video file not found: {video_path}"}

        embedded = 0
        skipped = 0

        for moment_data in markers:
            marker = moment_data["marker"]
            o_event_index = moment_data["o_event_index"]

            # Check if already embedded
            if not force and self.storage.has_o_moment_embedding(scene_id, o_event_index):
                self.log(
                    f"O-moment {o_event_index} for scene {scene_id} already embedded",
                    "debug",
                )
                skipped += 1
                continue

            # Extract frames
            center_position = marker["seconds"]
            frames_base64, window_start, window_end = (
                self.o_moment_extractor.extract_o_moment_frames(
                    scene_id,
                    video_path,
                    center_position,
                    duration,
                )
            )

            if not frames_base64:
                self.log(
                    f"No frames extracted for O-moment {o_event_index} at {center_position:.1f}s",
                    "warning",
                )
                continue

            # Create embedding (with GPU lock)
            self.log(f"Creating embedding from {len(frames_base64)} frames...", "debug")
            with self._gpu_lock:
                embedding = self._create_embedding(frames_base64)
            self.log(f"Embedding created: {embedding is not None}", "debug")

            if not embedding:
                self.log(
                    f"Failed to create embedding for O-moment {o_event_index}",
                    "warning",
                )
                continue

            # Store embedding
            actual_window = window_end - window_start
            self.storage.store_o_moment_embedding(
                scene_id=scene_id,
                o_event_index=o_event_index,
                marker_id=marker["marker_id"],
                center_timestamp=center_position,
                window_seconds=actual_window,
                embedding=embedding,
                frame_count=len(frames_base64),
            )

            embedded += 1
            self.log(
                f"Embedded O-moment {o_event_index} for scene {scene_id} "
                f"({len(frames_base64)} frames at {center_position:.1f}s)",
                "info",
            )

        return {
            "success": True,
            "scene_id": scene_id,
            "embedded": embedded,
            "skipped": skipped,
            "total_markers": len(markers),
        }

    def embed_all_o_moments(
        self,
        force: bool = False,
        scene_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Generate O-moment embeddings for all scenes with O markers.

        Args:
            force: If True, regenerate all embeddings
            scene_ids: Optional list of specific scene IDs to process

        Returns:
            Summary of embedding generation
        """
        # Get all O markers
        all_markers = self.o_moment_extractor.get_o_markers()

        if not all_markers:
            self.log("No O markers found in library", "warning")
            return {
                "total_scenes": 0,
                "total_markers": 0,
                "embedded": 0,
                "skipped": 0,
                "errors": 0,
            }

        # Group by scene
        scenes: dict[int, list[OMomentData]] = {}
        for marker in all_markers:
            sid = marker["scene_id"]
            if scene_ids and sid not in scene_ids:
                continue
            if sid not in scenes:
                scenes[sid] = []
            scenes[sid].append(marker)

        total_scenes = len(scenes)
        total_markers = sum(len(m) for m in scenes.values())

        self.log(
            f"Processing {total_markers} O-moments across {total_scenes} scenes",
            "info",
        )

        embedded = 0
        skipped = 0
        errors = 0
        error_details: list[str] = []

        for i, scene_id in enumerate(scenes.keys()):
            self.progress(i, total_scenes)

            try:
                result = self.embed_scene_o_moments(scene_id, force=force)
                if result.get("success"):
                    embedded += result.get("embedded", 0)
                    skipped += result.get("skipped", 0)
                else:
                    errors += 1
                    if len(error_details) < 10:
                        error_details.append(f"Scene {scene_id}: {result.get('error')}")
            except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
                # Handle expected errors - file issues, value errors, runtime issues
                # Note: IOError removed as it's an alias for OSError in Python 3
                errors += 1
                if len(error_details) < 10:
                    error_details.append(f"Scene {scene_id}: {type(e).__name__}: {e}")
                self.log(f"Error processing scene {scene_id}: {e}", "error")
            except Exception as e:
                # Catch remaining unexpected errors - log and continue to avoid blocking batch
                # Note: KeyboardInterrupt/SystemExit are BaseException, not Exception, so they propagate
                errors += 1
                self.log(f"Unexpected error for scene {scene_id}: {type(e).__name__}: {e}", "error")
                if len(error_details) < 10:
                    error_details.append(f"Scene {scene_id}: {type(e).__name__}: {e}")

        self.progress(total_scenes, total_scenes)

        # Get stats
        stats = self.storage.get_o_moment_stats()

        return {
            "total_scenes": total_scenes,
            "total_markers": total_markers,
            "embedded": embedded,
            "skipped": skipped,
            "errors": errors,
            "error_details": error_details,
            "storage_stats": stats,
        }

    def _create_embedding(self, frames_base64: list[str]) -> list[float] | None:
        """Create averaged embedding from frames."""
        import base64

        import numpy as np

        if not frames_base64:
            return None

        try:
            # Decode base64 strings to bytes (embedder expects bytes, not base64 strings)
            frames_bytes = [base64.b64decode(b64) for b64 in frames_base64]

            # Get embeddings for each frame
            self.log(f"Calling embedder.embed_images with {len(frames_bytes)} images...", "trace")
            results = self.embedder.embed_images(frames_bytes)
            self.log(f"Embedder returned {len(results) if results else 0} results", "trace")

            if not results:
                return None

            embeddings = [r["embedding"] for r in results]

            if not embeddings:
                return None

            # Validate dimensions are consistent
            if len(embeddings) > 1:
                first_dim = len(embeddings[0])
                if not all(len(e) == first_dim for e in embeddings):
                    self.log("Inconsistent embedding dimensions across frames", "error")
                    return None

            # Average the embeddings
            arr = np.array(embeddings, dtype=np.float32)
            avg_embedding = np.mean(arr, axis=0)

            # Normalize to unit vector
            norm = float(np.linalg.norm(avg_embedding))
            if norm > 0:
                avg_embedding = avg_embedding / norm

            return cast("list[float]", avg_embedding.tolist())

        except Exception as e:
            self.log(f"Error creating embedding: {e}", "error")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get O-moment embedding statistics."""
        return self.storage.get_o_moment_stats()

    def clear_all(self) -> int:
        """Clear all O-moment embeddings."""
        deleted = self.storage.clear_all_o_moments()
        self.log(f"Cleared {deleted} O-moment embeddings", "info")
        return deleted
