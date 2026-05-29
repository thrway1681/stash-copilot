"""Task for embedding cached frames that don't have frame embeddings yet.

This task backfills frame embeddings for scenes that already have frames
in the embedded_frames directory but were embedded before individual frame
storage was implemented.
"""

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

from ..embeddings.base import BaseEmbeddingProvider, BaseImageEmbeddingProvider
from ..embeddings.config import EmbeddingConfig
from ..embeddings.provider import get_embedding_provider
from ..embeddings.storage import EmbeddingStorage

if TYPE_CHECKING:
    from ..stash_client import StashClient


class EmbedCachedFramesTask:
    """
    Task for embedding cached frames that don't have frame-level embeddings.

    This is a backfill task for scenes that:
    1. Have frames extracted in assets/embedded_frames/scene_*
    2. Don't have frame embeddings in the database yet

    The task computes embeddings using the configured image embedding model
    and stores them for use with smart frame selection.
    """

    def __init__(
        self,
        stash: "StashClient",
        embedding_config: EmbeddingConfig,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        num_workers: int = 2,
    ) -> None:
        """
        Initialize the cached frames embedding task.

        Args:
            stash: StashClient instance (for scene info lookups)
            embedding_config: Config for image embedding model
            log_callback: Optional logging callback
            progress_callback: Optional progress callback
            num_workers: Number of parallel workers (default 2)
        """
        self.stash = stash
        self.embedding_config = embedding_config
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)
        self.num_workers = num_workers

        # Initialize storage with model key
        self.storage = EmbeddingStorage(model_key=embedding_config.model_key)

        # Lazy-load embedder
        self._embedder: BaseEmbeddingProvider | None = None

        # Lock for GPU operations to prevent resource contention
        self._gpu_lock = Lock()

        # Cache directory
        plugin_dir = Path(__file__).parent.parent.parent
        self.cache_dir = plugin_dir / "assets" / "embedded_frames"

    @property
    def embedder(self) -> BaseEmbeddingProvider:
        """Lazy-load image embedding provider."""
        if self._embedder is None:
            self._embedder = get_embedding_provider(self.embedding_config)
        return self._embedder

    def cleanup(self) -> None:
        """Release GPU resources."""
        if self._embedder is not None:
            if hasattr(self._embedder, "cleanup"):
                try:
                    self._embedder.cleanup()
                except Exception:
                    pass
            self._embedder = None

    def __del__(self) -> None:
        """Destructor - ensure resources are freed."""
        self.cleanup()

    def run(
        self,
        force: bool = False,
        scene_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Embed cached frames for scenes missing frame embeddings.

        Args:
            force: If True, re-embed even if frame embeddings exist
            scene_id: If provided, only process this scene

        Returns:
            Summary with counts of processed, skipped, and errors
        """
        try:
            return self._run_impl(force=force, scene_id=scene_id)
        finally:
            self.cleanup()

    def _run_impl(
        self,
        force: bool,
        scene_id: int | None,
    ) -> dict[str, Any]:
        """Internal implementation of run()."""
        if not self.cache_dir.exists():
            self.log(f"Cache directory not found: {self.cache_dir}", "warning")
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}

        # Find scenes with cached frames
        if scene_id is not None:
            # Single scene mode
            scene_dirs = [self.cache_dir / f"scene_{scene_id}"]
            if not scene_dirs[0].exists():
                self.log(f"No cached frames for scene {scene_id}", "warning")
                return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
        else:
            # All scenes mode
            scene_dirs = sorted(self.cache_dir.glob("scene_*"))

        if not scene_dirs:
            self.log("No cached frame directories found", "info")
            return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}

        total = len(scene_dirs)

        self.log(
            f"Found {total} scene(s) with cached frames, embedding with "
            f"{self.embedding_config.model_key} using {self.num_workers} workers",
            "info",
        )

        # Thread-safe counters
        counter_lock = Lock()
        counters = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "completed": 0,
        }
        error_details: list[str] = []

        def process_scene(scene_dir: Path) -> None:
            """Process a single scene (thread-safe)."""
            nonlocal counters, error_details

            # Extract scene ID from directory name
            try:
                scene_id_str = scene_dir.name.split("_")[1]
                current_scene_id = int(scene_id_str)
            except (IndexError, ValueError):
                self.log(f"Invalid scene directory: {scene_dir.name}", "warning")
                with counter_lock:
                    counters["errors"] += 1
                    counters["completed"] += 1
                return

            # Check if already has frame embeddings (outside GPU lock - read-only)
            if not force and self.storage.has_frame_embeddings(current_scene_id):
                self.log(
                    f"Scene {current_scene_id}: Already has frame embeddings, skipping",
                    "debug",
                )
                with counter_lock:
                    counters["skipped"] += 1
                    counters["completed"] += 1
                return

            # Load frame paths (I/O bound - outside GPU lock)
            frame_data = self._load_frame_paths(scene_dir)
            if not frame_data:
                self.log(f"Scene {current_scene_id}: No frames found", "warning")
                with counter_lock:
                    counters["errors"] += 1
                    counters["completed"] += 1
                return

            # Embed frames (GPU bound - inside lock)
            try:
                with self._gpu_lock:
                    result = self._embed_scene_frames(current_scene_id, frame_data)

                with counter_lock:
                    counters["completed"] += 1
                    if result["success"]:
                        counters["processed"] += 1
                        self.log(
                            f"Scene {current_scene_id}: Stored {result['frame_count']} "
                            f"frame embeddings",
                            "info",
                        )
                    else:
                        counters["errors"] += 1
                        if len(error_details) < 10:
                            error_details.append(f"Scene {current_scene_id}: {result.get('error')}")

                    # Update progress periodically
                    if counters["completed"] % 5 == 0:
                        self.progress(counters["completed"], total)

            except Exception as e:
                self.log(f"Scene {current_scene_id}: Error - {e}", "error")
                with counter_lock:
                    counters["errors"] += 1
                    counters["completed"] += 1
                    if len(error_details) < 10:
                        error_details.append(f"Scene {current_scene_id}: {e!s}")

        # Process scenes in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_scene, scene_dir): scene_dir for scene_dir in scene_dirs
            }

            # Wait for completion (as_completed allows for better progress tracking)
            for _future in as_completed(futures):
                # Exceptions are handled inside process_scene
                pass

        self.progress(total, total)

        summary = {
            "total": total,
            "processed": counters["processed"],
            "skipped": counters["skipped"],
            "errors": counters["errors"],
            "error_details": error_details,
            "model_key": self.embedding_config.model_key,
        }

        self.log(
            f"Embed Cached Frames complete: {counters['processed']} processed, "
            f"{counters['skipped']} skipped, {counters['errors']} errors",
            "info",
        )

        return summary

    def _load_frame_paths(
        self,
        scene_dir: Path,
    ) -> list[tuple[str, int, float]]:
        """
        Load frame paths from a scene's cache directory.

        Returns:
            List of (path, frame_number, timestamp) tuples sorted by frame number
        """
        frame_pattern = re.compile(r"frame_(\d+)\.(?:jpg|png)$", re.IGNORECASE)
        frames: list[tuple[str, int, float]] = []

        for f in scene_dir.glob("frame_*.jpg"):
            match = frame_pattern.search(f.name)
            if not match:
                continue

            frame_num = int(match.group(1))
            # 1fps extraction: frame_0001 = timestamp 0
            timestamp = float(frame_num - 1)
            frames.append((str(f), frame_num, timestamp))

        # Also check for PNG frames
        for f in scene_dir.glob("frame_*.png"):
            match = frame_pattern.search(f.name)
            if not match:
                continue

            frame_num = int(match.group(1))
            timestamp = float(frame_num - 1)
            frames.append((str(f), frame_num, timestamp))

        # Sort by frame number
        frames.sort(key=lambda x: x[1])
        return frames

    def _embed_scene_frames(
        self,
        scene_id: int,
        frame_data: list[tuple[str, int, float]],
    ) -> dict[str, Any]:
        """
        Embed all frames for a scene and store them.

        Args:
            scene_id: Stash scene ID
            frame_data: List of (path, frame_number, timestamp) tuples

        Returns:
            Result dict with success status and frame count
        """
        if not frame_data:
            return {"success": False, "error": "No frames to embed"}

        # Check if embedder supports images
        if not self.embedder.supports_images:
            return {
                "success": False,
                "error": f"Model {self.embedding_config.model_key} does not support images",
            }

        # Extract just the paths for embedding
        frame_paths = [f[0] for f in frame_data]

        # Embed all frames
        try:
            img_embedder = cast("BaseImageEmbeddingProvider", self.embedder)
            # Cast to satisfy mypy - list[str] is a valid list[ImageInput]
            results = img_embedder.embed_images(cast("list[Any]", frame_paths))
            if not results:
                return {"success": False, "error": "No embeddings returned"}
        except Exception as e:
            return {"success": False, "error": f"Embedding failed: {e}"}

        # Store each frame embedding
        embeddings = [r["embedding"] for r in results]
        first_timestamp: float | None = None
        last_timestamp: float | None = None

        for i, ((_path, _frame_num, timestamp), embedding) in enumerate(
            zip(frame_data, embeddings)
        ):
            self.storage.store_frame_embedding(
                scene_id=scene_id,
                frame_index=i,
                timestamp=timestamp,
                embedding=embedding,
            )

            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

        # Compute and store metadata
        avg_embedding = self._average_embeddings(embeddings)
        duration = last_timestamp + 1.0 if last_timestamp is not None else 0.0

        self.storage.store_frame_metadata(
            scene_id=scene_id,
            frame_count=len(embeddings),
            total_frames_extracted=len(frame_data),
            duration=duration,
            sampling_rate=1.0,
            composite_embedding=avg_embedding,
            dedup_ratio=0.0,
            first_frame_timestamp=first_timestamp,
            last_frame_timestamp=last_timestamp,
        )

        return {
            "success": True,
            "frame_count": len(embeddings),
        }

    def _average_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> list[float]:
        """Average multiple embeddings into a single normalized embedding."""
        import numpy as np
        from numpy.typing import NDArray

        if not embeddings:
            raise ValueError("Cannot average empty list of embeddings")

        if len(embeddings) == 1:
            return embeddings[0]

        arr: NDArray[np.float32] = np.array(embeddings, dtype=np.float32)
        averaged = np.mean(arr, axis=0)

        # Normalize
        norm: float = float(np.linalg.norm(averaged))
        if norm > 0:
            averaged = averaged / norm

        return cast("list[float]", averaged.tolist())
