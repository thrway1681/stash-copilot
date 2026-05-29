"""Dense frame extraction at 1fps with smart deduplication."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from ..embeddings.base import BaseImageEmbeddingProvider
    from ..embeddings.config import DenseFrameConfig


@dataclass
class DenseFrame:
    """Represents a single frame in dense extraction.

    Attributes:
        index: 0-based frame index (frame N at second N for 1fps)
        timestamp: Exact timestamp in seconds
        is_unique: Whether this frame is unique (not a duplicate)
        duplicate_of: If not unique, the frame index this is a duplicate of
    """

    index: int
    timestamp: float
    is_unique: bool
    duplicate_of: int | None = None


class DenseFrameExtractor:
    """Extract frames at 1fps with smart deduplication.

    This class handles:
    - Extracting frames from video at 1fps using FFmpeg
    - Generating embeddings for each frame
    - Deduplicating consecutive similar frames to reduce storage
    - Managing frame cache directory
    """

    def __init__(self, config: "DenseFrameConfig"):
        """Initialize the dense frame extractor.

        Args:
            config: Configuration for frame extraction and deduplication
        """
        self.config = config

    def extract_dense_frames(
        self,
        video_path: str,
        duration: float,
        embedder: "BaseImageEmbeddingProvider",
        scene_id: int | None = None,
    ) -> tuple[list[DenseFrame], list[list[float]]]:
        """
        Extract frames at 1fps, embedding and deduplicating on-the-fly.

        Processing flow:
        1. Extract frame at second N using FFmpeg
        2. Generate embedding for the frame
        3. Compare with previous frame's embedding
        4. If similarity < threshold, keep as unique
        5. Otherwise mark as duplicate and skip storage

        Args:
            video_path: Path to video file
            duration: Video duration in seconds
            embedder: Image embedding provider
            scene_id: Optional scene ID for cache directory naming

        Returns:
            Tuple of (frames_list, embeddings_list)
            - frames_list: All DenseFrame objects (unique + duplicates)
            - embeddings_list: Only embeddings for unique frames
        """
        frames: list[DenseFrame] = []
        embeddings: list[list[float]] = []
        prev_embedding: list[float] | None = None

        # Calculate max frames to process
        max_frames = min(
            int(duration * self.config.sampling_rate),
            self.config.max_frames_per_scene,
        )

        # Create cache directory for this scene
        if scene_id is not None:
            cache_dir = Path(self.config.cache_dir) / f"scene_{scene_id}"
        else:
            cache_dir = Path(self.config.cache_dir) / "temp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for frame_index in range(max_frames):
            timestamp = float(frame_index) / self.config.sampling_rate

            # Extract single frame at timestamp
            frame_path = self._extract_single_frame(video_path, timestamp, cache_dir, frame_index)

            # Generate embedding
            emb_result = embedder.embed_image(frame_path)
            emb = emb_result["embedding"]

            # Deduplication check
            is_unique = True
            duplicate_of = None

            if prev_embedding and self.config.use_deduplication:
                similarity = self._cosine_similarity(emb, prev_embedding)
                is_unique = similarity < self.config.deduplication_threshold

            # Always keep first N frames regardless of similarity
            if frame_index < self.config.min_unique_frames:
                is_unique = True

            if is_unique:
                frames.append(DenseFrame(index=frame_index, timestamp=timestamp, is_unique=True))
                embeddings.append(emb)
                prev_embedding = emb
            else:
                # Find the most recent unique frame
                for i in range(len(frames) - 1, -1, -1):
                    if frames[i].is_unique:
                        duplicate_of = frames[i].index
                        break

                frames.append(
                    DenseFrame(
                        index=frame_index,
                        timestamp=timestamp,
                        is_unique=False,
                        duplicate_of=duplicate_of,
                    )
                )

        return frames, embeddings

    def _extract_single_frame(
        self, video_path: str, timestamp: float, cache_dir: Path, frame_index: int
    ) -> str:
        """
        Extract a single frame at the specified timestamp using FFmpeg.

        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            cache_dir: Directory to save frame
            frame_index: Frame index for filename

        Returns:
            Path to extracted frame image
        """
        output_path = cache_dir / f"frame_{frame_index:06d}.jpg"

        # Skip if frame already exists
        if output_path.exists():
            return str(output_path)

        # FFmpeg command to extract single frame at timestamp
        # -ss: Seek to timestamp (fast seek before input)
        # -i: Input file
        # -vf: Video filter (scale width to config.frame_width, maintain aspect ratio)
        # -frames:v 1: Extract only 1 frame
        # -q:v 2: JPEG quality (2 is high quality)
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vf",
            f"scale={self.config.frame_width}:-1",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",  # Overwrite output file
            str(output_path),
        ]

        try:
            # Run FFmpeg, suppress output
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed to extract frame at {timestamp}s: {e}") from e

        return str(output_path)

    @staticmethod
    def _cosine_similarity(emb1: list[float], emb2: list[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Assumes embeddings are already normalized (unit vectors).

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity (0-1, where 1 = identical)
        """
        arr1 = np.array(emb1, dtype=np.float32)
        arr2 = np.array(emb2, dtype=np.float32)

        # Dot product of unit vectors = cosine similarity
        return float(np.dot(arr1, arr2))

    def extract_frames_batched(
        self,
        video_path: str,
        duration: float,
        embedder: "BaseImageEmbeddingProvider",
        scene_id: int | None = None,
        batch_size: int = 32,
    ) -> tuple[list[DenseFrame], list[list[float]]]:
        """
        Extract and embed frames in batches for better GPU utilization.

        This method extracts multiple frames before embedding them in batches,
        which is more efficient than one-by-one processing.

        Args:
            video_path: Path to video file
            duration: Video duration in seconds
            embedder: Image embedding provider
            scene_id: Optional scene ID for cache directory
            batch_size: Number of frames to process at once (default: 32)

        Returns:
            Tuple of (frames_list, embeddings_list)
        """
        frames: list[DenseFrame] = []
        embeddings: list[list[float]] = []
        prev_embedding: list[float] | None = None

        # Calculate max frames
        max_frames = min(
            int(duration * self.config.sampling_rate),
            self.config.max_frames_per_scene,
        )

        # Create cache directory
        if scene_id is not None:
            cache_dir = Path(self.config.cache_dir) / f"scene_{scene_id}"
        else:
            cache_dir = Path(self.config.cache_dir) / "temp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Process in batches
        for batch_start in range(0, max_frames, batch_size):
            batch_end = min(batch_start + batch_size, max_frames)
            batch_indices = list(range(batch_start, batch_end))

            # Extract all frames in batch
            batch_paths = []
            for frame_index in batch_indices:
                timestamp = float(frame_index) / self.config.sampling_rate
                frame_path = self._extract_single_frame(
                    video_path, timestamp, cache_dir, frame_index
                )
                batch_paths.append(frame_path)

            # Embed all frames in batch (GPU-optimal)
            # Cast to satisfy mypy - list[str] is a valid list[ImageInput]
            batch_results = embedder.embed_images(cast("list[Any]", batch_paths))
            batch_embeddings = [r["embedding"] for r in batch_results]

            # Process each frame for deduplication
            for frame_index, emb in zip(batch_indices, batch_embeddings):
                timestamp = float(frame_index) / self.config.sampling_rate

                # Deduplication check
                is_unique = True
                duplicate_of = None

                if prev_embedding and self.config.use_deduplication:
                    similarity = self._cosine_similarity(emb, prev_embedding)
                    is_unique = similarity < self.config.deduplication_threshold

                # Always keep first N frames
                if frame_index < self.config.min_unique_frames:
                    is_unique = True

                if is_unique:
                    frames.append(
                        DenseFrame(index=frame_index, timestamp=timestamp, is_unique=True)
                    )
                    embeddings.append(emb)
                    prev_embedding = emb
                else:
                    # Find most recent unique frame
                    for i in range(len(frames) - 1, -1, -1):
                        if frames[i].is_unique:
                            duplicate_of = frames[i].index
                            break

                    frames.append(
                        DenseFrame(
                            index=frame_index,
                            timestamp=timestamp,
                            is_unique=False,
                            duplicate_of=duplicate_of,
                        )
                    )

        return frames, embeddings

    def cleanup_frame_cache(self, scene_id: int) -> None:
        """
        Clean up extracted frames for a scene.

        Args:
            scene_id: Scene ID to clean up cache for
        """
        cache_dir = Path(self.config.cache_dir) / f"scene_{scene_id}"
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
