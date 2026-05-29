"""O-moment frame extraction and embedding for O markers.

This module extracts frames around O markers (O-event scene markers)
and creates embeddings for "Peak Moments" recommendations.

O markers provide EXACT playback positions via scene_markers.seconds,
so no estimation/correlation heuristics are needed.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..embeddings.storage import EmbeddingStorage
from ..recommendations.types import (
    OMomentData,
    OMomentExtractionConfig,
    OMomentMarker,
)
from ..tools.database import get_readonly_connection, get_stash_db_path
from .frame_extractor import FrameExtractor


@dataclass
class OMomentExtractionResult:
    """Result from extracting and embedding an O-moment."""

    scene_id: int
    marker_id: int
    o_event_index: int
    center_timestamp: float
    window_seconds: float
    frame_count: int
    embedding: list[float]


class OMomentExtractor:
    """
    Extract and embed frames around O markers (O-event positions).

    Uses O markers from Stash's scene_markers table for EXACT playback
    positions, then extracts frames in a window around each marker and
    creates embeddings from the averaged frame embeddings.
    """

    def __init__(
        self,
        frame_extractor: FrameExtractor,
        config: OMomentExtractionConfig | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        """
        Initialize O-moment extractor.

        Args:
            frame_extractor: FrameExtractor instance for extracting video frames
            config: O-moment extraction configuration
            log_callback: Optional callback for logging (message, level)
            progress_callback: Optional callback for progress (current, total)
        """
        self.frame_extractor = frame_extractor
        self.config = config or OMomentExtractionConfig()
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda current, total: None)

    def get_o_markers(self, scene_id: int | None = None) -> list[OMomentData]:
        """
        Get O markers from Stash database.

        Args:
            scene_id: If provided, only get markers for this scene.
                      Otherwise returns all O markers.

        Returns:
            List of OMomentData with marker information
        """
        db_path = get_stash_db_path()
        if not db_path.exists():
            self.log(f"Stash database not found at {db_path}", "error")
            return []

        conn = get_readonly_connection(db_path)
        try:
            cursor = conn.cursor()

            if scene_id is not None:
                cursor.execute(
                    """
                    SELECT sm.id, sm.scene_id, sm.seconds, sm.end_seconds, sm.created_at
                    FROM scene_markers sm
                    JOIN tags t ON sm.primary_tag_id = t.id
                    WHERE t.name = ? AND sm.scene_id = ?
                    ORDER BY sm.seconds
                """,
                    (self.config.o_tag_name, scene_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT sm.id, sm.scene_id, sm.seconds, sm.end_seconds, sm.created_at
                    FROM scene_markers sm
                    JOIN tags t ON sm.primary_tag_id = t.id
                    WHERE t.name = ?
                    ORDER BY sm.scene_id, sm.seconds
                """,
                    (self.config.o_tag_name,),
                )

            rows = cursor.fetchall()
        finally:
            conn.close()

        # Group markers by scene and assign o_event_index
        scene_markers: dict[int, list[Any]] = {}
        for row in rows:
            sid = row["scene_id"]
            if sid not in scene_markers:
                scene_markers[sid] = []
            scene_markers[sid].append(row)

        results: list[OMomentData] = []
        for sid, markers in scene_markers.items():
            for idx, row in enumerate(markers):
                marker: OMomentMarker = {
                    "marker_id": row["id"],
                    "scene_id": row["scene_id"],
                    "seconds": row["seconds"],
                    "end_seconds": row["end_seconds"],
                    "created_at": row["created_at"],
                }
                results.append(
                    {
                        "scene_id": sid,
                        "marker": marker,
                        "o_event_index": idx,
                    }
                )

        self.log(
            f"Found {len(results)} O markers across {len(scene_markers)} scenes",
            "info",
        )
        return results

    def get_scene_info(self, scene_id: int) -> dict[str, Any] | None:
        """
        Get scene information from Stash database.

        Args:
            scene_id: Scene ID

        Returns:
            Dict with scene info or None if not found:
            {
                "id": int,
                "duration": float,  # seconds
                "file_path": str,
            }
        """
        db_path = get_stash_db_path()
        if not db_path.exists():
            return None

        conn = get_readonly_connection(db_path)
        try:
            cursor = conn.cursor()

            # Get scene with file info
            # Note: files table has basename only, path is in folders table
            cursor.execute(
                """
                SELECT s.id, vf.duration, f.basename, fo.path as folder_path
                FROM scenes s
                JOIN scenes_files sf ON s.id = sf.scene_id
                JOIN video_files vf ON sf.file_id = vf.file_id
                JOIN files f ON vf.file_id = f.id
                JOIN folders fo ON f.parent_folder_id = fo.id
                WHERE s.id = ?
                LIMIT 1
            """,
                (scene_id,),
            )

            row = cursor.fetchone()
        finally:
            conn.close()

        if not row:
            return None

        # Construct full path from folder path + file basename
        full_path = os.path.join(row["folder_path"], row["basename"])
        return {
            "id": row["id"],
            "duration": row["duration"],
            "file_path": full_path,
        }

    def extract_o_moment_frames(
        self,
        scene_id: int,
        video_path: str,
        center_position: float,
        duration: float,
    ) -> tuple[list[str], float, float]:
        """
        Extract frames from window around O-moment position.

        Args:
            scene_id: Scene ID for cache key
            video_path: Path to video file
            center_position: Center of extraction window (seconds)
            duration: Total video duration (seconds)

        Returns:
            Tuple of (frames_base64, actual_start, actual_end)
        """
        half_window = self.config.window_seconds / 2

        # Clamp window to video duration bounds
        window_start = max(0, center_position - half_window)
        window_end = min(duration, center_position + half_window)
        actual_window = window_end - window_start

        # Calculate frame timestamps within window
        num_frames = self.config.frames_per_window
        if num_frames <= 0:
            self.log(f"Invalid frames_per_window: {num_frames}, using default 12", "warning")
            num_frames = 12

        if actual_window <= 0:
            return [], window_start, window_end

        # Use linspace for evenly distributed timestamps (excluding endpoint)
        timestamps = np.linspace(window_start, window_end, num_frames, endpoint=False).tolist()

        self.log(
            f"Extracting {num_frames} frames from {window_start:.1f}s to {window_end:.1f}s "
            f"(center: {center_position:.1f}s)",
            "debug",
        )

        # Extract frames at specific timestamps using FFmpeg directly
        frames_base64: list[str] = []
        for i, ts in enumerate(timestamps):
            self.log(f"Extracting frame {i + 1}/{num_frames} at {ts:.1f}s", "debug")
            frame_b64 = self._extract_single_frame(video_path, ts)
            if frame_b64:
                frames_base64.append(frame_b64)
                self.log(f"Frame {i + 1} extracted successfully", "debug")

        return frames_base64, window_start, window_end

    def _extract_single_frame(
        self,
        video_path: str,
        timestamp: float,
    ) -> str | None:
        """
        Extract a single frame at a specific timestamp.

        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds

        Returns:
            Base64-encoded frame image, or None if failed
        """
        import base64
        import subprocess
        import tempfile

        # Use ffmpeg to extract frame to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                self.frame_extractor.config.ffmpeg_path,
                "-ss",
                str(timestamp),
                "-i",
                video_path,
                "-vframes",
                "1",
                "-vf",
                f"scale={self.frame_extractor.config.frame_width}:-1",
                "-q:v",
                "2",
                "-y",
                tmp_path,
            ]

            self.log(f"Running ffmpeg for timestamp {timestamp}s", "trace")
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=30,
            )
            self.log(f"FFmpeg completed for timestamp {timestamp}s", "trace")

            # Read and encode frame
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        except subprocess.CalledProcessError as e:
            stderr_msg = (
                e.stderr.decode("utf-8", errors="replace")[:100] if e.stderr else "Unknown error"
            )
            self.log(f"FFmpeg error at {timestamp}s: {stderr_msg}", "warning")
            return None
        except subprocess.TimeoutExpired:
            self.log(f"FFmpeg timeout at {timestamp}s", "warning")
            return None
        except Exception as e:
            self.log(f"Error extracting frame at {timestamp}s: {e}", "warning")
            return None
        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except OSError as e:
                self.log(f"Failed to cleanup temp file {tmp_path}: {e}", "debug")

    def create_o_moment_embedding(
        self,
        frames_base64: list[str],
        embedder: Any,
    ) -> list[float] | None:
        """
        Create embedding from O-moment frames.

        Averages the embeddings of all frames in the window.

        Args:
            frames_base64: List of base64-encoded frame images
            embedder: Embedding provider with embed_images method

        Returns:
            Averaged embedding vector, or None if failed
        """
        if not frames_base64:
            self.log("No frames to embed", "warning")
            return None

        try:
            # Get embeddings for each frame
            results = embedder.embed_images(frames_base64)

            if not results:
                self.log("No embeddings returned", "warning")
                return None

            # Extract embeddings from response dicts
            embeddings = [r["embedding"] for r in results]

            if not embeddings:
                self.log("No embeddings extracted from results", "warning")
                return None

            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)

            # Normalize to unit vector
            norm = float(np.linalg.norm(avg_embedding))
            if norm > 0:
                avg_embedding = avg_embedding / norm

            return list(avg_embedding.tolist())

        except Exception as e:
            self.log(f"Error creating embedding: {e}", "error")
            return None

    def process_scene_o_moments(
        self,
        scene_id: int,
        embedder: Any,
        storage: EmbeddingStorage,
    ) -> list[OMomentExtractionResult]:
        """
        Process all O-moments for a single scene.

        Args:
            scene_id: Scene ID to process
            embedder: Embedding provider
            storage: EmbeddingStorage for storing results

        Returns:
            List of extraction results
        """
        # Get scene info
        scene_info = self.get_scene_info(scene_id)
        if not scene_info:
            self.log(f"Scene {scene_id} not found", "warning")
            return []

        video_path = scene_info["file_path"]
        duration = scene_info["duration"]

        if not video_path or not Path(video_path).exists():
            self.log(f"Video file not found: {video_path}", "warning")
            return []

        # Get O markers for this scene
        markers = self.get_o_markers(scene_id)
        if not markers:
            self.log(f"No O markers for scene {scene_id}", "debug")
            return []

        results: list[OMomentExtractionResult] = []

        for moment_data in markers:
            marker = moment_data["marker"]
            o_event_index = moment_data["o_event_index"]

            # Check if already embedded
            if storage.has_o_moment_embedding(scene_id, o_event_index):
                self.log(
                    f"O-moment {o_event_index} for scene {scene_id} already embedded, skipping",
                    "debug",
                )
                continue

            # Extract frames
            center_position = marker["seconds"]

            # Validate marker timestamp doesn't exceed video duration
            if center_position > duration:
                self.log(
                    f"Marker timestamp {center_position:.1f}s exceeds video duration "
                    f"{duration:.1f}s for scene {scene_id}, skipping",
                    "warning",
                )
                continue

            frames_base64, window_start, window_end = self.extract_o_moment_frames(
                scene_id,
                video_path,
                center_position,
                duration,
            )

            if not frames_base64:
                self.log(
                    f"No frames extracted for O-moment {o_event_index} at {center_position}s",
                    "warning",
                )
                continue

            # Create embedding
            embedding = self.create_o_moment_embedding(frames_base64, embedder)
            if not embedding:
                self.log(
                    f"Failed to create embedding for O-moment {o_event_index}",
                    "warning",
                )
                continue

            # Store embedding
            actual_window = window_end - window_start
            storage.store_o_moment_embedding(
                scene_id=scene_id,
                o_event_index=o_event_index,
                marker_id=marker["marker_id"],
                center_timestamp=center_position,
                window_seconds=actual_window,
                embedding=embedding,
                frame_count=len(frames_base64),
            )

            results.append(
                OMomentExtractionResult(
                    scene_id=scene_id,
                    marker_id=marker["marker_id"],
                    o_event_index=o_event_index,
                    center_timestamp=center_position,
                    window_seconds=actual_window,
                    frame_count=len(frames_base64),
                    embedding=embedding,
                )
            )

            self.log(
                f"Embedded O-moment {o_event_index} for scene {scene_id} "
                f"({len(frames_base64)} frames at {center_position:.1f}s)",
                "info",
            )

        return results

    def process_all_o_moments(
        self,
        embedder: Any,
        storage: EmbeddingStorage,
        scene_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Process O-moments for all scenes with O markers.

        Args:
            embedder: Embedding provider
            storage: EmbeddingStorage for storing results
            scene_ids: Optional list of specific scene IDs to process

        Returns:
            Summary dict with processing statistics
        """
        # Get all O markers
        all_markers = self.get_o_markers()

        # Get unique scene IDs
        if scene_ids:
            target_scenes = set(scene_ids)
            markers_to_process = [m for m in all_markers if m["scene_id"] in target_scenes]
        else:
            markers_to_process = all_markers

        # Group by scene
        scenes: dict[int, list[OMomentData]] = {}
        for marker in markers_to_process:
            sid = marker["scene_id"]
            if sid not in scenes:
                scenes[sid] = []
            scenes[sid].append(marker)

        total_scenes = len(scenes)
        total_markers = len(markers_to_process)

        self.log(
            f"Processing {total_markers} O-moments across {total_scenes} scenes",
            "info",
        )

        processed_scenes = 0
        processed_markers = 0
        skipped_markers = 0
        failed_markers = 0

        for i, (scene_id, scene_markers) in enumerate(scenes.items()):
            self.progress(i + 1, total_scenes)

            results = self.process_scene_o_moments(scene_id, embedder, storage)

            processed_scenes += 1
            processed_markers += len(results)
            skipped_markers += len(scene_markers) - len(results)

            if len(results) < len(scene_markers):
                # Some markers were skipped (already embedded) or failed
                existing = sum(
                    1
                    for m in scene_markers
                    if storage.has_o_moment_embedding(scene_id, m["o_event_index"])
                )
                failed_markers += len(scene_markers) - len(results) - existing

        return {
            "total_scenes": total_scenes,
            "total_markers": total_markers,
            "processed_scenes": processed_scenes,
            "processed_markers": processed_markers,
            "skipped_markers": skipped_markers,
            "failed_markers": failed_markers,
            "completed_at": datetime.now().isoformat(),
        }
