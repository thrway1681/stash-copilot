"""FFmpeg-based frame extraction with caching for scene embedding and vision analysis.

This module provides fps-based frame extraction using a single-pass FFmpeg filter,
optimized for dense frame extraction (e.g., 1fps) to capture time-series information.
"""

import base64
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

# Module-level defaults (optimized for dense extraction)
DEFAULT_FPS_RATE = 1.0  # 1 frame per second
DEFAULT_MIN_FRAMES = 0  # No minimum (allows very short scenes)
DEFAULT_MAX_FRAMES = 0  # 0 = no limit (fps-based)
DEFAULT_FRAME_WIDTH = 640  # Default resolution


@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction.

    Attributes:
        fps_rate: Frames per second to extract (default: 1.0 for 1fps)
        min_frames: Minimum frames to extract regardless of duration (0 = no minimum)
        max_frames: Maximum frames to extract (0 = no limit)
        frame_width: Width to scale frames to (height auto-calculated)
        ffmpeg_path: Path to ffmpeg binary
        ffmpeg_timeout: Timeout in seconds for FFmpeg extraction
    """

    fps_rate: float = DEFAULT_FPS_RATE
    min_frames: int = DEFAULT_MIN_FRAMES
    max_frames: int = DEFAULT_MAX_FRAMES
    frame_width: int = DEFAULT_FRAME_WIDTH
    ffmpeg_path: str = "ffmpeg"
    ffmpeg_timeout: int = 600  # 10 minutes for long videos at 1fps


@dataclass
class ExtractedFrame:
    """Represents an extracted frame with metadata.

    Attributes:
        index: 1-based frame number
        timestamp: Timestamp in seconds from video start
    """

    index: int
    timestamp: float


class FrameExtractor:
    """Extract and cache frames from video files using FFmpeg.

    Uses single-pass FFmpeg extraction with fps filter for efficient
    dense frame extraction. Supports caching with validation for
    duration, resolution, and fps rate changes.

    Example:
        config = FrameExtractionConfig(fps_rate=1.0)
        extractor = FrameExtractor(config, cache_dir="./frames")

        frames = extractor.get_or_extract_frames(
            scene_id="123",
            video_path="/path/to/video.mp4",
            duration=300.0,
        )

        # Get frames as base64 for VLM input
        frames_b64 = extractor.get_frames_base64("123")
    """

    def __init__(
        self,
        config: FrameExtractionConfig,
        cache_dir: str,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize the frame extractor.

        Args:
            config: Frame extraction configuration
            cache_dir: Directory for caching extracted frames
            log_callback: Optional callback for logging (message, level)
            progress_callback: Optional callback for progress (current, total)
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda current, total: None)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_scene_cache_dir(self, scene_id: str) -> Path:
        """Get cache directory for a scene.

        Args:
            scene_id: Scene identifier

        Returns:
            Path to scene's cache directory
        """
        return self.cache_dir / f"scene_{scene_id}"

    def get_or_extract_frames(
        self,
        scene_id: str,
        video_path: str,
        duration: float,
    ) -> list[ExtractedFrame]:
        """Get cached frames or extract new ones.

        Validates cached frames against current configuration and video
        properties. Re-extracts if any mismatch is detected.

        Args:
            scene_id: Scene ID for cache key
            video_path: Path to the video file
            duration: Video duration in seconds

        Returns:
            List of ExtractedFrame objects with index and timestamp
        """
        scene_cache = self.get_scene_cache_dir(scene_id)
        info_path = scene_cache / "frame_info.json"

        # Check cache validity
        if info_path.exists():
            try:
                with open(info_path) as f:
                    data = json.load(f)

                # Validate cached data against current config/video
                validation_result = self._validate_cache(data, duration)
                if validation_result is None:
                    # Cache is valid
                    frames = [ExtractedFrame(**f) for f in data["frames"]]
                    self.log(
                        f"Loaded {len(frames)} cached frames for scene {scene_id} "
                        f"({data.get('fps_rate', 'unknown')} fps, {data.get('frame_width', 'unknown')}px)",
                        "debug",
                    )
                    return frames
                else:
                    # Cache invalid - log reason and re-extract
                    self.log(validation_result, "info")
                    self.clear_cache(scene_id)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.log(f"Cache corrupted for scene {scene_id}: {e}", "warning")
                self.clear_cache(scene_id)

        # Extract new frames
        return self._extract_and_cache(scene_id, video_path, duration)

    def get_or_extract_frames_flexible(
        self,
        scene_id: str,
        video_path: str,
        duration: float,
        min_required_frames: int = 1,
        preferred_resolution: int | None = None,
        preferred_fps: float | None = None,
        force_extract: bool = False,
    ) -> list[ExtractedFrame]:
        """Get cached frames if sufficient, or extract new ones.

        More lenient than get_or_extract_frames() - accepts cached frames
        even if count differs, as long as minimum requirements are met.

        Args:
            scene_id: Scene ID for cache key
            video_path: Path to the video file
            duration: Video duration in seconds
            min_required_frames: Minimum frames needed
            preferred_resolution: If set, re-extract if cache differs
            preferred_fps: If set, re-extract if cache differs
            force_extract: If True, always re-extract

        Returns:
            List of ExtractedFrame objects
        """
        if force_extract:
            self.log(f"Force extraction requested for scene {scene_id}", "debug")
            self.clear_cache(scene_id)
            return self._extract_and_cache(
                scene_id,
                video_path,
                duration,
                override_resolution=preferred_resolution,
                override_fps=preferred_fps,
            )

        # Check existing cache
        cache_info = self.get_cached_frame_info(scene_id)

        if cache_info is None:
            self.log(f"No cached frames for scene {scene_id}, extracting...", "debug")
            self.clear_cache(scene_id)
            return self._extract_and_cache(
                scene_id,
                video_path,
                duration,
                override_resolution=preferred_resolution,
                override_fps=preferred_fps,
            )

        cached_count = cache_info["frame_count"]
        cached_width = cache_info["frame_width"]
        cached_fps = cache_info.get("fps_rate", self.config.fps_rate)
        cached_duration = cache_info.get("duration", 0.0)

        # Check duration mismatch (video may have been replaced)
        if abs(cached_duration - duration) > 1.0:
            self.log(
                f"Duration mismatch: cached {cached_duration:.1f}s vs current {duration:.1f}s, re-extracting",
                "info",
            )
            self.clear_cache(scene_id)
            return self._extract_and_cache(
                scene_id,
                video_path,
                duration,
                override_resolution=preferred_resolution,
                override_fps=preferred_fps,
            )

        # Check resolution match
        if preferred_resolution is not None and cached_width != preferred_resolution:
            self.log(
                f"Resolution mismatch: cached {cached_width}px vs preferred {preferred_resolution}px, re-extracting",
                "info",
            )
            self.clear_cache(scene_id)
            return self._extract_and_cache(
                scene_id,
                video_path,
                duration,
                override_resolution=preferred_resolution,
                override_fps=preferred_fps,
            )

        # Check fps match
        if preferred_fps is not None and abs(cached_fps - preferred_fps) > 0.01:
            self.log(
                f"FPS mismatch: cached {cached_fps} vs preferred {preferred_fps}, re-extracting",
                "info",
            )
            self.clear_cache(scene_id)
            return self._extract_and_cache(
                scene_id,
                video_path,
                duration,
                override_resolution=preferred_resolution,
                override_fps=preferred_fps,
            )

        # Check frame count sufficiency
        if cached_count < min_required_frames:
            self.log(
                f"Insufficient cached frames: {cached_count} < {min_required_frames} required, re-extracting",
                "info",
            )
            self.clear_cache(scene_id)
            return self._extract_and_cache(
                scene_id,
                video_path,
                duration,
                override_resolution=preferred_resolution,
                override_fps=preferred_fps,
            )

        # Cache is usable
        self.log(
            f"Using {cached_count} cached frames for scene {scene_id} "
            f"({cached_fps} fps, {cached_width}px)",
            "debug",
        )
        return cast("list[ExtractedFrame]", cache_info["frames"])

    def extract_frame_at_timestamp(
        self,
        video_path: str,
        timestamp: float,
        output_path: str | None = None,
    ) -> bytes | None:
        """Extract a single frame at a specific timestamp.

        Useful for extracting frames at arbitrary positions without
        full scene extraction.

        Args:
            video_path: Path to the video file
            timestamp: Timestamp in seconds to extract frame at
            output_path: Optional path to save frame (if None, returns bytes)

        Returns:
            Frame image bytes if output_path is None, otherwise None
        """
        import tempfile

        # Use temp file if no output path specified
        if output_path is None:
            fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            save_path = temp_path
        else:
            save_path = output_path

        cmd = [
            self.config.ffmpeg_path,
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-vf",
            f"scale={self.config.frame_width}:-1",
            "-q:v",
            "2",
            "-y",
            save_path,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=30,
            )

            if output_path is None:
                # Read and return bytes
                with open(save_path, "rb") as f:
                    frame_bytes = f.read()
                os.unlink(save_path)
                return frame_bytes
            return None

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.log(f"Failed to extract frame at {timestamp}s: {e}", "warning")
            if output_path is None and os.path.exists(save_path):
                os.unlink(save_path)
            return None

    def get_frames_base64(self, scene_id: str) -> list[str]:
        """Load all extracted frames as base64 for VLM input.

        Args:
            scene_id: Scene ID for cache lookup

        Returns:
            List of base64-encoded frame images

        Raises:
            FileNotFoundError: If no cached frames exist
        """
        scene_cache = self.get_scene_cache_dir(scene_id)
        info_path = scene_cache / "frame_info.json"

        if not info_path.exists():
            raise FileNotFoundError(f"No cached frames for scene {scene_id}")

        with open(info_path) as f:
            data = json.load(f)

        frames_base64 = []
        for frame_data in data["frames"]:
            frame_num = frame_data["index"]
            frame_path = scene_cache / f"frame_{frame_num:04d}.jpg"
            if frame_path.exists():
                with open(frame_path, "rb") as f:
                    frames_base64.append(base64.b64encode(f.read()).decode("utf-8"))

        self.log(f"Loaded {len(frames_base64)} frames as base64", "debug")
        return frames_base64

    def get_frames_as_grid(
        self,
        scene_id: str,
        max_cols: int = 4,
        quality: int = 85,
    ) -> str | None:
        """Combine all frames into a single grid image for single-image VLMs.

        Args:
            scene_id: Scene ID for cache lookup
            max_cols: Maximum columns in the grid
            quality: JPEG quality for output (1-100)

        Returns:
            Base64-encoded grid image, or None if failed
        """
        try:
            import io

            from PIL import Image
        except ImportError:
            self.log("PIL not available for grid creation", "warning")
            return None

        frame_paths = self.get_frame_paths(scene_id)
        if not frame_paths:
            return None

        # Load all frame images
        images = []
        for path in frame_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                self.log(f"Failed to load frame {path}: {e}", "warning")

        if not images:
            return None

        # Calculate grid dimensions
        num_images = len(images)
        cols = min(max_cols, num_images)
        rows = (num_images + cols - 1) // cols

        # Get dimensions (assume all frames are same size)
        frame_width, frame_height = images[0].size

        # Create grid canvas
        grid_width = cols * frame_width
        grid_height = rows * frame_height
        grid = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))

        # Paste frames into grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * frame_width
            y = row * frame_height
            grid.paste(img, (x, y))

        # Convert to base64
        buffer = io.BytesIO()
        grid.save(buffer, format="JPEG", quality=quality)
        grid_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        self.log(f"Created {cols}x{rows} grid from {num_images} frames", "debug")
        return grid_base64

    def get_frame_paths(self, scene_id: str) -> list[str]:
        """Get file paths for cached frames.

        Args:
            scene_id: Scene ID for cache lookup

        Returns:
            List of file paths to frame images, sorted by frame number
        """
        scene_cache = self.get_scene_cache_dir(scene_id)
        if not scene_cache.exists():
            return []

        # Find all frame files
        frames = list(scene_cache.glob("frame_*.jpg")) + list(scene_cache.glob("frame_*.png"))

        # Sort by frame number
        def extract_frame_num(path: Path) -> int:
            match = re.search(r"frame_(\d+)", path.name)
            return int(match.group(1)) if match else 0

        frames.sort(key=extract_frame_num)
        return [str(f) for f in frames]

    def get_cached_frame_count(self, scene_id: str) -> int | None:
        """Get the number of cached frames for a scene.

        Args:
            scene_id: Scene ID for cache lookup

        Returns:
            Number of cached frames, or None if no cache exists
        """
        info = self.get_cached_frame_info(scene_id)
        return info["frame_count"] if info else None

    def get_cached_frame_info(self, scene_id: str) -> dict[str, Any] | None:
        """Get cached frame information without triggering extraction.

        Args:
            scene_id: Scene ID for cache lookup

        Returns:
            Dict with cache info or None if no cache exists:
            {
                "frame_count": int,
                "frame_width": int,
                "fps_rate": float,
                "frames": List[ExtractedFrame],
                "duration": float,
                "video_path": str,
            }
        """
        scene_cache = self.get_scene_cache_dir(scene_id)
        info_path = scene_cache / "frame_info.json"

        if not info_path.exists():
            return None

        try:
            with open(info_path) as f:
                data = json.load(f)

            frames = [ExtractedFrame(**f) for f in data.get("frames", [])]
            return {
                "frame_count": len(frames),
                "frame_width": data.get("frame_width", DEFAULT_FRAME_WIDTH),
                "fps_rate": data.get("fps_rate", self.config.fps_rate),
                "frames": frames,
                "duration": data.get("duration", 0.0),
                "video_path": data.get("video_path", ""),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def has_cached_frames(self, scene_id: str) -> bool:
        """Check if scene has cached frames.

        Args:
            scene_id: Scene ID to check

        Returns:
            True if cached frames exist
        """
        return bool(self.get_frame_paths(scene_id))

    def clear_cache(self, scene_id: str) -> None:
        """Clear cached frames for a scene.

        Args:
            scene_id: Scene ID to clear cache for
        """
        scene_cache = self.get_scene_cache_dir(scene_id)
        if scene_cache.exists():
            shutil.rmtree(scene_cache)
            self.log(f"Cleared cache for scene {scene_id}", "debug")

    def clear_all_cache(self) -> None:
        """Clear all cached frames."""
        if self.cache_dir.exists():
            for entry in self.cache_dir.iterdir():
                if entry.is_dir() and entry.name.startswith("scene_"):
                    shutil.rmtree(entry)
            self.log("Cleared all frame cache", "info")

    def _validate_cache(
        self,
        cache_data: dict[str, Any],
        current_duration: float,
    ) -> str | None:
        """Validate cached frame data against current config and video.

        Args:
            cache_data: Loaded frame_info.json data
            current_duration: Current video duration

        Returns:
            Error message if invalid, None if valid
        """
        # Check duration mismatch (video may have been replaced)
        cached_duration = cache_data.get("duration", 0.0)
        if abs(cached_duration - current_duration) > 1.0:
            return (
                f"Duration mismatch: cached {cached_duration:.1f}s vs "
                f"current {current_duration:.1f}s, re-extracting"
            )

        # Check resolution mismatch
        cached_width = cache_data.get("frame_width", DEFAULT_FRAME_WIDTH)
        if cached_width != self.config.frame_width:
            return (
                f"Resolution mismatch: cached {cached_width}px vs "
                f"current {self.config.frame_width}px, re-extracting"
            )

        # Check fps rate mismatch
        cached_fps = cache_data.get("fps_rate")
        if cached_fps is not None and abs(cached_fps - self.config.fps_rate) > 0.01:
            return (
                f"FPS mismatch: cached {cached_fps} vs "
                f"current {self.config.fps_rate}, re-extracting"
            )

        # Check expected frame count
        cached_count = len(cache_data.get("frames", []))
        expected_count = self._calculate_frame_count(current_duration)
        # Allow some tolerance (FFmpeg may produce slightly fewer frames)
        if expected_count > 0 and cached_count < expected_count * 0.9:
            return (
                f"Frame count mismatch: cached {cached_count} vs "
                f"expected ~{expected_count}, re-extracting"
            )

        return None

    def _calculate_frame_count(self, duration: float) -> int:
        """Calculate expected frame count for a video duration.

        Args:
            duration: Video duration in seconds

        Returns:
            Expected number of frames (at least 1 for any positive duration)
        """
        # Calculate based on fps rate
        natural = int(duration * self.config.fps_rate)

        # Ensure at least 1 frame for any video with duration
        if natural == 0 and duration > 0:
            natural = 1

        # Apply min_frames if set (> 0)
        if self.config.min_frames > 0:
            natural = max(self.config.min_frames, natural)

        # Apply max_frames limit if set (> 0)
        if self.config.max_frames > 0:
            natural = min(self.config.max_frames, natural)

        return natural

    def _extract_and_cache(
        self,
        scene_id: str,
        video_path: str,
        duration: float,
        override_resolution: int | None = None,
        override_fps: float | None = None,
    ) -> list[ExtractedFrame]:
        """Extract frames using FFmpeg and save to cache.

        Uses single-pass FFmpeg extraction with fps filter for efficiency.

        Args:
            scene_id: Scene ID for cache key
            video_path: Path to the video file
            duration: Video duration in seconds
            override_resolution: If provided, use instead of config
            override_fps: If provided, use instead of config

        Returns:
            List of ExtractedFrame objects

        Raises:
            RuntimeError: If frame extraction fails
        """
        scene_cache = self.get_scene_cache_dir(scene_id)
        scene_cache.mkdir(parents=True, exist_ok=True)

        # Use overrides or fall back to config
        fps_rate = override_fps if override_fps is not None else self.config.fps_rate
        frame_width = (
            override_resolution if override_resolution is not None else self.config.frame_width
        )

        # Calculate expected frame count
        expected_frames = self._calculate_frame_count(duration)

        self.log(
            f"Extracting frames from scene {scene_id} at {fps_rate} fps "
            f"(~{expected_frames} frames, {frame_width}px)",
            "info",
        )
        self.log(f"Source: {video_path}", "debug")

        # Build FFmpeg command using fps filter (single-pass extraction)
        output_pattern = str(scene_cache / "frame_%04d.jpg")
        cmd = [
            self.config.ffmpeg_path,
            "-i",
            video_path,
            "-vf",
            f"fps={fps_rate},scale={frame_width}:-1",
            "-q:v",
            "2",  # High quality JPEG
            "-y",  # Overwrite existing files
            output_pattern,
        ]

        # Apply max_frames limit if set
        if self.config.max_frames > 0:
            cmd.insert(-1, "-frames:v")
            cmd.insert(-1, str(self.config.max_frames))

        try:
            self.log(f"Running FFmpeg for scene {scene_id}...", "debug")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.ffmpeg_timeout,
            )

            if result.returncode != 0:
                error_msg = self._parse_ffmpeg_error(result.stderr)
                self.log(f"Scene {scene_id}: {error_msg}", "error")
                raise RuntimeError(f"FFmpeg failed for scene {scene_id}: {error_msg}")

        except subprocess.TimeoutExpired as e:
            self.log(
                f"Scene {scene_id}: FFmpeg timed out after {self.config.ffmpeg_timeout}s", "error"
            )
            raise RuntimeError(f"FFmpeg timeout for scene {scene_id}") from e
        except FileNotFoundError as e:
            self.log("FFmpeg not found in PATH. Please install FFmpeg.", "error")
            raise RuntimeError("FFmpeg not found") from e

        # Get extracted frame paths
        frame_paths = self.get_frame_paths(scene_id)
        if not frame_paths:
            raise RuntimeError(f"No frames extracted for scene {scene_id}")

        # Build frame info with timestamps
        frames: list[ExtractedFrame] = []
        for i, _frame_path in enumerate(frame_paths):
            # Calculate timestamp based on frame index and fps
            timestamp = i / fps_rate
            frames.append(ExtractedFrame(index=i + 1, timestamp=round(timestamp, 3)))

        # Save frame_info.json
        frame_info = {
            "scene_id": scene_id,
            "video_path": video_path,
            "duration": duration,
            "fps_rate": fps_rate,
            "frame_width": frame_width,
            "frame_count": len(frames),
            "frames": [asdict(f) for f in frames],
        }
        with open(scene_cache / "frame_info.json", "w") as f:
            json.dump(frame_info, f, indent=2)

        self.log(
            f"Scene {scene_id}: Successfully extracted {len(frames)} frames at {fps_rate} fps",
            "info",
        )
        self.progress(len(frames), len(frames))

        return frames

    def _parse_ffmpeg_error(self, stderr: str) -> str:
        """Parse FFmpeg stderr to extract meaningful error message.

        Args:
            stderr: FFmpeg stderr output

        Returns:
            Human-readable error message
        """
        if "No such file or directory" in stderr:
            return "Cannot access video file (not found)"
        elif "Invalid data found" in stderr:
            return "Video file appears corrupted"
        elif "Permission denied" in stderr:
            return "Permission denied accessing video"
        elif "does not contain any stream" in stderr:
            return "Video file contains no video stream"
        else:
            # Extract last few meaningful lines
            error_lines = [line.strip() for line in stderr.split("\n") if line.strip()]
            if error_lines:
                return f"FFmpeg error: {' | '.join(error_lines[-3:])}"
            return "Unknown FFmpeg error"

    @staticmethod
    def check_ffmpeg(ffmpeg_path: str = "ffmpeg") -> bool:
        """Verify FFmpeg is available.

        Args:
            ffmpeg_path: Path to ffmpeg binary

        Returns:
            True if FFmpeg is available, False otherwise
        """
        try:
            subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
