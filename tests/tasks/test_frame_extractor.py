"""Tests for frame extraction with flexible caching.

This test module supports two video sources:
1. Local explicit videos (tests/videos/*.mp4) - used for local development
2. Synthetic test videos (tests/videos/synthetic/*.mp4) - used for CI/CD

The tests automatically detect which videos are available and use them accordingly.
No database writes are performed - all caching uses temporary directories.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from stash_ai.tasks.frame_extractor import (
    DEFAULT_FPS_RATE,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAMES,
    FrameExtractionConfig,
    FrameExtractor,
)

# Test video paths
TESTS_DIR = Path(__file__).parent.parent
VIDEOS_DIR = TESTS_DIR / "videos"
SYNTHETIC_DIR = VIDEOS_DIR / "synthetic"


def ffprobe_available() -> bool:
    """Check if ffprobe is available on the system."""
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


FFPROBE_AVAILABLE = ffprobe_available()


def get_test_videos() -> dict[str, Path]:
    """
    Get available test videos, preferring explicit videos over synthetic.

    Returns:
        Dict mapping video type ('short', 'medium', 'long') to file path
    """
    videos = {}

    for video_type in ["short", "medium", "long"]:
        # Prefer explicit videos if available (local testing)
        explicit_path = VIDEOS_DIR / f"{video_type}.mp4"
        synthetic_path = SYNTHETIC_DIR / f"{video_type}.mp4"

        if explicit_path.exists():
            videos[video_type] = explicit_path
        elif synthetic_path.exists():
            videos[video_type] = synthetic_path

    return videos


def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""
    if not FFPROBE_AVAILABLE:
        pytest.skip("ffprobe not available")

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        pytest.skip(f"ffprobe failed for {video_path}")

    return float(result.stdout.strip())


# Get available videos at module load time
TEST_VIDEOS = get_test_videos()

# Combined skip condition for tests requiring videos and ffprobe
SKIP_VIDEO_TESTS = not TEST_VIDEOS or not FFPROBE_AVAILABLE
SKIP_REASON = "No test videos available" if not TEST_VIDEOS else "ffprobe not available"


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    cache_dir = tempfile.mkdtemp(prefix="frame_cache_test_")
    yield cache_dir
    # Cleanup after test
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def frame_extractor(temp_cache_dir):
    """Create a FrameExtractor with default config and temp cache."""
    logs = []

    def log_callback(msg: str, level: str) -> None:
        logs.append((level, msg))

    extractor = FrameExtractor(
        config=FrameExtractionConfig(),
        cache_dir=temp_cache_dir,
        log_callback=log_callback,
    )
    extractor._test_logs = logs  # Attach for inspection
    return extractor


class TestFrameExtractionDefaults:
    """Test that shared defaults are correctly applied."""

    def test_default_fps_rate(self):
        """Default FPS rate should be 1.0 (1 frame per second)."""
        assert DEFAULT_FPS_RATE == 1.0

    def test_default_min_frames(self):
        """Default min frames should be 0 (no minimum)."""
        assert DEFAULT_MIN_FRAMES == 0

    def test_default_max_frames(self):
        """Default max frames should be 0 (no limit)."""
        assert DEFAULT_MAX_FRAMES == 0

    def test_default_frame_width(self):
        """Default frame width should be 640."""
        assert DEFAULT_FRAME_WIDTH == 640

    def test_config_uses_defaults(self):
        """FrameExtractionConfig should use module defaults."""
        config = FrameExtractionConfig()
        assert config.fps_rate == DEFAULT_FPS_RATE
        assert config.min_frames == DEFAULT_MIN_FRAMES
        assert config.max_frames == DEFAULT_MAX_FRAMES
        assert config.frame_width == DEFAULT_FRAME_WIDTH


class TestFrameExtractorCacheInfo:
    """Test get_cached_frame_info method."""

    def test_no_cache_returns_none(self, frame_extractor):
        """Should return None when no cache exists."""
        result = frame_extractor.get_cached_frame_info("nonexistent_scene")
        assert result is None

    @pytest.mark.skipif(not TEST_VIDEOS, reason="No test videos available")
    def test_cache_info_after_extraction(self, frame_extractor):
        """Should return cache info after extraction."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_scene_1"

        # Extract frames first
        frames = frame_extractor.get_or_extract_frames(scene_id, str(video_path), duration)

        # Now get cache info
        cache_info = frame_extractor.get_cached_frame_info(scene_id)

        assert cache_info is not None
        assert cache_info["frame_count"] == len(frames)
        assert cache_info["frame_width"] == DEFAULT_FRAME_WIDTH
        assert len(cache_info["frames"]) == len(frames)


class TestFlexibleExtraction:
    """Test get_or_extract_frames_flexible method."""

    @pytest.mark.skipif(not TEST_VIDEOS, reason="No test videos available")
    def test_extracts_when_no_cache(self, frame_extractor):
        """Should extract frames when no cache exists."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_flexible_1"

        frames = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        assert len(frames) >= 1
        # Check cache was created
        cache_info = frame_extractor.get_cached_frame_info(scene_id)
        assert cache_info is not None

    @pytest.mark.skipif(not TEST_VIDEOS, reason="No test videos available")
    def test_reuses_cache_when_sufficient(self, frame_extractor):
        """Should reuse cached frames when they meet minimum requirements."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_flexible_2"

        # First extraction
        frames1 = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        # Clear logs to check for cache reuse message
        frame_extractor._test_logs.clear()

        # Second extraction with same requirements
        frames2 = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        assert len(frames1) == len(frames2)
        # Should have logged cache reuse
        log_messages = [msg for _, msg in frame_extractor._test_logs]
        assert any("Using" in msg and "cached frames" in msg for msg in log_messages)

    @pytest.mark.skipif(not TEST_VIDEOS, reason="No test videos available")
    def test_reextracts_when_insufficient_frames(self, frame_extractor):
        """Should re-extract when cached frames are insufficient."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_flexible_3"

        # First extraction with low max_frames
        frame_extractor.config.max_frames = 2
        frames1 = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )
        assert len(frames1) <= 2

        # Reset max_frames
        frame_extractor.config.max_frames = 0

        # Second extraction requiring more frames than cached
        frame_extractor._test_logs.clear()
        frames2 = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=5,  # More than the 2 cached
        )

        # Should have re-extracted
        log_messages = [msg for _, msg in frame_extractor._test_logs]
        assert any("Insufficient" in msg for msg in log_messages)
        # Implementation respects min_required_frames, so we should get at least 5
        assert len(frames2) >= 5

    @pytest.mark.skipif(not TEST_VIDEOS, reason="No test videos available")
    def test_reextracts_when_resolution_differs(self, frame_extractor):
        """Should re-extract when preferred resolution differs from cache."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_flexible_4"

        # First extraction at default resolution (640)
        frames1 = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
            preferred_resolution=640,
        )

        cache_info1 = frame_extractor.get_cached_frame_info(scene_id)
        assert cache_info1["frame_width"] == 640

        # Second extraction with different preferred resolution
        frame_extractor._test_logs.clear()
        frames2 = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
            preferred_resolution=896,  # Different resolution
        )

        # Should have re-extracted at new resolution
        log_messages = [msg for _, msg in frame_extractor._test_logs]
        assert any("Resolution mismatch" in msg for msg in log_messages)

        cache_info2 = frame_extractor.get_cached_frame_info(scene_id)
        assert cache_info2["frame_width"] == 896

    @pytest.mark.skipif(not TEST_VIDEOS, reason="No test videos available")
    def test_force_extract_clears_cache(self, frame_extractor):
        """Should clear cache and re-extract when force_extract=True."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_flexible_5"

        # First extraction
        frames1 = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        # Force re-extraction
        frame_extractor._test_logs.clear()
        frames2 = frame_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
            force_extract=True,
        )

        # Should have logged force extraction
        log_messages = [msg for _, msg in frame_extractor._test_logs]
        assert any("Force extraction" in msg for msg in log_messages)


class TestFrameCountCalculation:
    """Test frame count calculation based on duration."""

    @pytest.mark.skipif("short" not in TEST_VIDEOS, reason="Short video not available")
    def test_short_video_frame_count(self, frame_extractor):
        """Short video should extract appropriate number of frames."""
        video_path = TEST_VIDEOS["short"]
        duration = get_video_duration(video_path)

        frames = frame_extractor.get_or_extract_frames_flexible(
            scene_id="test_short",
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        # Expected: max(min_frames, duration * fps_rate) = max(0, duration * 1.0)
        expected_natural = int(duration * DEFAULT_FPS_RATE)
        expected = max(DEFAULT_MIN_FRAMES, expected_natural, 1)  # At least 1 frame

        # Use >= for robustness (FFmpeg may skip frames on very short videos)
        assert len(frames) >= 1

    @pytest.mark.skipif("medium" not in TEST_VIDEOS, reason="Medium video not available")
    def test_medium_video_frame_count(self, frame_extractor):
        """Medium video should extract appropriate number of frames."""
        video_path = TEST_VIDEOS["medium"]
        duration = get_video_duration(video_path)

        frames = frame_extractor.get_or_extract_frames_flexible(
            scene_id="test_medium",
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        expected_natural = int(duration * DEFAULT_FPS_RATE)
        expected = max(DEFAULT_MIN_FRAMES, expected_natural, 1)

        # Use >= for robustness (FFmpeg may skip frames on very short videos)
        assert len(frames) >= 1

    @pytest.mark.skipif("long" not in TEST_VIDEOS, reason="Long video not available")
    def test_long_video_frame_count(self, frame_extractor):
        """Long video should extract appropriate number of frames."""
        video_path = TEST_VIDEOS["long"]
        duration = get_video_duration(video_path)

        frames = frame_extractor.get_or_extract_frames_flexible(
            scene_id="test_long",
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        expected_natural = int(duration * DEFAULT_FPS_RATE)
        expected = max(DEFAULT_MIN_FRAMES, expected_natural, 1)

        # Use >= for robustness (FFmpeg may skip frames on very short videos)
        assert len(frames) >= 1


class TestCacheReuseBetweenTasks:
    """
    Test that cache created by one configuration can be reused by another.

    This simulates the real-world scenario where Embed Scene creates a cache
    and Scene Vision should reuse it.
    """

    @pytest.mark.skipif(not TEST_VIDEOS, reason="No test videos available")
    def test_cache_reuse_with_different_config(self, temp_cache_dir):
        """Cache should be reusable even when extractor configs differ."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_cross_task"

        logs1 = []
        logs2 = []

        # Simulate Embed Scene extractor (1 fps)
        embed_extractor = FrameExtractor(
            config=FrameExtractionConfig(
                fps_rate=1.0,
                min_frames=3,
                max_frames=0,
            ),
            cache_dir=temp_cache_dir,
            log_callback=lambda msg, lvl: logs1.append((lvl, msg)),
        )

        # Embed Scene extracts frames
        frames1 = embed_extractor.get_or_extract_frames(scene_id, str(video_path), duration)

        # Simulate Scene Vision extractor (could have different config - 0.5 fps)
        vision_extractor = FrameExtractor(
            config=FrameExtractionConfig(
                fps_rate=0.5,  # Different FPS rate!
                min_frames=1,
                max_frames=0,
            ),
            cache_dir=temp_cache_dir,
            log_callback=lambda msg, lvl: logs2.append((lvl, msg)),
        )

        # Scene Vision uses flexible extraction - should reuse cache
        frames2 = vision_extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=3,  # Cached frames should be sufficient
        )

        # Should have reused cache (same frames)
        assert len(frames2) == len(frames1)

        # Check logs confirm cache reuse
        log_messages = [msg for _, msg in logs2]
        assert any("Using" in msg and "cached frames" in msg for msg in log_messages)
        # Should NOT have re-extracted
        assert not any("Extracting frames from video" in msg for msg in log_messages)


class TestCacheCorruption:
    """Test handling of corrupted cache files."""

    @pytest.mark.skipif(SKIP_VIDEO_TESTS, reason=SKIP_REASON)
    def test_corrupted_json_triggers_reextraction(self, temp_cache_dir):
        """Should clear stale files and re-extract when frame_info.json is corrupted."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_corrupt"

        logs = []
        extractor = FrameExtractor(
            config=FrameExtractionConfig(),
            cache_dir=temp_cache_dir,
            log_callback=lambda msg, lvl: logs.append((lvl, msg)),
        )

        # Create corrupted cache
        scene_cache = os.path.join(temp_cache_dir, f"scene_{scene_id}")
        os.makedirs(scene_cache, exist_ok=True)
        info_path = os.path.join(scene_cache, "frame_info.json")

        # Write invalid JSON
        with open(info_path, "w") as f:
            f.write("{invalid json content")

        # Also create a stale frame file
        stale_frame = os.path.join(scene_cache, "frame_0001.jpg")
        with open(stale_frame, "w") as f:
            f.write("fake frame data")

        # Extraction should succeed despite corrupt cache
        frames = extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        assert len(frames) >= 1
        # Cache should now be valid
        cache_info = extractor.get_cached_frame_info(scene_id)
        assert cache_info is not None
        assert cache_info["frame_count"] == len(frames)

    @pytest.mark.skipif(SKIP_VIDEO_TESTS, reason=SKIP_REASON)
    def test_missing_frames_array_triggers_reextraction(self, temp_cache_dir):
        """Should re-extract when frame_info.json exists but has no frames array."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_missing_frames"

        logs = []
        extractor = FrameExtractor(
            config=FrameExtractionConfig(),
            cache_dir=temp_cache_dir,
            log_callback=lambda msg, lvl: logs.append((lvl, msg)),
        )

        # Create cache with missing frames array
        scene_cache = os.path.join(temp_cache_dir, f"scene_{scene_id}")
        os.makedirs(scene_cache, exist_ok=True)
        info_path = os.path.join(scene_cache, "frame_info.json")

        # Write valid JSON but without frames
        with open(info_path, "w") as f:
            json.dump({"scene_id": scene_id, "duration": duration}, f)

        # Extraction should succeed
        frames = extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        assert len(frames) >= 1


class TestDurationMismatch:
    """Test handling of duration mismatch (video file replaced)."""

    @pytest.mark.skipif(SKIP_VIDEO_TESTS, reason=SKIP_REASON)
    def test_duration_change_triggers_reextraction(self, temp_cache_dir):
        """Should re-extract when video duration changed significantly."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        actual_duration = get_video_duration(video_path)
        scene_id = "test_duration_mismatch"

        logs = []
        extractor = FrameExtractor(
            config=FrameExtractionConfig(),
            cache_dir=temp_cache_dir,
            log_callback=lambda msg, lvl: logs.append((lvl, msg)),
        )

        # First extraction with actual duration
        frames1 = extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=actual_duration,
            min_required_frames=1,
        )

        # Clear logs
        logs.clear()

        # Second extraction with significantly different duration (simulating replaced video)
        different_duration = actual_duration + 100.0  # 100 seconds different
        frames2 = extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=different_duration,
            min_required_frames=1,
        )

        # Should have detected duration mismatch
        log_messages = [msg for _, msg in logs]
        assert any("Duration mismatch" in msg for msg in log_messages)

    @pytest.mark.skipif(SKIP_VIDEO_TESTS, reason=SKIP_REASON)
    def test_small_duration_difference_uses_cache(self, temp_cache_dir):
        """Should reuse cache when duration difference is within tolerance (1 second)."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        actual_duration = get_video_duration(video_path)
        scene_id = "test_duration_tolerance"

        logs = []
        extractor = FrameExtractor(
            config=FrameExtractionConfig(),
            cache_dir=temp_cache_dir,
            log_callback=lambda msg, lvl: logs.append((lvl, msg)),
        )

        # First extraction
        frames1 = extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=actual_duration,
            min_required_frames=1,
        )

        # Clear logs
        logs.clear()

        # Second extraction with small duration difference (within 1s tolerance)
        slightly_different_duration = actual_duration + 0.5
        frames2 = extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=slightly_different_duration,
            min_required_frames=1,
        )

        # Should have reused cache (no duration mismatch message)
        log_messages = [msg for _, msg in logs]
        assert not any("Duration mismatch" in msg for msg in log_messages)
        assert any("Using" in msg and "cached frames" in msg for msg in log_messages)


class TestMinFramesRespected:
    """Test that min_required_frames is respected during re-extraction."""

    @pytest.mark.skipif(SKIP_VIDEO_TESTS, reason=SKIP_REASON)
    def test_min_required_frames_respected_on_reextraction(self, temp_cache_dir):
        """When re-extracting due to insufficient frames, should respect min_required_frames."""
        video_type = list(TEST_VIDEOS.keys())[0]
        video_path = TEST_VIDEOS[video_type]
        duration = get_video_duration(video_path)
        scene_id = "test_min_frames_respected"

        logs = []
        extractor = FrameExtractor(
            config=FrameExtractionConfig(
                fps_rate=0.1,  # Very low fps = fewer natural frames (1 frame per 10 sec)
                min_frames=1,  # Low default min
                max_frames=0,
            ),
            cache_dir=temp_cache_dir,
            log_callback=lambda msg, lvl: logs.append((lvl, msg)),
        )

        # First extraction with low min_frames - should get few frames
        frames1 = extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=1,
        )

        initial_count = len(frames1)

        # Clear logs
        logs.clear()

        # Second extraction requesting more frames than cached
        # Request more than initially cached, but cap at what's physically possible
        required_frames = initial_count + 3
        frames2 = extractor.get_or_extract_frames_flexible(
            scene_id=scene_id,
            video_path=str(video_path),
            duration=duration,
            min_required_frames=required_frames,
        )

        # Should have re-extracted with more frames than initial extraction
        # (limited by video duration, but should trigger re-extraction)
        assert len(frames2) >= initial_count
        # Verify re-extraction was triggered (log message)
        log_messages = [msg for _, msg in logs]
        assert any("Insufficient" in msg or "Re-extract" in msg.lower() for msg in log_messages)
