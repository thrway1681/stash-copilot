#!/usr/bin/env python3
"""
Standalone embedding script for running scene embeddings on a remote machine.

This script allows you to run GPU-intensive embedding tasks on a machine with
a better GPU while storing results in the Stash server's database.

Usage:
    # Basic usage (assumes paths are mounted at same locations)
    python standalone_embed.py

    # Specify paths explicitly
    python standalone_embed.py \
        --stash-db /mnt/stash/stash-go.sqlite \
        --output-db /mnt/stash/plugins/stash-copilot/assets/stash_copilot.sqlite \
        --cache-dir /mnt/stash/plugins/stash-copilot/assets/embedded_frames

    # Remap video paths (server path -> local mount)
    python standalone_embed.py \
        --path-remap "/data/videos=/mnt/videos" \
        --path-remap "/media/library=/mnt/library"

    # Use specific model
    python standalone_embed.py --model ViT-H-14 --provider openclip

    # Force re-embed all scenes
    python standalone_embed.py --force

    # Embed specific scenes
    python standalone_embed.py --scene-ids 123 456 789

    # Extract frames at 1fps during embedding
    python standalone_embed.py --extract-frames --fps 1.0

    # Extract frames at 2fps without confirmation prompts
    python standalone_embed.py --extract-frames --fps 2.0 --yes

    # Extract frames, skip fps rate consistency check
    python standalone_embed.py --extract-frames --fps 0.5 --skip-rate-check
"""

import argparse
import json
import os
import re
import signal
import sqlite3
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import shared frame extractor
from stash_ai.tasks.frame_extractor import (
    FrameExtractionConfig,
    FrameExtractor,
)

# =============================================================================
# Logging
# =============================================================================


class Logger:
    """Thread-safe logger with timestamps and colors."""

    # ANSI color codes
    COLORS = {
        "TRACE": "\033[90m",  # Gray
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "RESET": "\033[0m",
    }

    def __init__(self, quiet: bool = False, debug: bool = False, use_color: bool = True):
        self.quiet = quiet
        self.debug_enabled = debug
        self.use_color = use_color and sys.stdout.isatty()
        self._lock = Lock()
        self._start_time = datetime.now()

    def _format_elapsed(self) -> str:
        """Format elapsed time since start."""
        elapsed = datetime.now() - self._start_time
        total_seconds = int(elapsed.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def log(self, msg: str, level: str = "info") -> None:
        """Log a message with timestamp."""
        level_upper = level.upper()

        # Filter by log level
        if self.quiet and level_upper in ("DEBUG", "TRACE"):
            return
        if not self.debug_enabled and level_upper in ("DEBUG", "TRACE"):
            return

        with self._lock:
            elapsed = self._format_elapsed()

            if self.use_color:
                color = self.COLORS.get(level_upper, "")
                reset = self.COLORS["RESET"]
                print(f"{color}[{elapsed}] [{level_upper:7}]{reset} {msg}")
            else:
                print(f"[{elapsed}] [{level_upper:7}] {msg}")

    def trace(self, msg: str) -> None:
        self.log(msg, "trace")

    def debug(self, msg: str) -> None:
        self.log(msg, "debug")

    def info(self, msg: str) -> None:
        self.log(msg, "info")

    def warning(self, msg: str) -> None:
        self.log(msg, "warning")

    def error(self, msg: str, exc: Exception | None = None) -> None:
        self.log(msg, "error")
        if exc and self.debug_enabled:
            with self._lock:
                traceback.print_exception(type(exc), exc, exc.__traceback__)

    def progress(self, current: int, total: int, prefix: str = "Progress") -> None:
        """Print progress on a single line."""
        if self.quiet:
            return
        pct = (current / total * 100) if total > 0 else 0
        elapsed = self._format_elapsed()
        with self._lock:
            print(f"\r[{elapsed}] {prefix}: {current}/{total} ({pct:.1f}%)", end="", flush=True)
            if current == total:
                print()


# Global stop event for graceful shutdown
_stop_event = Event()


# =============================================================================
# Benchmarking
# =============================================================================


class Benchmark:
    """Thread-safe benchmarking for tracking operation timings."""

    def __init__(self) -> None:
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    @contextmanager
    def time(self, operation: str) -> Generator[None, None, None]:
        """Context manager to time an operation."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        with self._lock:
            self.timings[operation].append(elapsed)

    def summary(self) -> str:
        """Generate a formatted summary of all timings."""
        with self._lock:
            if not self.timings:
                return "No benchmarks recorded."

            lines = []
            lines.append("")
            lines.append("=" * 70)
            lines.append("Benchmark Summary")
            lines.append("=" * 70)
            lines.append(
                f"{'Operation':<25} {'Count':>8} {'Total (s)':>12} {'Avg (s)':>10} {'% Time':>8}"
            )
            lines.append("-" * 70)

            # Calculate total time across all operations
            total_time = sum(sum(times) for times in self.timings.values())

            # Sort by total time descending
            sorted_ops = sorted(
                self.timings.items(),
                key=lambda x: sum(x[1]),
                reverse=True,
            )

            for operation, times in sorted_ops:
                count = len(times)
                op_total = sum(times)
                avg = op_total / count if count > 0 else 0.0
                pct = (op_total / total_time * 100) if total_time > 0 else 0.0

                lines.append(
                    f"{operation:<25} {count:>8} {op_total:>12.2f} {avg:>10.3f} {pct:>7.1f}%"
                )

            lines.append("-" * 70)
            lines.append(f"{'TOTAL':<25} {'':<8} {total_time:>12.2f}")
            lines.append("=" * 70)

            return "\n".join(lines)


# =============================================================================
# Frame Extraction Helpers
# =============================================================================


def scan_existing_frame_rates(cache_dir: Path, sample_size: int = 5) -> dict[float, int]:
    """Sample a few scene directories to detect existing frame rates.

    Uses random sampling instead of scanning all directories for performance.
    With ~12,000 scene directories, scanning all would be slow. Sampling 5
    is enough to detect existing fps rates for consistency checking.

    Args:
        cache_dir: Path to the embedded_frames directory
        sample_size: Number of scene directories to sample (default: 5)

    Returns:
        Dict mapping fps_rate -> count of scenes with that rate in the sample
    """
    import random

    rate_counts: dict[float, int] = defaultdict(int)

    if not cache_dir.exists():
        return dict(rate_counts)

    # Use os.listdir() - fast, just reads directory names without stat calls
    try:
        all_names = os.listdir(cache_dir)
    except OSError:
        return dict(rate_counts)

    # Filter to scene_* directories by name (no stat calls)
    scene_names = [n for n in all_names if n.startswith("scene_")]
    if not scene_names:
        return dict(rate_counts)

    # Random sample for performance
    sampled_names = random.sample(scene_names, min(sample_size, len(scene_names)))

    for name in sampled_names:
        scene_dir = cache_dir / name
        # Check both possible metadata file names
        for filename in ("frame_info.json", "metadata.json"):
            info_path = scene_dir / filename
            if info_path.exists():
                try:
                    with open(info_path) as f:
                        data = json.load(f)

                    # Check for fps_rate field (new format)
                    if "fps_rate" in data:
                        rate_counts[float(data["fps_rate"])] += 1
                        break
                    # Fallback: estimate from frame_interval if available
                    elif "frame_interval" in data:
                        interval = float(data["frame_interval"])
                        if interval > 0:
                            rate_counts[round(1.0 / interval, 2)] += 1
                        break
                except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError):
                    continue

    return dict(rate_counts)


def prompt_user_confirmation(
    message: str,
    default: bool = False,
    auto_yes: bool = False,
) -> bool:
    """Prompt user for yes/no confirmation.

    Args:
        message: The prompt message to display
        default: Default value if user just presses Enter
        auto_yes: If True, skip prompt and return True

    Returns:
        True if user confirmed, False otherwise
    """
    if auto_yes:
        return True

    suffix = "[Y/n]" if default else "[y/N]"
    try:
        response = input(f"{message} {suffix}: ").strip().lower()

        if not response:
            return default

        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def estimate_total_duration(
    db_path: Path,
    scene_ids: list[int] | None = None,
) -> tuple[float, int]:
    """Estimate total video duration for scenes.

    Args:
        db_path: Path to Stash database
        scene_ids: Optional list of specific scene IDs (None = all scenes)

    Returns:
        Tuple of (total_duration_seconds, scene_count)
    """
    if not db_path.exists():
        return 0.0, 0

    conn = get_readonly_connection(db_path)
    cursor = conn.cursor()

    if scene_ids:
        placeholders = ",".join("?" * len(scene_ids))
        cursor.execute(
            f"""
            SELECT SUM(vf.duration), COUNT(*)
            FROM scenes s
            JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
            JOIN files f ON sf.file_id = f.id
            JOIN video_files vf ON f.id = vf.file_id
            WHERE s.id IN ({placeholders})
            """,
            scene_ids,
        )
    else:
        cursor.execute(
            """
            SELECT SUM(vf.duration), COUNT(*)
            FROM scenes s
            JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
            JOIN files f ON sf.file_id = f.id
            JOIN video_files vf ON f.id = vf.file_id
            """
        )

    row = cursor.fetchone()
    conn.close()

    total_duration = row[0] if row[0] else 0.0
    scene_count = row[1] if row[1] else 0

    return total_duration, scene_count


def format_duration_human(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable string like "2h 30m" or "45m"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_storage_size(bytes_size: float) -> str:
    """Format storage size in human-readable format.

    Args:
        bytes_size: Size in bytes

    Returns:
        Human-readable string like "450 MB" or "1.2 GB"
    """
    if bytes_size >= 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024 * 1024):.1f} GB"
    elif bytes_size >= 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.0f} MB"
    elif bytes_size >= 1024:
        return f"{bytes_size / 1024:.0f} KB"
    return f"{bytes_size:.0f} B"


# =============================================================================
# Frame Validation
# =============================================================================


@dataclass
class FrameValidationResult:
    """Result of frame count validation for a scene."""

    scene_id: int
    video_duration: float
    expected_frames: int
    cached_frames: int
    db_embeddings: int
    needs_reprocessing: bool
    reason: str  # Why reprocessing is needed (empty if not needed)


def get_scene_duration(db_path: Path, scene_id: int) -> float | None:
    """Get video duration for a scene from the Stash database.

    Args:
        db_path: Path to Stash database
        scene_id: Scene ID to look up

    Returns:
        Video duration in seconds, or None if scene not found
    """
    if not db_path.exists():
        return None

    conn = get_readonly_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT vf.duration
        FROM scenes s
        JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
        JOIN files f ON sf.file_id = f.id
        JOIN video_files vf ON f.id = vf.file_id
        WHERE s.id = ?
        """,
        (scene_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return float(row[0]) if row[0] is not None else None


def count_cached_frames(cache_dir: Path, scene_id: int) -> int:
    """Count the number of cached frame files for a scene.

    Args:
        cache_dir: Path to vision cache directory
        scene_id: Scene ID

    Returns:
        Number of frame files in the cache directory
    """
    scene_dir = cache_dir / f"scene_{scene_id}"
    if not scene_dir.exists():
        return 0

    # Count frame_*.jpg and frame_*.png files
    jpg_frames = list(scene_dir.glob("frame_*.jpg"))
    png_frames = list(scene_dir.glob("frame_*.png"))

    return len(jpg_frames) + len(png_frames)


def validate_scene_frames(
    scene_id: int,
    stash_db: Path,
    cache_dir: Path,
    storage: "StandaloneEmbeddingStorage",
    fps_rate: float = 1.0,
) -> FrameValidationResult | None:
    """Validate frame count for a scene against expected 1fps rate.

    Args:
        scene_id: Scene ID to validate
        stash_db: Path to Stash database
        cache_dir: Path to vision cache directory
        storage: Embedding storage instance
        fps_rate: Expected frames per second rate (default: 1.0)

    Returns:
        FrameValidationResult with validation details, or None if scene not found
    """
    # Get video duration
    duration = get_scene_duration(stash_db, scene_id)
    if duration is None:
        return None

    # Calculate expected frames (floor of duration * fps)
    expected_frames = int(duration * fps_rate)
    if expected_frames < 1:
        expected_frames = 1

    # Count cached frames
    cached_frames = count_cached_frames(cache_dir, scene_id)

    # Count database embeddings
    db_embeddings = storage.get_frame_embedding_count(scene_id)

    # Determine if reprocessing is needed
    needs_reprocessing = False
    reasons: list[str] = []

    if cached_frames != expected_frames:
        needs_reprocessing = True
        reasons.append(f"cached frames mismatch (have {cached_frames}, expected {expected_frames})")

    if db_embeddings != expected_frames:
        needs_reprocessing = True
        reasons.append(f"DB embeddings mismatch (have {db_embeddings}, expected {expected_frames})")

    reason = "; ".join(reasons) if reasons else ""

    return FrameValidationResult(
        scene_id=scene_id,
        video_duration=duration,
        expected_frames=expected_frames,
        cached_frames=cached_frames,
        db_embeddings=db_embeddings,
        needs_reprocessing=needs_reprocessing,
        reason=reason,
    )


def clear_cached_frames(cache_dir: Path, scene_id: int) -> int:
    """Delete all cached frame files for a scene.

    Args:
        cache_dir: Path to vision cache directory
        scene_id: Scene ID

    Returns:
        Number of files deleted
    """
    scene_dir = cache_dir / f"scene_{scene_id}"
    if not scene_dir.exists():
        return 0

    deleted = 0

    # Delete frame files
    for frame_file in list(scene_dir.glob("frame_*.jpg")) + list(scene_dir.glob("frame_*.png")):
        try:
            frame_file.unlink()
            deleted += 1
        except OSError:
            pass

    # Delete metadata files
    for metadata_file in ["frame_info.json", "metadata.json"]:
        try:
            (scene_dir / metadata_file).unlink()
        except (OSError, FileNotFoundError):
            pass

    return deleted


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EmbedConfig:
    """Configuration for embedding generation."""

    # Visual embedding weight (0-1), metadata gets 1-visual_weight
    # Default 1.0 = visual only, no metadata influence
    visual_weight: float = 1.0

    # Reuse existing vision descriptions if available
    use_cached_descriptions: bool = True

    # Frame extraction settings (fps-based)
    fps_rate: float = 1.0  # Frames per second for extraction
    min_frames: int = 0  # No minimum (allows very short scenes)
    max_frames: int = 0  # 0 = no limit
    frame_width: int = 640

    # Whether to extract frames during embedding (vs using cached)
    extract_frames: bool = False

    # Metadata components to include
    include_tags: bool = True
    include_performers: bool = True
    include_studio: bool = True
    include_title: bool = False

    # Direct image embedding (CLIP/OpenCLIP/SigLIP)
    use_direct_image_embedding: bool = True

    # Parallel processing
    num_workers: int = 2

    # GPU batch size for embedding
    batch_size: int = 8


@dataclass
class PathConfig:
    """Path configuration for remote embedding."""

    # Stash database path (read-only)
    stash_db: Path

    # Output embedding database path (read-write)
    output_db: Path

    # Vision cache directory
    cache_dir: Path

    # Path remappings: {server_path: local_path}
    path_remaps: dict[str, str]

    def remap_path(self, path: str) -> str:
        """Remap a server path to local mount point."""
        for server_path, local_path in self.path_remaps.items():
            if path.startswith(server_path):
                remapped = path.replace(server_path, local_path, 1)
                # On Windows, convert forward slashes to backslashes
                if sys.platform == "win32":
                    remapped = remapped.replace("/", "\\")
                return remapped
        return path


# =============================================================================
# Database helpers
# =============================================================================


def get_readonly_connection(db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection to the SQLite database."""
    # Convert Windows UNC paths properly
    path_str = str(db_path)
    # SQLite URI format requires forward slashes and proper escaping
    if path_str.startswith("\\\\"):
        # UNC path: \\server\share -> file://server/share
        uri = "file:" + path_str.replace("\\", "/") + "?mode=ro"
    else:
        uri = f"file:{path_str}?mode=ro"

    try:
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError as e:
        # Provide more context for common errors
        if "unable to open database" in str(e).lower():
            raise sqlite3.OperationalError(
                f"Cannot open database: {db_path}\n  URI: {uri}\n  Original error: {e}"
            ) from e
        raise


def get_all_scene_ids(db_path: Path) -> list[int]:
    """Get all scene IDs from the database."""
    if not db_path.exists():
        return []

    conn = get_readonly_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM scenes ORDER BY id")
    ids = [row["id"] for row in cursor.fetchall()]
    conn.close()
    return ids


def get_scene_info(db_path: Path, scene_id: int) -> dict[str, Any] | None:
    """Fetch scene info from Stash database."""
    if not db_path.exists():
        return None

    conn = get_readonly_connection(db_path)
    cursor = conn.cursor()

    # Get scene basic info with video file path
    cursor.execute(
        """
        SELECT
            s.id, s.title,
            fo.path || '/' || f.basename as video_path,
            vf.duration,
            st.name as studio_name
        FROM scenes s
        JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
        JOIN files f ON sf.file_id = f.id
        JOIN folders fo ON f.parent_folder_id = fo.id
        JOIN video_files vf ON f.id = vf.file_id
        LEFT JOIN studios st ON s.studio_id = st.id
        WHERE s.id = ?
    """,
        (scene_id,),
    )

    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    scene_info: dict[str, Any] = {
        "id": row["id"],
        "title": row["title"],
        "video_path": row["video_path"],
        "duration": row["duration"],
        "studio": row["studio_name"],
        "performers": [],
        "tags": [],
    }

    # Get performers
    cursor.execute(
        """
        SELECT p.name FROM performers p
        JOIN performers_scenes ps ON p.id = ps.performer_id
        WHERE ps.scene_id = ?
    """,
        (scene_id,),
    )
    scene_info["performers"] = [row["name"] for row in cursor.fetchall()]

    # Get tags
    cursor.execute(
        """
        SELECT t.name FROM tags t
        JOIN scenes_tags st ON t.id = st.tag_id
        WHERE st.scene_id = ?
    """,
        (scene_id,),
    )
    scene_info["tags"] = [row["name"] for row in cursor.fetchall()]

    conn.close()
    return scene_info


# =============================================================================
# Embedding Storage (simplified for standalone use)
# =============================================================================


class StandaloneEmbeddingStorage:
    """SQLite-based storage for scene embeddings."""

    SCHEMA_VERSION = 4

    def __init__(self, db_path: str, model_key: str = "openclip:ViT-H-14") -> None:
        self.db_path = db_path
        self.model_key = model_key
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Schema info table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        # Check version
        cursor.execute("SELECT value FROM schema_info WHERE key = 'version'")
        row = cursor.fetchone()
        current_version = int(row["value"]) if row else 0

        # Check for legacy table that needs renaming (from older standalone_embed.py)
        # This must run unconditionally, not just during migrations
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scene_embeddings_v2'"
        )
        has_v2 = cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scene_embeddings'"
        )
        has_final = cursor.fetchone() is not None

        if has_v2 and not has_final:
            # Rename legacy table to match main plugin storage
            cursor.execute("ALTER TABLE scene_embeddings_v2 RENAME TO scene_embeddings")

        # Ensure scene_embeddings has all required columns (migrate from old schema)
        if has_v2 or has_final:
            # Check if dimensions column exists
            cursor.execute("PRAGMA table_info(scene_embeddings)")
            columns = {row[1] for row in cursor.fetchall()}

            if "dimensions" not in columns:
                # Add dimensions column with default value (will be updated on next insert)
                cursor.execute(
                    "ALTER TABLE scene_embeddings ADD COLUMN dimensions INTEGER NOT NULL DEFAULT 0"
                )

        if current_version < self.SCHEMA_VERSION:
            self._run_migrations(conn, current_version)

        conn.commit()
        conn.close()

    def _run_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Run schema migrations."""
        cursor = conn.cursor()

        if from_version < 2:
            # Create scene embeddings table (schema matches main plugin storage.py)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_embeddings (
                    scene_id INTEGER NOT NULL,
                    model_key TEXT NOT NULL DEFAULT 'openclip',
                    visual_embedding BLOB,
                    metadata_embedding BLOB,
                    composite_embedding BLOB NOT NULL,
                    visual_model TEXT,
                    text_model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    visual_description TEXT,
                    metadata_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (scene_id, model_key)
                )
            """
            )

        if from_version < 3:
            # Create frame embeddings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS frame_embeddings (
                    scene_id INTEGER NOT NULL,
                    model_key TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (scene_id, model_key, frame_index)
                )
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_frame_emb_scene_model ON frame_embeddings(scene_id, model_key)"
            )

        if from_version < 4:
            # Create frame metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS frame_embedding_metadata (
                    scene_id INTEGER NOT NULL,
                    model_key TEXT NOT NULL,
                    frame_count INTEGER NOT NULL,
                    total_frames_extracted INTEGER NOT NULL,
                    duration REAL NOT NULL,
                    sampling_rate REAL NOT NULL,
                    composite_embedding BLOB,
                    dedup_ratio REAL DEFAULT 0.0,
                    first_frame_timestamp REAL,
                    last_frame_timestamp REAL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (scene_id, model_key)
                )
            """
            )

        # Update version
        cursor.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?)",
            (str(self.SCHEMA_VERSION),),
        )

    def has_embedding(self, scene_id: int) -> bool:
        """Check if scene has an embedding."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM scene_embeddings WHERE scene_id = ? AND model_key = ?",
            (scene_id, self.model_key),
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def store_embedding(
        self,
        scene_id: int,
        composite_embedding: list[float],
        text_model: str,
        visual_embedding: list[float] | None = None,
        metadata_embedding: list[float] | None = None,
        visual_model: str | None = None,
        visual_description: str | None = None,
        metadata_text: str | None = None,
    ) -> None:
        """Store scene embedding."""
        from datetime import datetime

        conn = self._get_connection()
        cursor = conn.cursor()

        # Pack embeddings as float32 blobs
        composite_blob = np.array(composite_embedding, dtype=np.float32).tobytes()
        visual_blob = (
            np.array(visual_embedding, dtype=np.float32).tobytes() if visual_embedding else None
        )
        metadata_blob = (
            np.array(metadata_embedding, dtype=np.float32).tobytes() if metadata_embedding else None
        )

        now = datetime.now(tz=UTC).isoformat()
        dimensions = len(composite_embedding)

        cursor.execute(
            """
            INSERT OR REPLACE INTO scene_embeddings (
                scene_id, model_key, visual_embedding, metadata_embedding,
                composite_embedding, visual_model, text_model, dimensions,
                visual_description, metadata_text, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                scene_id,
                self.model_key,
                visual_blob,
                metadata_blob,
                composite_blob,
                visual_model,
                text_model,
                dimensions,
                visual_description,
                metadata_text,
                now,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def store_frame_embedding(
        self,
        scene_id: int,
        frame_index: int,
        timestamp: float,
        embedding: list[float],
    ) -> None:
        """Store individual frame embedding."""
        from datetime import UTC, datetime

        conn = self._get_connection()
        cursor = conn.cursor()

        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
        now = datetime.now(tz=UTC).isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO frame_embeddings (
                scene_id, model_key, frame_index, timestamp, embedding, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (scene_id, self.model_key, frame_index, timestamp, embedding_blob, now),
        )

        conn.commit()
        conn.close()

    def store_frame_embeddings_batch(
        self,
        scene_id: int,
        frames: list[tuple[int, float, list[float]]],
        duration: float,
        composite_embedding: list[float] | None = None,
    ) -> int:
        """Store all frame embeddings and metadata in a single transaction.

        Args:
            scene_id: Scene ID
            frames: List of (frame_index, timestamp, embedding) tuples
            duration: Video duration in seconds
            composite_embedding: Optional averaged embedding for all frames

        Returns:
            Number of frames stored
        """
        if not frames:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now(tz=UTC).isoformat()

        try:
            # Prepare frame data for batch insert
            frame_rows = []
            for frame_index, timestamp, embedding in frames:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                frame_rows.append(
                    (
                        scene_id,
                        self.model_key,
                        frame_index,
                        timestamp,
                        embedding_blob,
                        now,
                    )
                )

            # Batch insert all frames
            cursor.executemany(
                """
                INSERT OR REPLACE INTO frame_embeddings (
                    scene_id, model_key, frame_index, timestamp, embedding, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                frame_rows,
            )

            # Store metadata
            first_timestamp = frames[0][1] if frames else None
            last_timestamp = frames[-1][1] if frames else None
            composite_blob = (
                np.array(composite_embedding, dtype=np.float32).tobytes()
                if composite_embedding
                else None
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO frame_embedding_metadata (
                    scene_id, model_key, frame_count, total_frames_extracted,
                    duration, sampling_rate, composite_embedding, dedup_ratio,
                    first_frame_timestamp, last_frame_timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    scene_id,
                    self.model_key,
                    len(frames),
                    len(frames),
                    duration,
                    1.0,
                    composite_blob,
                    0.0,
                    first_timestamp,
                    last_timestamp,
                    now,
                ),
            )

            conn.commit()
            return len(frames)

        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def store_frame_metadata(
        self,
        scene_id: int,
        frame_count: int,
        total_frames_extracted: int,
        duration: float,
        sampling_rate: float,
        composite_embedding: list[float] | None = None,
        dedup_ratio: float = 0.0,
        first_frame_timestamp: float | None = None,
        last_frame_timestamp: float | None = None,
    ) -> None:
        """Store frame extraction metadata."""
        from datetime import datetime

        conn = self._get_connection()
        cursor = conn.cursor()

        composite_blob = (
            np.array(composite_embedding, dtype=np.float32).tobytes()
            if composite_embedding
            else None
        )

        now = datetime.now(tz=UTC).isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO frame_embedding_metadata (
                scene_id, model_key, frame_count, total_frames_extracted,
                duration, sampling_rate, composite_embedding, dedup_ratio,
                first_frame_timestamp, last_frame_timestamp, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                scene_id,
                self.model_key,
                frame_count,
                total_frames_extracted,
                duration,
                sampling_rate,
                composite_blob,
                dedup_ratio,
                first_frame_timestamp,
                last_frame_timestamp,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def get_frame_embedding_count(self, scene_id: int) -> int:
        """Count frame embeddings for a scene.

        Args:
            scene_id: Scene ID to count embeddings for

        Returns:
            Number of frame embeddings stored for this scene
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM frame_embeddings WHERE scene_id = ? AND model_key = ?",
            (scene_id, self.model_key),
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def delete_frame_embeddings(self, scene_id: int) -> int:
        """Delete all frame embeddings for a scene.

        Args:
            scene_id: Scene ID to delete embeddings for

        Returns:
            Number of frame embeddings deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete frame embeddings
        cursor.execute(
            "DELETE FROM frame_embeddings WHERE scene_id = ? AND model_key = ?",
            (scene_id, self.model_key),
        )
        deleted = cursor.rowcount

        # Delete frame metadata
        cursor.execute(
            "DELETE FROM frame_embedding_metadata WHERE scene_id = ? AND model_key = ?",
            (scene_id, self.model_key),
        )

        conn.commit()
        conn.close()
        return deleted


# =============================================================================
# Frame Validation Task
# =============================================================================


class FrameValidationTask:
    """Task to validate and fix frame counts across all scenes.

    Validates that each scene has the correct number of frames based on
    1fps extraction rate. Scenes with incorrect frame counts (in cache
    or database) are re-extracted and re-embedded.
    """

    def __init__(
        self,
        path_config: PathConfig,
        embed_config: EmbedConfig,
        storage: StandaloneEmbeddingStorage,
        frame_extractor: FrameExtractor,
        logger: Logger,
        dry_run: bool = False,
    ) -> None:
        self.path_config = path_config
        self.embed_config = embed_config
        self.storage = storage
        self.frame_extractor = frame_extractor
        self.logger = logger
        self.dry_run = dry_run

        # For re-embedding, we need a lazy embedder reference
        self._embed_task: StandaloneEmbedTask | None = None

    def _get_embed_task(
        self,
        embedding_provider: str,
        embedding_model: str,
        embedding_device: str,
        pretrained: str | None = None,
    ) -> "StandaloneEmbedTask":
        """Get or create an embedding task for re-embedding scenes."""
        if self._embed_task is None:
            # Create an embed config with extract_frames enabled
            embed_config = EmbedConfig(
                visual_weight=self.embed_config.visual_weight,
                fps_rate=self.embed_config.fps_rate,
                min_frames=self.embed_config.min_frames,
                max_frames=self.embed_config.max_frames,
                frame_width=self.embed_config.frame_width,
                num_workers=1,  # Single worker for validation
                batch_size=self.embed_config.batch_size,
                extract_frames=True,  # Enable extraction
            )

            self._embed_task = StandaloneEmbedTask(
                path_config=self.path_config,
                embed_config=embed_config,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
                pretrained=pretrained,
                log_callback=lambda msg, level: self.logger.log(msg, level),
                progress_callback=lambda cur, total: None,
            )

        return self._embed_task

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._embed_task is not None:
            self._embed_task.cleanup()
            self._embed_task = None

    def validate_all(
        self,
        scene_ids: list[int] | None = None,
        embedding_provider: str = "openclip",
        embedding_model: str = "ViT-H-14",
        embedding_device: str = "cuda",
        pretrained: str | None = None,
    ) -> dict[str, Any]:
        """Validate all scenes and optionally fix incorrect frame counts.

        Args:
            scene_ids: Specific scene IDs to validate (None = all scenes)
            embedding_provider: Provider for re-embedding
            embedding_model: Model for re-embedding
            embedding_device: Device for re-embedding
            pretrained: Pretrained weights for re-embedding

        Returns:
            Summary dictionary with validation results
        """
        try:
            return self._validate_all_impl(
                scene_ids=scene_ids,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
                pretrained=pretrained,
            )
        finally:
            self.cleanup()

    def _validate_all_impl(
        self,
        scene_ids: list[int] | None,
        embedding_provider: str,
        embedding_model: str,
        embedding_device: str,
        pretrained: str | None,
    ) -> dict[str, Any]:
        """Internal implementation of validate_all."""
        # Get scene IDs
        if scene_ids is None:
            self.logger.info("Fetching scene IDs from database...")
            scene_ids = get_all_scene_ids(self.path_config.stash_db)
            self.logger.info(f"Found {len(scene_ids)} scenes in database")

        total = len(scene_ids)
        fps_rate = self.embed_config.fps_rate

        # Print header
        mode_str = "DRY RUN" if self.dry_run else "REPAIR"
        print()
        print("=" * 70)
        print(f"Frame Validation Report ({mode_str})")
        print("=" * 70)
        print(f"Expected FPS rate: {fps_rate}")
        print()

        # Track results
        ok_count = 0
        needs_reprocessing_count = 0
        fixed_count = 0
        error_count = 0
        scenes_needing_fix: list[FrameValidationResult] = []

        # Validate each scene
        for i, scene_id in enumerate(scene_ids):
            # Check for stop signal
            if _stop_event.is_set():
                self.logger.warning("Stop signal received, aborting validation...")
                break

            result = validate_scene_frames(
                scene_id=scene_id,
                stash_db=self.path_config.stash_db,
                cache_dir=self.path_config.cache_dir,
                storage=self.storage,
                fps_rate=fps_rate,
            )

            if result is None:
                self.logger.warning(f"Scene {scene_id}: Not found in database")
                error_count += 1
                continue

            if result.needs_reprocessing:
                needs_reprocessing_count += 1
                scenes_needing_fix.append(result)

                if self.dry_run:
                    # Just report, don't fix
                    print(f"Scene {scene_id}: NEEDS REPROCESSING")
                    print(
                        f"  Duration: {result.video_duration:.1f}s, Expected: {result.expected_frames} frames"
                    )
                    print(
                        f"  Cached: {result.cached_frames} frames, DB: {result.db_embeddings} embeddings"
                    )
                    print(f"  Reason: {result.reason}")
                    print()
                else:
                    # Fix the scene
                    success = self._fix_scene(
                        result=result,
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                        embedding_device=embedding_device,
                        pretrained=pretrained,
                    )
                    if success:
                        fixed_count += 1
                    else:
                        error_count += 1
            else:
                ok_count += 1
                if self.logger.debug_enabled:
                    self.logger.debug(
                        f"Scene {scene_id}: OK ({result.cached_frames} cached, {result.db_embeddings} DB)"
                    )

            # Progress update every 100 scenes
            if (i + 1) % 100 == 0:
                self.logger.info(f"Progress: {i + 1}/{total} scenes validated")

        # Print summary
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"  Total scenes:        {total}")
        print(f"  OK:                  {ok_count}")
        print(f"  Need reprocessing:   {needs_reprocessing_count}")

        if not self.dry_run:
            print(f"  Fixed:               {fixed_count}")

        print(f"  Errors:              {error_count}")
        print("=" * 70)

        return {
            "total": total,
            "ok": ok_count,
            "needs_reprocessing": needs_reprocessing_count,
            "fixed": fixed_count,
            "errors": error_count,
            "dry_run": self.dry_run,
        }

    def _fix_scene(
        self,
        result: FrameValidationResult,
        embedding_provider: str,
        embedding_model: str,
        embedding_device: str,
        pretrained: str | None,
    ) -> bool:
        """Re-extract frames and re-embed for a scene.

        Args:
            result: Validation result for the scene
            embedding_provider: Provider for re-embedding
            embedding_model: Model for re-embedding
            embedding_device: Device for re-embedding
            pretrained: Pretrained weights

        Returns:
            True if fix was successful, False otherwise
        """
        scene_id = result.scene_id

        self.logger.info(f"Processing scene {scene_id}...")

        # Step 1: Clear cached frames
        if result.cached_frames > 0:
            deleted = clear_cached_frames(self.path_config.cache_dir, scene_id)
            self.logger.info(f"  Cleared {deleted} cached frames")

        # Step 2: Delete existing DB embeddings
        if result.db_embeddings > 0:
            deleted = self.storage.delete_frame_embeddings(scene_id)
            self.logger.info(f"  Deleted {deleted} DB embeddings")

        # Step 3: Re-extract and re-embed
        try:
            embed_task = self._get_embed_task(
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
                pretrained=pretrained,
            )

            embed_result = embed_task.embed_scene(scene_id, force=True)

            if embed_result.get("success"):
                self.logger.info(
                    f"  Re-embedded successfully ({embed_result.get('dimensions', 0)} dims)"
                )
                return True
            else:
                self.logger.error(
                    f"  Re-embedding failed: {embed_result.get('error', 'Unknown error')}"
                )
                return False

        except Exception as e:
            self.logger.error(f"  Error fixing scene {scene_id}: {type(e).__name__}: {e}")
            return False


# =============================================================================
# Standalone Embedding Task
# =============================================================================


class StandaloneEmbedTask:
    """
    Standalone task for generating scene embeddings.

    This is a simplified version of EmbedScenesTask that doesn't require
    a Stash GraphQL connection and supports configurable paths.
    """

    def __init__(
        self,
        path_config: PathConfig,
        embed_config: EmbedConfig,
        embedding_provider: str,
        embedding_model: str,
        embedding_device: str = "cuda",
        pretrained: str | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.path_config = path_config
        self.config = embed_config
        self.log = log_callback or (lambda msg, level: print(f"[{level}] {msg}"))
        self.progress = progress_callback or (lambda cur, total: None)

        # Build model key
        if embedding_provider == "siglip":
            model_key = "siglip"
        else:
            model_key = f"{embedding_provider}:{embedding_model}"

        self.log(f"Using embedding model: {model_key}", "info")

        # Initialize storage
        self.storage = StandaloneEmbeddingStorage(
            db_path=str(path_config.output_db),
            model_key=model_key,
        )

        # Initialize frame extractor
        self.frame_extractor = FrameExtractor(
            config=FrameExtractionConfig(
                fps_rate=embed_config.fps_rate,
                min_frames=embed_config.min_frames,
                max_frames=embed_config.max_frames,
                frame_width=embed_config.frame_width,
            ),
            cache_dir=str(path_config.cache_dir),
            log_callback=self.log,
        )

        # Lazy-load embedder
        self._embedder = None
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._embedding_device = embedding_device
        self._pretrained = pretrained

        # GPU lock for thread safety
        self._gpu_lock = Lock()

        # Benchmark instance for timing operations
        self._benchmark = Benchmark()

    @property
    def embedder(self):
        """Lazy-load image embedding provider."""
        if self._embedder is None:
            self._embedder = self._create_embedder()
        return self._embedder

    def _create_embedder(self):
        """Create the embedding provider."""
        provider = self._embedding_provider.lower()
        batch_size = self.config.batch_size

        self.log(f"Creating {provider} embedder with batch_size={batch_size}", "info")

        if provider == "openclip":
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.providers.openclip import OpenCLIPEmbeddingProvider

            config = EmbeddingConfig(
                provider="openclip",
                model=self._embedding_model,
                device=self._embedding_device,
                pretrained=self._pretrained,
                batch_size=batch_size,
            )
            return OpenCLIPEmbeddingProvider(config)

        elif provider == "siglip":
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.providers.siglip import SigLIPEmbeddingProvider

            config = EmbeddingConfig(
                provider="siglip",
                model=self._embedding_model,
                device=self._embedding_device,
                batch_size=batch_size,
            )
            return SigLIPEmbeddingProvider(config)

        elif provider == "clip":
            from stash_ai.embeddings.config import EmbeddingConfig
            from stash_ai.embeddings.providers.clip import CLIPEmbeddingProvider

            config = EmbeddingConfig(
                provider="clip",
                model=self._embedding_model,
                device=self._embedding_device,
                batch_size=batch_size,
            )
            return CLIPEmbeddingProvider(config)

        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

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
        self.cleanup()

    def embed_scene(self, scene_id: int, force: bool = False) -> dict[str, Any]:
        """Generate and store embedding for a single scene."""
        # Check for stop signal
        if _stop_event.is_set():
            return {"success": False, "error": "Interrupted", "scene_id": scene_id}

        # Check if already embedded
        if not force and self.storage.has_embedding(scene_id):
            self.log(f"Scene {scene_id}: Already has embedding, skipping", "debug")
            return {"success": True, "scene_id": scene_id, "cached": True}

        # Get scene info
        self.log(f"Scene {scene_id}: Fetching scene info from database", "debug")
        scene_info = get_scene_info(self.path_config.stash_db, scene_id)
        if not scene_info:
            self.log(f"Scene {scene_id}: Not found in database", "warning")
            return {"success": False, "error": f"Scene {scene_id} not found in database"}

        # Remap video path if needed
        original_path = scene_info.get("video_path", "")
        video_path = self.path_config.remap_path(original_path)
        scene_info["video_path"] = video_path

        self.log(f"Scene {scene_id}: Video path: {video_path}", "debug")

        if not Path(video_path).exists():
            self.log(f"Scene {scene_id}: Video file not found", "warning")
            self.log(f"  Original path: {original_path}", "debug")
            self.log(f"  Remapped path: {video_path}", "debug")
            return {
                "success": False,
                "error": f"Video not found: {video_path}",
                "original_path": original_path,
            }

        # Check for stop signal before loading frames
        if _stop_event.is_set():
            return {"success": False, "error": "Interrupted", "scene_id": scene_id}

        # Load cached frames (this script does not extract frames)
        self.log(f"Scene {scene_id}: Loading cached frames", "debug")
        try:
            with self._benchmark.time("frame_loading"):
                frame_paths = self._prepare_frames(scene_id, scene_info)
        except FileNotFoundError as e:
            self.log(f"Scene {scene_id}: {e}", "warning")
            return {
                "success": False,
                "error": str(e),
            }

        self.log(f"Scene {scene_id}: Loaded {len(frame_paths)} cached frames", "debug")

        # Build metadata text
        metadata_text = self._build_metadata_text(scene_info)

        # Check for stop signal before GPU work
        if _stop_event.is_set():
            return {"success": False, "error": "Interrupted", "scene_id": scene_id}

        # Embed with GPU lock
        with self._gpu_lock:
            return self._embed_scene_gpu_locked(
                scene_id=scene_id,
                scene_info=scene_info,
                frame_paths=frame_paths,
                metadata_text=metadata_text,
            )

    def _prepare_frames(
        self,
        scene_id: int,
        scene_info: dict[str, Any],
    ) -> list[str]:
        """Get frames for embedding, optionally extracting them.

        If config.extract_frames is True and no cached frames exist, extracts
        frames at the configured fps_rate. Existing cached frames are always
        preserved and reused.

        Raises:
            FileNotFoundError: If no cached frames exist and extract_frames is False
        """
        video_path = scene_info.get("video_path", "")
        duration = scene_info.get("duration", 0.0)

        # Check for existing cached frames first
        if self.frame_extractor.has_cached_frames(str(scene_id)):
            frames = self.frame_extractor.get_frame_paths(str(scene_id))
            if frames:
                self.log(f"Scene {scene_id}: Using {len(frames)} cached frames", "debug")
                return frames

        # No cached frames - extract if enabled
        if self.config.extract_frames:
            frames = self.frame_extractor.extract_frames_at_fps(
                scene_id=str(scene_id),
                video_path=video_path,
                duration=duration,
                fps_rate=self.config.fps_rate,
            )

            if not frames:
                raise RuntimeError(
                    f"Failed to extract frames for scene {scene_id} from: {video_path}"
                )

            self.log(
                f"Scene {scene_id}: Extracted {len(frames)} frames at {self.config.fps_rate} fps",
                "debug",
            )
            return frames

        # No cached frames and extraction not enabled
        cache_dir = self.frame_extractor.get_scene_cache_dir(str(scene_id))
        raise FileNotFoundError(
            f"No cached frames for scene {scene_id}. "
            f"Expected frames in: {cache_dir}\n"
            f"Use --extract-frames to extract frames during embedding."
        )

    def _embed_scene_gpu_locked(
        self,
        scene_id: int,
        scene_info: dict[str, Any],
        frame_paths: list[str] | None,
        metadata_text: str | None,
    ) -> dict[str, Any]:
        """Perform GPU embedding operations (must hold GPU lock)."""
        visual_embedding: list[float] | None = None
        visual_model: str | None = None

        # Stage 1: Visual embedding
        if frame_paths and self.config.use_direct_image_embedding:
            self.log(
                f"Scene {scene_id}: Embedding {len(frame_paths)} frames with {self._embedding_model}",
                "info",
            )

            try:
                with self._benchmark.time("gpu_embedding"):
                    results = self.embedder.embed_images(frame_paths)

                if results:
                    embeddings = [r["embedding"] for r in results]

                    # Store individual frame embeddings
                    with self._benchmark.time("db_store_frames"):
                        self._store_frame_embeddings(
                            scene_id=scene_id,
                            frame_paths=frame_paths,
                            embeddings=embeddings,
                            duration=scene_info.get("duration", 0.0),
                        )

                    visual_embedding = self._average_embeddings(embeddings)
                    visual_model = self._embedding_model

            except Exception as e:
                self.log(f"Failed to embed frames: {e}", "warning")

        # Stage 2: Metadata embedding
        metadata_embedding: list[float] | None = None

        if metadata_text and self.embedder is not None:
            try:
                metadata_result = self.embedder.embed_text(metadata_text)
                metadata_embedding = metadata_result["embedding"]
            except Exception as e:
                self.log(f"Failed to embed metadata: {e}", "warning")

        # Stage 3: Combine embeddings
        composite = self._combine_embeddings(
            visual_embedding,
            metadata_embedding,
            self.config.visual_weight,
        )

        if composite is None:
            return {
                "success": False,
                "error": "No content to embed (no visual or metadata)",
            }

        # Stage 4: Store
        with self._benchmark.time("db_store_embedding"):
            self.storage.store_embedding(
                scene_id=scene_id,
                composite_embedding=composite,
                text_model=self._embedding_model,
                visual_embedding=visual_embedding,
                metadata_embedding=metadata_embedding,
                visual_model=visual_model,
                visual_description=None,
                metadata_text=metadata_text,
            )

        self.log(f"Stored embedding for scene {scene_id} ({len(composite)} dims)", "info")

        return {
            "success": True,
            "scene_id": scene_id,
            "dimensions": len(composite),
            "has_visual": visual_embedding is not None,
            "has_metadata": metadata_embedding is not None,
        }

    def _store_frame_embeddings(
        self,
        scene_id: int,
        frame_paths: list[str],
        embeddings: list[list[float]],
        duration: float,
    ) -> None:
        """Store all frame embeddings in a single batch transaction."""
        if not frame_paths or not embeddings:
            return

        if len(frame_paths) != len(embeddings):
            self.log(
                f"Frame/embedding count mismatch: {len(frame_paths)} vs {len(embeddings)}",
                "warning",
            )
            return

        frame_pattern = re.compile(r"frame_(\d+)\.(?:jpg|png)$", re.IGNORECASE)

        # Build list of (frame_index, timestamp, embedding) tuples
        frames: list[tuple[int, float, list[float]]] = []
        frame_index = 0

        for frame_path, embedding in zip(frame_paths, embeddings):
            match = frame_pattern.search(frame_path)
            if not match:
                continue

            frame_num = int(match.group(1))
            timestamp = float(frame_num - 1)
            frames.append((frame_index, timestamp, embedding))
            frame_index += 1

        if frames:
            # Calculate composite embedding
            composite = self._average_embeddings(embeddings)

            # Store all in one transaction
            stored_count = self.storage.store_frame_embeddings_batch(
                scene_id=scene_id,
                frames=frames,
                duration=duration,
                composite_embedding=composite,
            )

            self.log(
                f"Stored {stored_count} frame embeddings for scene {scene_id}",
                "debug",
            )

    def _build_metadata_text(self, scene_info: dict[str, Any]) -> str | None:
        """Build concatenated metadata text for embedding."""
        parts: list[str] = []

        if self.config.include_title and scene_info.get("title"):
            parts.append(f"Title: {scene_info['title']}")

        if self.config.include_performers and scene_info.get("performers"):
            performers = ", ".join(scene_info["performers"])
            parts.append(f"Performers: {performers}")

        if self.config.include_studio and scene_info.get("studio"):
            parts.append(f"Studio: {scene_info['studio']}")

        if self.config.include_tags and scene_info.get("tags"):
            tags = ", ".join(scene_info["tags"])
            parts.append(f"Tags: {tags}")

        if not parts:
            return None

        return ". ".join(parts)

    def _average_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> list[float] | None:
        """Average multiple embeddings into a single embedding."""
        if not embeddings:
            return None

        if len(embeddings) == 1:
            return embeddings[0]

        arr: NDArray[np.float32] = np.array(embeddings, dtype=np.float32)
        averaged = np.mean(arr, axis=0)

        # Re-normalize
        norm: float = float(np.linalg.norm(averaged))
        if norm > 0:
            averaged = averaged / norm

        return averaged.tolist()

    def _combine_embeddings(
        self,
        visual: list[float] | None,
        metadata: list[float] | None,
        visual_weight: float,
    ) -> list[float] | None:
        """Combine visual and metadata embeddings with weighting."""
        if visual is None and metadata is None:
            return None

        if visual is None:
            return metadata

        if metadata is None:
            return visual

        # Both available - weighted average
        if len(visual) != len(metadata):
            self.log(
                f"Embedding dimension mismatch: visual={len(visual)}, "
                f"metadata={len(metadata)}. Using visual only.",
                "warning",
            )
            return visual

        v_arr: NDArray[np.float32] = np.array(visual, dtype=np.float32)
        m_arr: NDArray[np.float32] = np.array(metadata, dtype=np.float32)

        combined = visual_weight * v_arr + (1 - visual_weight) * m_arr

        # Re-normalize
        norm: float = float(np.linalg.norm(combined))
        if norm > 0:
            combined = combined / norm

        return combined.tolist()

    def embed_all(
        self,
        scene_ids: list[int] | None = None,
        force: bool = False,
        batch_size: int = 10,
    ) -> dict[str, Any]:
        """
        Generate embeddings for all (or specified) scenes.

        Args:
            scene_ids: Specific scene IDs to embed (None = all scenes)
            force: If True, regenerate all embeddings
            batch_size: Number of scenes per progress update

        Returns:
            Summary of embedding generation
        """
        try:
            return self._embed_all_impl(scene_ids, force, batch_size)
        finally:
            self.cleanup()

    def _embed_all_impl(
        self,
        scene_ids: list[int] | None,
        force: bool,
        batch_size: int,
    ) -> dict[str, Any]:
        """Internal implementation of embed_all."""
        # Get scene IDs
        if scene_ids is None:
            self.log("Fetching scene IDs from database...", "info")
            scene_ids = get_all_scene_ids(self.path_config.stash_db)
            self.log(f"Found {len(scene_ids)} scenes in database", "info")

        total = len(scene_ids)

        # Determine number of workers
        num_workers = self.config.num_workers
        if num_workers <= 0:
            import multiprocessing

            num_workers = min(multiprocessing.cpu_count(), 8)

        self.log(f"Starting embedding of {total} scenes using {num_workers} workers...", "info")

        # Thread-safe counters
        counter_lock = Lock()
        counters = {
            "success": 0,
            "skip": 0,
            "error": 0,
            "processed": 0,
        }
        errors: list[str] = []
        interrupted = False

        def process_scene(scene_id: int) -> str | None:
            """Process a single scene (thread-safe). Returns error string or None."""
            nonlocal counters, errors, interrupted

            # Check for stop signal before starting
            if _stop_event.is_set():
                return None

            try:
                result = self.embed_scene(scene_id, force=force)

                # Check again after potentially long operation
                if _stop_event.is_set():
                    return None

                with counter_lock:
                    counters["processed"] += 1
                    if result.get("success"):
                        if result.get("cached"):
                            counters["skip"] += 1
                        else:
                            counters["success"] += 1
                    else:
                        counters["error"] += 1
                        err_msg = f"Scene {scene_id}: {result.get('error')}"
                        if len(errors) < 50:
                            errors.append(err_msg)
                        return err_msg

                    # Update progress
                    if counters["processed"] % batch_size == 0:
                        self.progress(counters["processed"], total)
                        self.log(
                            f"Progress: {counters['processed']}/{total} "
                            f"(success={counters['success']}, skip={counters['skip']}, err={counters['error']})",
                            "info",
                        )

            except Exception as e:
                with counter_lock:
                    counters["processed"] += 1
                    counters["error"] += 1
                    err_msg = f"Scene {scene_id}: {type(e).__name__}: {e!s}"
                    if len(errors) < 50:
                        errors.append(err_msg)
                self.log(f"Error embedding scene {scene_id}: {type(e).__name__}: {e}", "error")
                return err_msg

            return None

        # Process scenes in parallel with graceful shutdown support
        executor = ThreadPoolExecutor(max_workers=num_workers)
        try:
            futures = {executor.submit(process_scene, scene_id): scene_id for scene_id in scene_ids}

            for _future in as_completed(futures):
                # Check for interrupt after each future completes
                if _stop_event.is_set():
                    interrupted = True
                    self.log(
                        "Stop signal received, waiting for current tasks to complete...", "warning"
                    )
                    break

        except KeyboardInterrupt:
            interrupted = True
            self.log("Keyboard interrupt received, shutting down...", "warning")
        finally:
            if interrupted or _stop_event.is_set():
                self.log("Cancelling pending tasks...", "info")
                # Cancel pending futures
                for future in futures:
                    future.cancel()

            # Shutdown executor - wait for running tasks but don't start new ones
            executor.shutdown(wait=True, cancel_futures=True)

            if interrupted:
                self.log(
                    f"Interrupted after processing {counters['processed']}/{total} scenes",
                    "warning",
                )

        self.progress(counters["processed"], total)

        status_msg = "interrupted" if interrupted else "complete"
        self.log(
            f"Embedding {status_msg}: {counters['success']} success, "
            f"{counters['skip']} skipped, {counters['error']} errors",
            "info",
        )

        # Print benchmark summary
        print(self._benchmark.summary())

        return {
            "total": total,
            "success": counters["success"],
            "skipped": counters["skip"],
            "errors": counters["error"],
            "error_details": errors[:10],
            "interrupted": interrupted,
        }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone scene embedding script for remote GPU processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Path configuration
    parser.add_argument(
        "--stash-db",
        type=Path,
        default=Path.home() / ".stash" / "stash-go.sqlite",
        help="Path to Stash database (default: ~/.stash/stash-go.sqlite)",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=None,
        help="Path to output embedding database (default: ./assets/stash_copilot.sqlite)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Path to frame cache directory (default: ./assets/embedded_frames)",
    )
    parser.add_argument(
        "--path-remap",
        action="append",
        default=[],
        metavar="SERVER=LOCAL",
        help="Remap server paths to local mounts (can be used multiple times)",
    )

    # Embedding model
    parser.add_argument(
        "--provider",
        choices=["openclip", "siglip", "clip"],
        default="openclip",
        help="Embedding provider (default: openclip)",
    )
    parser.add_argument(
        "--model",
        default="ViT-H-14",
        help="Model name (default: ViT-H-14)",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Pretrained weights (default: auto-selected based on model)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for embedding (default: cuda)",
    )

    # Embedding settings
    parser.add_argument(
        "--visual-weight",
        type=float,
        default=1.0,
        help="Weight for visual embedding (0-1, default: 1.0 = visual only)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=0,
        help="Minimum frames to extract (default: 0, no minimum)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to extract (default: 0, no limit)",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Frame width for extraction (default: 640)",
    )

    # Processing
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="GPU batch size for embedding (default: 8, increase to use more VRAM)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-embed all scenes (ignore existing embeddings)",
    )
    parser.add_argument(
        "--scene-ids",
        type=int,
        nargs="+",
        help="Specific scene IDs to embed (default: all scenes)",
    )

    # Frame extraction
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract and save frames during embedding (default: use existing cached frames)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second for extraction (default: 1.0)",
    )
    parser.add_argument(
        "--skip-rate-check",
        action="store_true",
        help="Skip fps rate consistency check with existing frames",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (for non-interactive use)",
    )

    # Frame validation
    parser.add_argument(
        "--validate-frames",
        action="store_true",
        help="Validate frame counts and re-extract/re-embed scenes with incorrect counts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which scenes need reprocessing (no changes made)",
    )

    # Output
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    return parser.parse_args()


def setup_signal_handlers(logger: Logger) -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(signum: int, frame: Any) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            sig_name = str(signum)

        if _stop_event.is_set():
            # Second interrupt - force exit
            logger.error(f"Received second {sig_name}, forcing exit...")
            sys.exit(130)
        else:
            logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
            logger.info("Press Ctrl+C again to force exit immediately")
            _stop_event.set()

    # Register SIGINT handler (Ctrl+C) - works on all platforms
    signal.signal(signal.SIGINT, signal_handler)

    # Register SIGTERM handler (kill command) - Unix only
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    # On Windows, also handle SIGBREAK (Ctrl+Break)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Create logger early so we can use it for all output
    logger = Logger(quiet=args.quiet, debug=args.debug)

    # Set up signal handlers
    setup_signal_handlers(logger)

    # Validate stash database exists
    if not args.stash_db.exists():
        logger.error(f"Stash database not found: {args.stash_db}")
        logger.info("Use --stash-db to specify the correct path")
        return 1

    # Set default paths relative to script location
    script_dir = Path(__file__).parent
    if args.output_db is None:
        args.output_db = script_dir / "assets" / "stash_copilot.sqlite"
    if args.cache_dir is None:
        args.cache_dir = script_dir / "assets" / "embedded_frames"

    # Create output directories
    try:
        args.output_db.parent.mkdir(parents=True, exist_ok=True)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directories: {e}", exc=e)
        return 1

    # Parse path remaps
    path_remaps: dict[str, str] = {}
    for remap in args.path_remap:
        if "=" not in remap:
            logger.error(f"Invalid path remap format: {remap}")
            logger.info("Use format: --path-remap '/server/path=/local/path'")
            return 1
        server_path, local_path = remap.split("=", 1)
        path_remaps[server_path] = local_path
        logger.debug(f"Path remap: {server_path} -> {local_path}")

    # Create path config
    path_config = PathConfig(
        stash_db=args.stash_db,
        output_db=args.output_db,
        cache_dir=args.cache_dir,
        path_remaps=path_remaps,
    )

    # Create embed config
    embed_config = EmbedConfig(
        visual_weight=args.visual_weight,
        fps_rate=args.fps,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        frame_width=args.frame_width,
        num_workers=args.workers,
        batch_size=args.batch_size,
        extract_frames=args.extract_frames,
    )

    # Print configuration
    print("=" * 60)
    print("Standalone Scene Embedding")
    print("=" * 60)
    print(f"Stash DB:    {path_config.stash_db}")
    print(f"Output DB:   {path_config.output_db}")
    print(f"Cache Dir:   {path_config.cache_dir}")
    print(f"Provider:    {args.provider}")
    print(f"Model:       {args.model}")
    if args.pretrained:
        print(f"Pretrained:  {args.pretrained}")
    print(f"Device:      {args.device}")
    print(f"Workers:     {args.workers}")
    print(f"Batch Size:  {args.batch_size}")
    print(f"Visual Wgt:  {args.visual_weight}")
    if args.extract_frames:
        print(f"Extract:     Yes ({args.fps} fps)")
    else:
        print("Extract:     No (use cached frames)")
    if path_remaps:
        print("Path Remaps:")
        for server, local in path_remaps.items():
            print(f"  {server} -> {local}")
    print("=" * 60)
    print()

    # Add stash_ai to path
    sys.path.insert(0, str(script_dir))

    # Validate database access before starting
    logger.info("Validating database access...")
    try:
        conn = get_readonly_connection(path_config.stash_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scenes")
        scene_count = cursor.fetchone()[0]
        conn.close()
        logger.info(f"Stash database accessible: {scene_count} scenes found")
    except sqlite3.OperationalError as e:
        logger.error(f"Cannot access Stash database: {e}")
        logger.info("Check that the database path is correct and accessible")
        logger.info(f"  Path: {path_config.stash_db}")
        return 1
    except Exception as e:
        logger.error(f"Database validation failed: {type(e).__name__}: {e}", exc=e)
        return 1

    # Test a path remap if any are configured
    if path_remaps:
        logger.info("Testing path remaps...")
        try:
            conn = get_readonly_connection(path_config.stash_db)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT fo.path || '/' || f.basename as video_path
                FROM scenes s
                JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
                JOIN files f ON sf.file_id = f.id
                JOIN folders fo ON f.parent_folder_id = fo.id
                LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()

            if row:
                original_path = row[0]
                remapped_path = path_config.remap_path(original_path)
                logger.debug("Sample path remap:")
                logger.debug(f"  Original: {original_path}")
                logger.debug(f"  Remapped: {remapped_path}")

                if Path(remapped_path).exists():
                    logger.info("Path remap test: OK (file accessible)")
                else:
                    logger.warning("Path remap test: File not found at remapped path")
                    logger.warning("  Check your --path-remap settings")
        except Exception as e:
            logger.warning(f"Path remap test failed: {e}")

    # Handle frame validation mode
    if args.validate_frames:
        logger.info(f"Frame validation mode (dry_run={args.dry_run})")

        # Build model key for storage
        if args.provider == "siglip":
            model_key = "siglip"
        else:
            model_key = f"{args.provider}:{args.model}"

        # Initialize storage and frame extractor
        storage = StandaloneEmbeddingStorage(
            db_path=str(path_config.output_db),
            model_key=model_key,
        )

        frame_extractor = FrameExtractor(
            config=FrameExtractionConfig(
                fps_rate=embed_config.fps_rate,
                min_frames=embed_config.min_frames,
                max_frames=embed_config.max_frames,
                frame_width=embed_config.frame_width,
            ),
            cache_dir=str(path_config.cache_dir),
            log_callback=lambda msg, level: logger.log(msg, level),
        )

        # Create and run validation task
        validation_task = FrameValidationTask(
            path_config=path_config,
            embed_config=embed_config,
            storage=storage,
            frame_extractor=frame_extractor,
            logger=logger,
            dry_run=args.dry_run,
        )

        try:
            result = validation_task.validate_all(
                scene_ids=args.scene_ids,
                embedding_provider=args.provider,
                embedding_model=args.model,
                embedding_device=args.device,
                pretrained=args.pretrained,
            )

            # Return code based on results
            if result["errors"] > 0:
                return 1
            return 0

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Validation failed: {type(e).__name__}: {e}", exc=e)
            return 1

    # Frame extraction confirmation flow
    if args.extract_frames:
        logger.info(f"Frame extraction enabled ({args.fps} fps)")

        # Scan existing frame rates
        existing_rates = scan_existing_frame_rates(path_config.cache_dir)
        if existing_rates:
            logger.info("Existing frame rates in cache:")
            for rate, count in sorted(existing_rates.items()):
                logger.info(f"  {rate} fps: {count} scenes")

            # Check for rate mismatch
            if not args.skip_rate_check:
                existing_fps_set = set(existing_rates.keys())
                if existing_fps_set and args.fps not in existing_fps_set:
                    logger.warning(
                        f"FPS rate {args.fps} differs from existing: {sorted(existing_fps_set)}"
                    )
                    logger.warning("Mixed rates may affect embedding consistency.")

                    if not prompt_user_confirmation(
                        "Continue with different rate?",
                        default=False,
                        auto_yes=args.yes,
                    ):
                        logger.info("Aborted by user.")
                        return 0

        # Estimate total duration and frames
        total_duration, scenes_to_process = estimate_total_duration(
            path_config.stash_db,
            args.scene_ids,
        )

        estimated_frames = int(total_duration * args.fps)
        # Estimate storage: ~50KB per frame (JPEG quality 2)
        estimated_storage = estimated_frames * 50 * 1024

        print()
        print("Frame Extraction Summary:")
        print(f"  Scenes to process: {scenes_to_process}")
        print(f"  Estimated duration: {format_duration_human(total_duration)}")
        print(f"  Estimated frames: {estimated_frames:,}")
        print(f"  Storage estimate: ~{format_storage_size(estimated_storage)}")
        print()

        if not prompt_user_confirmation(
            "Proceed with frame extraction?",
            default=True,
            auto_yes=args.yes,
        ):
            logger.info("Aborted by user.")
            return 0

        print()

    # Create log callback that uses our logger
    def log_callback(msg: str, level: str) -> None:
        logger.log(msg, level)

    def progress_callback(current: int, total: int) -> None:
        logger.progress(current, total)

    # Create and run task
    task: StandaloneEmbedTask | None = None
    try:
        logger.info("Initializing embedding task...")
        task = StandaloneEmbedTask(
            path_config=path_config,
            embed_config=embed_config,
            embedding_provider=args.provider,
            embedding_model=args.model,
            embedding_device=args.device,
            pretrained=args.pretrained,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        logger.info("Starting embedding process...")
        result = task.embed_all(
            scene_ids=args.scene_ids,
            force=args.force,
        )

        print()
        print("=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Total:    {result['total']}")
        print(f"Success:  {result['success']}")
        print(f"Skipped:  {result['skipped']}")
        print(f"Errors:   {result['errors']}")

        if result.get("interrupted"):
            print("\nNote: Process was interrupted before completion")

        if result.get("error_details"):
            print("\nSample Errors:")
            for err in result["error_details"][:5]:
                print(f"  - {err}")

        # Return code: 0 for success, 130 for interrupt, 1 for errors
        if result.get("interrupted"):
            return 130
        return 0 if result["errors"] == 0 else 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {type(e).__name__}: {e}", exc=e)
        return 1
    finally:
        # Ensure cleanup happens
        if task is not None:
            try:
                task.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
