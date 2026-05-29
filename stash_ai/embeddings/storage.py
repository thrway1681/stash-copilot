"""SQLite storage for scene embeddings."""

import sqlite3
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray


class SceneEmbeddingRecord(TypedDict):
    """Represents a stored scene embedding."""

    scene_id: int
    model_key: str
    visual_embedding: list[float] | None
    metadata_embedding: list[float] | None
    composite_embedding: list[float]
    visual_model: str | None
    text_model: str
    dimensions: int
    visual_description: str | None
    metadata_text: str | None
    created_at: str
    updated_at: str


@dataclass
class SimilarityResult:
    """Result from a similarity search."""

    scene_id: int
    similarity: float
    visual_description: str | None = None


# Default model key for legacy embeddings (SigLIP)
DEFAULT_MODEL_KEY = "siglip"


class OMomentEmbeddingRecord(TypedDict):
    """Represents a stored O-moment embedding.

    Note: marker_id references scene_markers.id in the Stash database.
    """

    scene_id: int
    o_event_index: int
    marker_id: int  # FK to scene_markers.id in Stash DB
    center_timestamp: float
    window_seconds: float
    embedding: list[float]
    frame_count: int
    model_key: str
    created_at: str


class FrameEmbeddingRecord(TypedDict):
    """Represents a single frame embedding."""

    scene_id: int
    frame_index: int
    timestamp: float
    embedding: list[float]
    model_key: str
    created_at: str


class FrameEmbeddingMetadata(TypedDict):
    """Represents scene-level frame embedding metadata."""

    scene_id: int
    model_key: str
    frame_count: int
    total_frames_extracted: int
    duration: float
    sampling_rate: float
    composite_embedding: list[float]
    dedup_ratio: float
    first_frame_timestamp: float | None
    last_frame_timestamp: float | None
    created_at: str
    updated_at: str


class SceneSegment(TypedDict):
    """Represents a scene segment boundary."""

    scene_id: int
    model_key: str
    segment_index: int
    start_timestamp: float
    end_timestamp: float
    start_frame: int
    end_frame: int
    avg_embedding: list[float]
    boundary_score: float
    segment_type: str | None
    created_at: str


class FrameTagCoverageRecord(TypedDict):
    """Represents a frame's tag coverage analysis result."""

    scene_id: int
    frame_index: int
    model_key: str
    best_tag: str
    best_similarity: float
    is_covered: bool


class EmbeddingStorage:
    """
    SQLite-based storage for scene embeddings.

    Uses a separate database file to avoid modifying Stash's database.
    Embeddings are stored as BLOB (packed float32 array) for efficiency.

    Supports multiple embedding models per scene via model_key namespacing.
    """

    # Database schema version for migrations
    SCHEMA_VERSION = 13  # Added dismissed_tag_merges table

    def __init__(
        self,
        db_path: str | None = None,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> None:
        """
        Initialize embedding storage.

        Args:
            db_path: Path to SQLite database. Defaults to plugin assets directory.
            model_key: Model identifier for this storage instance (e.g., "siglip",
                "openclip:ViT-H-14"). All operations will be scoped to this model.
        """
        if db_path is None:
            # Default to plugin assets directory
            plugin_dir = Path(__file__).parent.parent.parent
            assets_dir = plugin_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            db_path = str(assets_dir / "stash_copilot.sqlite")

        self.db_path = db_path
        self.model_key = model_key
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection.

        Uses WAL mode for better concurrent write performance.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency (multiple readers, single writer)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_database(self) -> None:
        """Initialize database schema and run migrations."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check current schema version
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        cursor.execute("SELECT value FROM schema_info WHERE key = 'version'")
        version_row = cursor.fetchone()
        current_version = int(version_row["value"]) if version_row else 0

        # Run migrations
        if current_version < 1:
            self._migrate_to_v1(cursor)
        if current_version < 2:
            self._migrate_to_v2(cursor)
        if current_version < 3:
            self._migrate_to_v3(cursor)
        if current_version < 4:
            self._migrate_to_v4(cursor)
        if current_version < 5:
            self._migrate_to_v5(cursor)
        if current_version < 6:
            self._migrate_to_v6(cursor)
        if current_version < 7:
            self._migrate_to_v7(cursor)
        if current_version < 8:
            self._migrate_to_v8(cursor)
        if current_version < 9:
            self._migrate_to_v9(cursor)
        if current_version < 10:
            self._migrate_to_v10(cursor)
        if current_version < 11:
            self._migrate_to_v11(cursor)
        if current_version < 12:
            self._migrate_to_v12(cursor)
        if current_version < 13:
            self._migrate_to_v13(cursor)

        # Update schema version
        cursor.execute(
            """
            INSERT OR REPLACE INTO schema_info (key, value)
            VALUES ('version', ?)
        """,
            (str(self.SCHEMA_VERSION),),
        )

        conn.commit()
        conn.close()

    def _migrate_to_v1(self, cursor: sqlite3.Cursor) -> None:
        """Create initial schema (v1)."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scene_embeddings (
                scene_id INTEGER PRIMARY KEY,
                visual_embedding BLOB,
                metadata_embedding BLOB,
                composite_embedding BLOB NOT NULL,
                visual_model TEXT,
                text_model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                visual_description TEXT,
                metadata_text TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scene_embeddings_scene_id
            ON scene_embeddings(scene_id)
        """
        )

    def _migrate_to_v2(self, cursor: sqlite3.Cursor) -> None:
        """Add model_key column for multi-model support (v2).

        Migrates existing embeddings to have model_key='siglip' (legacy default).
        Creates new table with composite primary key (scene_id, model_key).
        """
        # Check if model_key column already exists
        cursor.execute("PRAGMA table_info(scene_embeddings)")
        columns = [row["name"] for row in cursor.fetchall()]

        if "model_key" in columns:
            # Already migrated
            return

        # Create new table with model_key
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scene_embeddings_v2 (
                scene_id INTEGER NOT NULL,
                model_key TEXT NOT NULL DEFAULT 'siglip',
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

        # Copy existing data with default model_key='siglip'
        cursor.execute(
            """
            INSERT INTO scene_embeddings_v2 (
                scene_id, model_key, visual_embedding, metadata_embedding,
                composite_embedding, visual_model, text_model, dimensions,
                visual_description, metadata_text, created_at, updated_at
            )
            SELECT
                scene_id, 'siglip', visual_embedding, metadata_embedding,
                composite_embedding, visual_model, text_model, dimensions,
                visual_description, metadata_text, created_at, updated_at
            FROM scene_embeddings
        """
        )

        # Drop old table and rename new one
        cursor.execute("DROP TABLE scene_embeddings")
        cursor.execute("ALTER TABLE scene_embeddings_v2 RENAME TO scene_embeddings")

        # Create indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scene_embeddings_scene_id
            ON scene_embeddings(scene_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scene_embeddings_model_key
            ON scene_embeddings(model_key)
        """
        )

    def _migrate_to_v3(self, cursor: sqlite3.Cursor) -> None:
        """Add o_moment_embeddings table for O-moment embeddings (v3).

        Stores embeddings derived from frames around O markers.
        Primary key is (scene_id, o_event_index, model_key) to support
        multiple O-moments per scene and multiple embedding models.
        """
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS o_moment_embeddings (
                scene_id INTEGER NOT NULL,
                o_event_index INTEGER NOT NULL,
                marker_id INTEGER NOT NULL,
                center_timestamp REAL NOT NULL,
                window_seconds REAL NOT NULL,
                embedding BLOB NOT NULL,
                frame_count INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (scene_id, o_event_index, model_key)
            )
        """
        )

        # Create indexes for efficient queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_o_moment_scene_id
            ON o_moment_embeddings(scene_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_o_moment_model_key
            ON o_moment_embeddings(model_key)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_o_moment_marker_id
            ON o_moment_embeddings(marker_id)
        """
        )

    def _migrate_to_v4(self, cursor: sqlite3.Cursor) -> None:
        """Add dense frame-level embedding tables (v4).

        Adds three new tables for dense (1fps) frame-level embeddings:
        - frame_embeddings: Individual frame embeddings with timestamps
        - frame_embedding_metadata: Scene-level metadata and composite embeddings
        - scene_segments: Pre-computed scene boundaries (optional)

        This migration is non-destructive and does not modify scene_embeddings.
        """
        # Table: frame_embeddings
        # Stores individual frame embeddings at 1fps sampling rate
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_embeddings (
                scene_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                embedding BLOB NOT NULL,
                model_key TEXT NOT NULL DEFAULT 'siglip',
                created_at TEXT NOT NULL,
                PRIMARY KEY (scene_id, frame_index, model_key)
            )
        """
        )

        # Index for temporal queries (critical for performance)
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_frame_scene_timestamp
            ON frame_embeddings(scene_id, model_key, timestamp)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_frame_model_key
            ON frame_embeddings(model_key)
        """
        )

        # Table: frame_embedding_metadata
        # Stores scene-level composite embeddings and metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_embedding_metadata (
                scene_id INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                frame_count INTEGER NOT NULL,
                total_frames_extracted INTEGER NOT NULL,
                duration REAL NOT NULL,
                sampling_rate REAL NOT NULL,
                composite_embedding BLOB NOT NULL,
                dedup_ratio REAL NOT NULL,
                first_frame_timestamp REAL,
                last_frame_timestamp REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (scene_id, model_key)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_frame_metadata_model_key
            ON frame_embedding_metadata(model_key)
        """
        )

        # Table: scene_segments
        # Stores pre-computed scene boundaries (computed on-demand)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scene_segments (
                scene_id INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                start_timestamp REAL NOT NULL,
                end_timestamp REAL NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                avg_embedding BLOB NOT NULL,
                boundary_score REAL NOT NULL,
                segment_type TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (scene_id, model_key, segment_index)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_segment_scene
            ON scene_segments(scene_id, model_key)
        """
        )

    def _migrate_to_v5(self, cursor: sqlite3.Cursor) -> None:
        """Add performer_embeddings table (v5).

        Stores aggregated performer embeddings derived from their scenes.
        Primary key is (performer_id, model_key) to support multiple models.
        """
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performer_embeddings (
                performer_id INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                embedding BLOB NOT NULL,
                contributing_scenes INTEGER NOT NULL,
                total_engagement_score REAL NOT NULL,
                visual_description TEXT,
                top_tags TEXT,
                scene_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (performer_id, model_key)
            )
        """
        )

        # Create indexes for efficient queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_performer_embedding_performer_id
            ON performer_embeddings(performer_id)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_performer_embedding_model_key
            ON performer_embeddings(model_key)
        """
        )

    def _migrate_to_v6(self, cursor: sqlite3.Cursor) -> None:
        """Add updated_at column to frame_embedding_metadata if missing (v6).

        Earlier versions of v4 migration may have created the table without
        this column. This migration adds it for existing tables.
        """
        # Check if column exists
        cursor.execute("PRAGMA table_info(frame_embedding_metadata)")
        columns = {row["name"] for row in cursor.fetchall()}

        if "updated_at" not in columns:
            # Add the missing column with default value
            cursor.execute(
                """
                ALTER TABLE frame_embedding_metadata
                ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''
            """
            )
            # Set updated_at to created_at for existing rows
            cursor.execute(
                """
                UPDATE frame_embedding_metadata
                SET updated_at = created_at
                WHERE updated_at = ''
            """
            )

    def _migrate_to_v7(self, cursor: sqlite3.Cursor) -> None:
        """Add taste map tables: taste_clusters, scene_umap_coords, tag_embeddings."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS taste_clusters (
                cluster_id INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                centroid BLOB NOT NULL,
                scene_ids TEXT NOT NULL,
                engagement_total REAL NOT NULL,
                engagement_share REAL NOT NULL,
                auto_label TEXT NOT NULL,
                user_label TEXT,
                weight_override REAL,
                excluded INTEGER DEFAULT 0,
                tag_matches TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (cluster_id, model_key)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scene_umap_coords (
                scene_id INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                cluster_id INTEGER,
                created_at TEXT NOT NULL,
                PRIMARY KEY (scene_id, model_key)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tag_embeddings (
                text TEXT NOT NULL,
                model_key TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (text, model_key)
            )
        """)

    def _migrate_to_v8(self, cursor: sqlite3.Cursor) -> None:
        """Add preference learning tables (v8)."""
        # Pairwise comparison history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_a_id INTEGER NOT NULL,
                scene_b_id INTEGER NOT NULL,
                winner_id INTEGER NOT NULL,
                phase TEXT NOT NULL,
                response_time_ms INTEGER,
                session_id TEXT NOT NULL,
                model_key TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pref_comp_session
            ON preference_comparisons(session_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pref_comp_model_key
            ON preference_comparisons(model_key)
        """)

        # Learned preference model state (one row per model_key)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_model_state (
                model_key TEXT PRIMARY KEY,
                preference_mean BLOB NOT NULL,
                preference_covariance_diag BLOB NOT NULL,
                n_comparisons INTEGER NOT NULL,
                noise_variance REAL NOT NULL,
                phase TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Session metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                comparison_count INTEGER DEFAULT 0,
                phase TEXT NOT NULL,
                convergence_avg_sigma REAL
            )
        """)

    def _migrate_to_v9(self, cursor: sqlite3.Cursor) -> None:
        """Add z column to scene_umap_coords for 3D UMAP projection."""
        cursor.execute("""
            ALTER TABLE scene_umap_coords ADD COLUMN z REAL NOT NULL DEFAULT 0.0
        """)

    def _migrate_to_v10(self, cursor: sqlite3.Cursor) -> None:
        """Add frame_tag_coverage table for tag gap detection (v10)."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frame_tag_coverage (
                scene_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                best_tag TEXT NOT NULL,
                best_similarity REAL NOT NULL,
                is_covered INTEGER NOT NULL,
                PRIMARY KEY (scene_id, frame_index, model_key)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_tag_coverage_scene
            ON frame_tag_coverage(scene_id, model_key)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_frame_tag_coverage_uncovered
            ON frame_tag_coverage(is_covered, model_key)
        """)

    def _migrate_to_v11(self, cursor: sqlite3.Cursor) -> None:
        """Add dismissed_tag_suggestions table (v11)."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dismissed_tag_suggestions (
                scene_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                dismissed_at TEXT NOT NULL,
                PRIMARY KEY (scene_id, tag_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_dismissed_scene
            ON dismissed_tag_suggestions(scene_id)
            """
        )

    def _migrate_to_v12(self, cursor: sqlite3.Cursor) -> None:
        """Add labeling tables (v12)."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS labeling_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                sampling_method TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                total_frames INTEGER NOT NULL,
                labeled_count INTEGER NOT NULL DEFAULT 0,
                skipped_count INTEGER NOT NULL DEFAULT 0,
                config_json TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_annotations (
                annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES labeling_sessions(session_id),
                scene_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                image_source TEXT NOT NULL DEFAULT 'extracted_frame',
                tag_text TEXT NOT NULL,
                tag_source TEXT NOT NULL,
                label TEXT NOT NULL,
                similarity_score REAL,
                labeled_at TEXT NOT NULL,
                UNIQUE(session_id, scene_id, frame_index, tag_text)
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_annotations_session
            ON frame_annotations(session_id)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_annotations_frame
            ON frame_annotations(scene_id, frame_index)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS labeling_progress (
                scene_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                image_source TEXT NOT NULL DEFAULT 'extracted_frame',
                session_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                PRIMARY KEY (scene_id, frame_index, session_id)
            )
            """
        )

    def _migrate_to_v13(self, cursor: sqlite3.Cursor) -> None:
        """Add dismissed_tag_merges table (v13)."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dismissed_tag_merges (
                tag_a_name TEXT NOT NULL,
                tag_b_name TEXT NOT NULL,
                dismissed_at TEXT NOT NULL,
                PRIMARY KEY (tag_a_name, tag_b_name)
            )
            """
        )

    @staticmethod
    def _pack_embedding(embedding: list[float]) -> bytes:
        """Pack embedding list into binary BLOB (float32 array)."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _unpack_embedding(blob: bytes) -> list[float]:
        """Unpack binary BLOB to embedding list."""
        if len(blob) % 4 != 0:
            raise ValueError(f"Invalid embedding blob: length {len(blob)} not divisible by 4")
        try:
            count = len(blob) // 4  # float32 is 4 bytes
            return list(struct.unpack(f"{count}f", blob))
        except struct.error as e:
            raise ValueError(f"Failed to unpack embedding blob: {e}") from e

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
        """
        Store or update a scene embedding for the current model_key.

        Args:
            scene_id: Stash scene ID
            composite_embedding: Combined embedding vector
            text_model: Model used for text embedding
            visual_embedding: Optional visual-only embedding
            metadata_embedding: Optional metadata-only embedding
            visual_model: Model used for visual description
            visual_description: Text description from VLM
            metadata_text: Concatenated metadata text
        """
        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if record exists to preserve created_at
        cursor.execute(
            "SELECT created_at FROM scene_embeddings WHERE scene_id = ? AND model_key = ?",
            (scene_id, self.model_key),
        )
        existing = cursor.fetchone()
        created_at = existing["created_at"] if existing else now

        cursor.execute(
            """
            INSERT OR REPLACE INTO scene_embeddings (
                scene_id, model_key, visual_embedding, metadata_embedding,
                composite_embedding, visual_model, text_model,
                dimensions, visual_description, metadata_text,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                scene_id,
                self.model_key,
                self._pack_embedding(visual_embedding) if visual_embedding else None,
                (self._pack_embedding(metadata_embedding) if metadata_embedding else None),
                self._pack_embedding(composite_embedding),
                visual_model,
                text_model,
                len(composite_embedding),
                visual_description,
                metadata_text,
                created_at,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def get_embedding(self, scene_id: int) -> SceneEmbeddingRecord | None:
        """
        Retrieve embedding for a scene with the current model_key.

        Args:
            scene_id: Stash scene ID

        Returns:
            SceneEmbeddingRecord or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM scene_embeddings WHERE scene_id = ? AND model_key = ?
        """,
            (scene_id, self.model_key),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "scene_id": row["scene_id"],
            "model_key": row["model_key"],
            "visual_embedding": (
                self._unpack_embedding(row["visual_embedding"]) if row["visual_embedding"] else None
            ),
            "metadata_embedding": (
                self._unpack_embedding(row["metadata_embedding"])
                if row["metadata_embedding"]
                else None
            ),
            "composite_embedding": self._unpack_embedding(row["composite_embedding"]),
            "visual_model": row["visual_model"],
            "text_model": row["text_model"],
            "dimensions": row["dimensions"],
            "visual_description": row["visual_description"],
            "metadata_text": row["metadata_text"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def has_embedding(self, scene_id: int) -> bool:
        """Check if a scene has an embedding stored for the current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM scene_embeddings WHERE scene_id = ? AND model_key = ?",
            (scene_id, self.model_key),
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def delete_embedding(self, scene_id: int) -> bool:
        """Delete embedding for a scene with the current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM scene_embeddings WHERE scene_id = ? AND model_key = ?",
            (scene_id, self.model_key),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def get_all_embeddings(self) -> list[tuple[int, list[float]]]:
        """
        Get all scene IDs and their composite embeddings for the current model_key.

        Returns:
            List of (scene_id, embedding) tuples
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT scene_id, composite_embedding FROM scene_embeddings
            WHERE model_key = ?
        """,
            (self.model_key,),
        )

        results = [
            (row["scene_id"], self._unpack_embedding(row["composite_embedding"]))
            for row in cursor.fetchall()
        ]

        conn.close()
        return results

    def get_all_frame_representatives(self, k: int = 16) -> dict[int, NDArray[np.float32]]:
        """Load K evenly-spaced frame embeddings per scene for the current model_key.

        Uses a ROW_NUMBER() window function to subsample K frames per
        scene without loading all ~300 frames per scene.  With 12K
        scenes and k=16, this reads ~200K rows (~1.5s) instead of
        4M rows (~30s).

        Set *k* = 0 to load all frames (slow for large libraries).

        Returns numpy arrays directly from BLOBs via ``np.frombuffer``
        to avoid the ~6x memory overhead of intermediate Python float
        lists (24 bytes per Python float vs 4 bytes per float32).

        Args:
            k: Max frames per scene.  Defaults to 16.  0 means all frames.

        Returns:
            Dict mapping scene_id to a ``(N, dims)`` float32 array of
            frame embeddings, ordered by frame_index.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if k > 0:
            # Compute step size from the average frame count per scene.
            # Using modular arithmetic (frame_index % step = 0) is ~30x
            # faster than ROW_NUMBER() window functions because SQLite
            # can evaluate the filter per-row without scanning all rows
            # to compute partition counts.
            cursor.execute(
                """
                SELECT COUNT(*) AS total,
                       COUNT(DISTINCT scene_id) AS n_scenes
                FROM frame_embeddings
                WHERE model_key = ?
                """,
                (self.model_key,),
            )
            stats = cursor.fetchone()
            total_frames = stats["total"] if stats else 0
            n_scenes = stats["n_scenes"] if stats else 0

            if n_scenes > 0 and total_frames > 0:
                avg_per_scene = total_frames / n_scenes
                step = max(1, int(avg_per_scene / k))
            else:
                step = 1

            cursor.execute(
                """
                SELECT scene_id, embedding
                FROM frame_embeddings
                WHERE model_key = ? AND frame_index % ? = 0
                ORDER BY scene_id, frame_index
                """,
                (self.model_key, step),
            )
        else:
            cursor.execute(
                """
                SELECT scene_id, embedding
                FROM frame_embeddings
                WHERE model_key = ?
                ORDER BY scene_id, frame_index
                """,
                (self.model_key,),
            )

        # Stream rows and build numpy arrays per scene.  Rows arrive
        # ordered by (scene_id, frame_index), so we flush each scene's
        # accumulated BLOBs into a contiguous numpy array as soon as
        # the scene_id changes.  Peak overhead is one scene's worth of
        # raw bytes (~300 × 3 KB ≈ 900 KB) on top of the final arrays.
        result: dict[int, NDArray[np.float32]] = {}
        current_sid: int | None = None
        current_blobs: list[bytes] = []

        def _flush_scene(sid: int, blobs: list[bytes]) -> None:
            dims = len(blobs[0]) // 4
            arr = np.empty((len(blobs), dims), dtype=np.float32)
            for i, blob in enumerate(blobs):
                arr[i] = np.frombuffer(blob, dtype=np.float32)
            result[sid] = arr

        for row in cursor:
            sid: int = row["scene_id"]
            if sid != current_sid:
                if current_sid is not None:
                    _flush_scene(current_sid, current_blobs)
                    current_blobs = []
                current_sid = sid
            current_blobs.append(bytes(row["embedding"]))

        if current_sid is not None and current_blobs:
            _flush_scene(current_sid, current_blobs)

        conn.close()
        return result

    def get_scene_frames(self, scene_id: int) -> NDArray[np.float32] | None:
        """Load ALL frame embeddings for a single scene.

        Used for lazy per-scene loading when full frame-level precision is
        needed (e.g. during a swipe update) without the cost of reading
        every scene's frames from the database.

        Returns:
            ``(N, dims)`` float32 array of all frame embeddings for the
            scene, ordered by frame_index.  ``None`` if no frame
            embeddings exist for this scene.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT embedding
            FROM frame_embeddings
            WHERE model_key = ? AND scene_id = ?
            ORDER BY frame_index
            """,
            (self.model_key, scene_id),
        )

        blobs: list[bytes] = [bytes(row["embedding"]) for row in cursor]
        conn.close()

        if not blobs:
            return None

        dims = len(blobs[0]) // 4
        arr = np.empty((len(blobs), dims), dtype=np.float32)
        for i, blob in enumerate(blobs):
            arr[i] = np.frombuffer(blob, dtype=np.float32)
        return arr

    # ------------------------------------------------------------------
    # Frame-level scoring via numpy memmap
    # ------------------------------------------------------------------

    def _vectors_npy_path(self) -> Path:
        """Path to the numpy vectors file derived from the FAISS index."""
        safe_key = self.model_key.replace(":", "-").replace("/", "-")
        return Path(self.db_path).parent / f"frame_vectors_{safe_key}.npy"

    def _faiss_meta_path(self) -> Path:
        """Path to the FAISS metadata file (scene_ids per frame)."""
        safe_key = self.model_key.replace(":", "-").replace("/", "-")
        return Path(self.db_path).parent / f"frame_search_{safe_key}_meta.npz"

    def _faiss_index_path(self) -> Path:
        """Path to the FAISS index file."""
        safe_key = self.model_key.replace(":", "-").replace("/", "-")
        return Path(self.db_path).parent / f"frame_search_{safe_key}.index"

    def _ensure_vectors_npy(self) -> bool:
        """Ensure the numpy vectors file exists and is in sync with FAISS.

        Rebuilds the numpy file if the FAISS index is newer (e.g. after
        rebuilding the index when new scenes are embedded).

        Returns:
            True if the file exists (or was built), False if no FAISS
            index is available to build from.
        """
        npy_path = self._vectors_npy_path()
        faiss_path = self._faiss_index_path()

        if not faiss_path.exists():
            return npy_path.exists()

        # Rebuild if npy is missing or older than the FAISS index.
        if npy_path.exists():
            npy_mtime = npy_path.stat().st_mtime
            faiss_mtime = faiss_path.stat().st_mtime
            if npy_mtime >= faiss_mtime:
                return True
            # FAISS index is newer — npy is stale.
            npy_path.unlink()

        try:
            import faiss
        except ImportError:
            return False

        index = faiss.read_index(str(faiss_path))
        vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(
            index.ntotal, index.d
        )
        np.save(npy_path, np.array(vectors, dtype=np.float32))
        del index
        return True

    def score_all_frames(
        self,
        mu: NDArray[np.float32],
    ) -> dict[int, NDArray[np.float32]] | None:
        """Score ALL frame embeddings against mu using numpy memmap.

        Opens the vectors file via memory-mapped I/O (instant, no Python
        row iteration) and computes ``vectors @ mu`` as a single
        vectorized BLAS operation.  Groups results by scene_id and
        returns the best-matching frame per scene.

        Performance: ~0.5s warm cache, ~7s cold (vs ~30s via SQLite).

        Returns ``None`` if the FAISS index hasn't been built yet.

        Args:
            mu: Preference vector to score against.

        Returns:
            Dict mapping scene_id to the best-matching frame embedding,
            or ``None`` if no frame index is available.
        """
        if not self._ensure_vectors_npy():
            return None

        meta_path = self._faiss_meta_path()
        if not meta_path.exists():
            return None

        # Load metadata (scene_ids per frame, ~60 MB).
        meta = np.load(meta_path)
        scene_ids: NDArray[np.int64] = meta["scene_ids"]

        # Open vectors via memmap (instant — no data read until accessed).
        npy_path = self._vectors_npy_path()
        vectors = np.load(npy_path, mmap_mode="r")

        if len(scene_ids) != vectors.shape[0]:
            # FAISS metadata and vectors file are out of sync.
            return None

        # Vectorized dot product: 4M × 1024 → 4M scores.
        all_scores: NDArray[np.float32] = vectors @ mu

        # Group by scene_id and pick best frame per scene.
        unique_sids = np.unique(scene_ids)
        sorted_order = np.argsort(scene_ids, kind="mergesort")
        sorted_sids = scene_ids[sorted_order]
        sorted_scores = all_scores[sorted_order]
        boundaries = np.searchsorted(sorted_sids, unique_sids)

        result: dict[int, NDArray[np.float32]] = {}
        n = len(unique_sids)
        for i in range(n):
            start = int(boundaries[i])
            end = int(boundaries[i + 1]) if i + 1 < n else len(sorted_sids)
            best_local = start + int(np.argmax(sorted_scores[start:end]))
            result[int(unique_sids[i])] = np.array(
                vectors[sorted_order[best_local]], dtype=np.float32
            )

        del vectors  # Release memmap.
        return result

    def get_embedded_scene_ids(self) -> list[int]:
        """Get list of all scene IDs that have embeddings for the current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT scene_id FROM scene_embeddings WHERE model_key = ? ORDER BY scene_id",
            (self.model_key,),
        )
        ids = [row["scene_id"] for row in cursor.fetchall()]
        conn.close()
        return ids

    def find_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        offset: int = 0,
        exclude_scene_ids: list[int] | None = None,
        min_similarity: float = 0.0,
        visual_weight: float | None = None,
        query_visual_embedding: list[float] | None = None,
        query_metadata_embedding: list[float] | None = None,
    ) -> list[SimilarityResult]:
        """
        Find scenes most similar to the query embedding within the current model_key.

        Uses cosine similarity (assumes normalized embeddings).
        Currently loads all embeddings into memory for brute-force search.
        For large libraries (>10k scenes), consider adding SQLite vector
        extension or approximate nearest neighbors.

        Args:
            query_embedding: Query vector (composite, used as fallback)
            limit: Maximum results to return
            offset: Number of results to skip (for pagination)
            exclude_scene_ids: Scene IDs to exclude from results
            min_similarity: Minimum similarity threshold
            visual_weight: Optional weight for dynamic embedding blend (0.0-1.0).
                If provided, embeddings are recomputed as:
                weighted = visual_weight * visual + (1 - visual_weight) * metadata
            query_visual_embedding: Query scene's visual embedding (for dynamic weight)
            query_metadata_embedding: Query scene's metadata embedding (for dynamic weight)

        Returns:
            List of SimilarityResult sorted by similarity descending
        """
        exclude_set = set(exclude_scene_ids or [])
        use_dynamic_weight = (
            visual_weight is not None
            and query_visual_embedding is not None
            and query_metadata_embedding is not None
        )

        # Prepare query embedding
        if use_dynamic_weight:
            query_arr = self._compute_weighted_embedding(
                query_visual_embedding,  # type: ignore[arg-type]
                query_metadata_embedding,  # type: ignore[arg-type]
                visual_weight,  # type: ignore[arg-type]
            )
        else:
            query_arr = np.array(query_embedding, dtype=np.float32)

        # Normalize query if not already
        query_norm: float = float(np.linalg.norm(query_arr))
        if query_norm > 0:
            query_arr = query_arr / query_norm

        conn = self._get_connection()
        cursor = conn.cursor()

        # Select columns based on whether we need dynamic weighting
        # Always filter by model_key
        if use_dynamic_weight:
            cursor.execute(
                """
                SELECT scene_id, composite_embedding, visual_embedding,
                       metadata_embedding, visual_description
                FROM scene_embeddings
                WHERE model_key = ?
            """,
                (self.model_key,),
            )
        else:
            cursor.execute(
                """
                SELECT scene_id, composite_embedding, visual_description
                FROM scene_embeddings
                WHERE model_key = ?
            """,
                (self.model_key,),
            )

        similarities: list[tuple[int, float, str | None]] = []

        for row in cursor.fetchall():
            scene_id: int = row["scene_id"]
            if scene_id in exclude_set:
                continue

            # Compute stored embedding (dynamic or composite)
            if use_dynamic_weight:
                stored_arr = self._get_weighted_embedding_from_row(
                    row,
                    visual_weight,  # type: ignore
                )
            else:
                stored_emb = self._unpack_embedding(row["composite_embedding"])
                stored_arr = np.array(stored_emb, dtype=np.float32)

            # Cosine similarity (dot product of normalized vectors)
            similarity = float(np.dot(query_arr, stored_arr))

            if similarity >= min_similarity:
                similarities.append((scene_id, similarity, row["visual_description"]))

        conn.close()

        # Sort by similarity descending and apply offset/limit
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [
            SimilarityResult(
                scene_id=s[0],
                similarity=s[1],
                visual_description=s[2],
            )
            for s in similarities[offset : offset + limit]
        ]

    def _compute_weighted_embedding(
        self,
        visual_embedding: list[float],
        metadata_embedding: list[float],
        visual_weight: float,
    ) -> NDArray[np.float32]:
        """
        Compute weighted combination of visual and metadata embeddings.

        Args:
            visual_embedding: Visual embedding vector
            metadata_embedding: Metadata embedding vector
            visual_weight: Weight for visual component (0.0-1.0)

        Returns:
            Normalized weighted embedding as numpy array
        """
        v_arr = np.array(visual_embedding, dtype=np.float32)
        m_arr = np.array(metadata_embedding, dtype=np.float32)

        # Weighted combination
        combined = visual_weight * v_arr + (1.0 - visual_weight) * m_arr

        # Normalize to unit vector
        norm = float(np.linalg.norm(combined))
        if norm > 0:
            combined = combined / norm

        return combined

    def _get_weighted_embedding_from_row(
        self,
        row: sqlite3.Row,
        visual_weight: float,
    ) -> NDArray[np.float32]:
        """
        Get weighted embedding from a database row, falling back to composite.

        Args:
            row: Database row with embedding columns
            visual_weight: Weight for visual component (0.0-1.0)

        Returns:
            Normalized embedding as numpy array
        """
        visual_blob = row["visual_embedding"]
        metadata_blob = row["metadata_embedding"]

        # Fall back to composite if either embedding is missing
        if visual_blob is None or metadata_blob is None:
            stored_emb = self._unpack_embedding(row["composite_embedding"])
            return np.array(stored_emb, dtype=np.float32)

        visual_emb = self._unpack_embedding(visual_blob)
        metadata_emb = self._unpack_embedding(metadata_blob)

        return self._compute_weighted_embedding(visual_emb, metadata_emb, visual_weight)

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics for the current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Stats for current model_key
        cursor.execute(
            "SELECT COUNT(*) as count FROM scene_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        count = cursor.fetchone()["count"]

        cursor.execute(
            """
            SELECT dimensions, COUNT(*) as count
            FROM scene_embeddings
            WHERE model_key = ?
            GROUP BY dimensions
        """,
            (self.model_key,),
        )
        dims = {row["dimensions"]: row["count"] for row in cursor.fetchall()}

        cursor.execute(
            """
            SELECT MIN(created_at) as oldest, MAX(updated_at) as newest
            FROM scene_embeddings
            WHERE model_key = ?
        """,
            (self.model_key,),
        )
        dates = cursor.fetchone()

        # Also get overall stats across all models
        cursor.execute("SELECT COUNT(*) as count FROM scene_embeddings")
        total_count = cursor.fetchone()["count"]

        cursor.execute(
            """
            SELECT model_key, COUNT(*) as count, dimensions
            FROM scene_embeddings
            GROUP BY model_key
        """
        )
        models = {
            row["model_key"]: {"count": row["count"], "dimensions": row["dimensions"]}
            for row in cursor.fetchall()
        }

        conn.close()

        return {
            "model_key": self.model_key,
            "total_embeddings": count,
            "dimensions_distribution": dims,
            "oldest_embedding": dates["oldest"],
            "newest_embedding": dates["newest"],
            "db_path": self.db_path,
            # Cross-model stats
            "all_models": models,
            "total_across_all_models": total_count,
        }

    def get_available_model_keys(self) -> list[str]:
        """Get list of all model keys that have embeddings stored."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT model_key FROM scene_embeddings ORDER BY model_key")
        keys = [row["model_key"] for row in cursor.fetchall()]
        conn.close()
        return keys

    def clear_all(self) -> int:
        """
        Delete all embeddings for the current model_key.

        Returns:
            Number of embeddings deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM scene_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def clear_all_models(self) -> int:
        """
        Delete all embeddings across all model keys.

        Returns:
            Number of embeddings deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scene_embeddings")
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    # =========================================================================
    # O-Moment Embedding Methods
    # =========================================================================

    def store_o_moment_embedding(
        self,
        scene_id: int,
        o_event_index: int,
        marker_id: int,
        center_timestamp: float,
        window_seconds: float,
        embedding: list[float],
        frame_count: int,
    ) -> None:
        """
        Store an O-moment embedding for the current model_key.

        Args:
            scene_id: Stash scene ID
            o_event_index: Index of O-moment for this scene (0-indexed)
            marker_id: O marker ID from scene_markers table
            center_timestamp: Center position of extraction window (seconds)
            window_seconds: Total window size (e.g., 120 for +/- 60s)
            embedding: Embedding vector from averaged frames
            frame_count: Number of frames that were averaged
        """
        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO o_moment_embeddings (
                scene_id, o_event_index, marker_id, center_timestamp,
                window_seconds, embedding, frame_count, model_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                scene_id,
                o_event_index,
                marker_id,
                center_timestamp,
                window_seconds,
                self._pack_embedding(embedding),
                frame_count,
                self.model_key,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def get_o_moment_embedding(
        self,
        scene_id: int,
        o_event_index: int,
    ) -> OMomentEmbeddingRecord | None:
        """
        Retrieve O-moment embedding for a scene/index with current model_key.

        Args:
            scene_id: Stash scene ID
            o_event_index: Index of O-moment for this scene (0-indexed)

        Returns:
            OMomentEmbeddingRecord or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM o_moment_embeddings
            WHERE scene_id = ? AND o_event_index = ? AND model_key = ?
        """,
            (scene_id, o_event_index, self.model_key),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "scene_id": row["scene_id"],
            "o_event_index": row["o_event_index"],
            "marker_id": row["marker_id"],
            "center_timestamp": row["center_timestamp"],
            "window_seconds": row["window_seconds"],
            "embedding": self._unpack_embedding(row["embedding"]),
            "frame_count": row["frame_count"],
            "model_key": row["model_key"],
            "created_at": row["created_at"],
        }

    def get_all_o_moment_embeddings_for_scene(
        self,
        scene_id: int,
    ) -> list[OMomentEmbeddingRecord]:
        """
        Get all O-moment embeddings for a scene with current model_key.

        Args:
            scene_id: Stash scene ID

        Returns:
            List of OMomentEmbeddingRecord for all O-moments in the scene
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM o_moment_embeddings
            WHERE scene_id = ? AND model_key = ?
            ORDER BY o_event_index
        """,
            (scene_id, self.model_key),
        )

        results: list[OMomentEmbeddingRecord] = []
        for row in cursor.fetchall():
            record = cast(
                "OMomentEmbeddingRecord",
                {
                    "scene_id": row["scene_id"],
                    "o_event_index": row["o_event_index"],
                    "marker_id": row["marker_id"],
                    "center_timestamp": row["center_timestamp"],
                    "window_seconds": row["window_seconds"],
                    "embedding": self._unpack_embedding(row["embedding"]),
                    "frame_count": row["frame_count"],
                    "model_key": row["model_key"],
                    "created_at": row["created_at"],
                },
            )
            results.append(record)

        conn.close()
        return results

    def get_all_o_moment_embeddings(
        self,
    ) -> list[tuple[int, int, list[float]]]:
        """
        Get all O-moment embeddings for current model_key.

        Returns:
            List of (scene_id, marker_id, embedding) tuples
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT scene_id, marker_id, embedding FROM o_moment_embeddings
            WHERE model_key = ?
            ORDER BY scene_id, o_event_index
        """,
            (self.model_key,),
        )

        results = [
            (row["scene_id"], row["marker_id"], self._unpack_embedding(row["embedding"]))
            for row in cursor.fetchall()
        ]

        conn.close()
        return results

    def get_scenes_with_o_moments(self) -> list[int]:
        """Get list of scene IDs that have O-moment embeddings for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT scene_id FROM o_moment_embeddings
            WHERE model_key = ?
            ORDER BY scene_id
        """,
            (self.model_key,),
        )
        ids = [row["scene_id"] for row in cursor.fetchall()]
        conn.close()
        return ids

    def has_o_moment_embedding(self, scene_id: int, o_event_index: int) -> bool:
        """Check if an O-moment embedding exists for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM o_moment_embeddings
            WHERE scene_id = ? AND o_event_index = ? AND model_key = ?
        """,
            (scene_id, o_event_index, self.model_key),
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def delete_o_moment_embedding(self, scene_id: int, o_event_index: int) -> bool:
        """Delete O-moment embedding for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM o_moment_embeddings
            WHERE scene_id = ? AND o_event_index = ? AND model_key = ?
        """,
            (scene_id, o_event_index, self.model_key),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def delete_all_o_moments_for_scene(self, scene_id: int) -> int:
        """Delete all O-moment embeddings for a scene with current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM o_moment_embeddings
            WHERE scene_id = ? AND model_key = ?
        """,
            (scene_id, self.model_key),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def clear_all_o_moments(self) -> int:
        """Delete all O-moment embeddings for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM o_moment_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def get_o_moment_stats(self) -> dict[str, Any]:
        """Get O-moment embedding statistics for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Count total O-moment embeddings
        cursor.execute(
            "SELECT COUNT(*) as count FROM o_moment_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        count = cursor.fetchone()["count"]

        # Count unique scenes with O-moments
        cursor.execute(
            """
            SELECT COUNT(DISTINCT scene_id) as scene_count
            FROM o_moment_embeddings WHERE model_key = ?
        """,
            (self.model_key,),
        )
        scene_count = cursor.fetchone()["scene_count"]

        # Get date range
        cursor.execute(
            """
            SELECT MIN(created_at) as oldest, MAX(created_at) as newest
            FROM o_moment_embeddings WHERE model_key = ?
        """,
            (self.model_key,),
        )
        dates = cursor.fetchone()

        conn.close()

        return {
            "model_key": self.model_key,
            "total_o_moments": count,
            "scenes_with_o_moments": scene_count,
            "oldest_o_moment": dates["oldest"],
            "newest_o_moment": dates["newest"],
        }

    def find_similar_o_moments(
        self,
        query_embedding: list[float],
        limit: int = 10,
        exclude_scene_ids: list[int] | None = None,
        min_similarity: float = 0.0,
    ) -> list[SimilarityResult]:
        """
        Find scenes with O-moment embeddings similar to the query.

        Uses cosine similarity (assumes normalized embeddings).
        Returns the best-matching O-moment for each scene.

        Note: Results are per-scene (best matching O-moment), so marker_id is
        used internally but not included in SimilarityResult. Use
        get_all_o_moment_embeddings_for_scene() if you need marker details.

        Args:
            query_embedding: Query vector (e.g., averaged O-moment profile)
            limit: Maximum results to return
            exclude_scene_ids: Scene IDs to exclude from results
            min_similarity: Minimum similarity threshold

        Returns:
            List of SimilarityResult sorted by similarity descending
        """
        exclude_set = set(exclude_scene_ids or [])

        query_arr = np.array(query_embedding, dtype=np.float32)
        query_norm: float = float(np.linalg.norm(query_arr))
        if query_norm > 0:
            query_arr = query_arr / query_norm

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT scene_id, marker_id, embedding FROM o_moment_embeddings
            WHERE model_key = ?
        """,
            (self.model_key,),
        )

        # Group by scene_id, keep best match
        scene_best: dict[int, tuple[float, int]] = {}  # scene_id -> (similarity, marker_id)

        for row in cursor.fetchall():
            scene_id: int = row["scene_id"]
            if scene_id in exclude_set:
                continue

            stored_emb = self._unpack_embedding(row["embedding"])
            stored_arr = np.array(stored_emb, dtype=np.float32)

            similarity = float(np.dot(query_arr, stored_arr))

            if similarity >= min_similarity:
                if scene_id not in scene_best or similarity > scene_best[scene_id][0]:
                    scene_best[scene_id] = (similarity, row["marker_id"])

        conn.close()

        # Sort by similarity descending
        sorted_results = sorted(
            scene_best.items(),
            key=lambda x: x[1][0],
            reverse=True,
        )

        return [
            SimilarityResult(
                scene_id=scene_id,
                similarity=sim,
                visual_description=None,
            )
            for scene_id, (sim, _) in sorted_results[:limit]
        ]

    # =========================================================================
    # Dense Frame-Level Embedding Methods
    # =========================================================================

    def store_frame_embedding(
        self,
        scene_id: int,
        frame_index: int,
        timestamp: float,
        embedding: list[float],
    ) -> None:
        """
        Store a single frame embedding for the current model_key.

        Args:
            scene_id: Stash scene ID
            frame_index: 0-based frame index (frame N at second N for 1fps)
            timestamp: Exact timestamp in seconds
            embedding: Embedding vector for this frame
        """
        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO frame_embeddings (
                scene_id, frame_index, timestamp, embedding, model_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                scene_id,
                frame_index,
                timestamp,
                self._pack_embedding(embedding),
                self.model_key,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def store_frame_metadata(
        self,
        scene_id: int,
        frame_count: int,
        total_frames_extracted: int,
        duration: float,
        sampling_rate: float,
        composite_embedding: list[float],
        dedup_ratio: float,
        first_frame_timestamp: float | None = None,
        last_frame_timestamp: float | None = None,
    ) -> None:
        """
        Store scene-level frame embedding metadata for the current model_key.

        Args:
            scene_id: Stash scene ID
            frame_count: Number of unique frames stored (after deduplication)
            total_frames_extracted: Total frames extracted (before deduplication)
            duration: Scene duration in seconds
            sampling_rate: Frames per second (should be 1.0 for 1fps)
            composite_embedding: Averaged embedding from ALL frames (fast similarity)
            dedup_ratio: Percentage of frames skipped (0-1)
            first_frame_timestamp: Timestamp of first frame
            last_frame_timestamp: Timestamp of last frame
        """
        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if record exists to preserve created_at
        cursor.execute(
            """
            SELECT created_at FROM frame_embedding_metadata
            WHERE scene_id = ? AND model_key = ?
        """,
            (scene_id, self.model_key),
        )
        existing = cursor.fetchone()
        created_at = existing["created_at"] if existing else now

        cursor.execute(
            """
            INSERT OR REPLACE INTO frame_embedding_metadata (
                scene_id, model_key, frame_count, total_frames_extracted,
                duration, sampling_rate, composite_embedding, dedup_ratio,
                first_frame_timestamp, last_frame_timestamp,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                scene_id,
                self.model_key,
                frame_count,
                total_frames_extracted,
                duration,
                sampling_rate,
                self._pack_embedding(composite_embedding),
                dedup_ratio,
                first_frame_timestamp,
                last_frame_timestamp,
                created_at,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def get_frame_embedding_metadata(self, scene_id: int) -> FrameEmbeddingMetadata | None:
        """
        Retrieve frame embedding metadata for a scene with current model_key.

        Args:
            scene_id: Stash scene ID

        Returns:
            FrameEmbeddingMetadata or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM frame_embedding_metadata
            WHERE scene_id = ? AND model_key = ?
        """,
            (scene_id, self.model_key),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "scene_id": row["scene_id"],
            "model_key": row["model_key"],
            "frame_count": row["frame_count"],
            "total_frames_extracted": row["total_frames_extracted"],
            "duration": row["duration"],
            "sampling_rate": row["sampling_rate"],
            "composite_embedding": self._unpack_embedding(row["composite_embedding"]),
            "dedup_ratio": row["dedup_ratio"],
            "first_frame_timestamp": row["first_frame_timestamp"],
            "last_frame_timestamp": row["last_frame_timestamp"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def has_frame_embeddings(self, scene_id: int) -> bool:
        """Check if a scene has frame embeddings for current model_key.

        Queries the frame_embeddings table directly rather than metadata,
        since metadata may be missing for scenes embedded via standalone_embed
        or due to incomplete embedding runs.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM frame_embeddings
            WHERE scene_id = ? AND model_key = ?
            LIMIT 1
        """,
            (scene_id, self.model_key),
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def find_frames_by_embedding(
        self,
        scene_id: int,
        query_embedding: list[float],
        min_similarity: float = 0.7,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find timestamps within a scene matching query embedding.

        Use case: "When does X happen in scene Y?"

        Args:
            scene_id: Scene to search within
            query_embedding: Visual query (e.g., from text "POV angle")
            min_similarity: Minimum cosine similarity threshold
            max_results: Return top N matches

        Returns:
            List of dicts with timestamp, frame_index, and similarity
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Load all frame embeddings for scene
        cursor.execute(
            """
            SELECT frame_index, timestamp, embedding
            FROM frame_embeddings
            WHERE scene_id = ? AND model_key = ?
            ORDER BY timestamp
        """,
            (scene_id, self.model_key),
        )

        query_arr = np.array(query_embedding, dtype=np.float32)
        # Normalize query
        query_norm: float = float(np.linalg.norm(query_arr))
        if query_norm > 0:
            query_arr = query_arr / query_norm

        results: list[dict[str, Any]] = []

        for row in cursor.fetchall():
            emb = self._unpack_embedding(row["embedding"])
            emb_arr = np.array(emb, dtype=np.float32)

            # Cosine similarity (both are unit vectors)
            similarity = float(np.dot(query_arr, emb_arr))

            if similarity >= min_similarity:
                results.append(
                    {
                        "timestamp": row["timestamp"],
                        "frame_index": row["frame_index"],
                        "similarity": similarity,
                    }
                )

        conn.close()

        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:max_results]

    def _load_all_frames_for_scene(self, scene_id: int) -> list[dict[str, Any]]:
        """
        Load all frame embeddings for a scene (for segmentation).

        Returns:
            List of dicts with frame_index, timestamp, and embedding
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT frame_index, timestamp, embedding
            FROM frame_embeddings
            WHERE scene_id = ? AND model_key = ?
            ORDER BY frame_index
        """,
            (scene_id, self.model_key),
        )

        frames = []
        for row in cursor.fetchall():
            frames.append(
                {
                    "frame_index": row["frame_index"],
                    "timestamp": row["timestamp"],
                    "embedding": self._unpack_embedding(row["embedding"]),
                }
            )

        conn.close()
        return frames

    def compute_scene_segments(
        self,
        scene_id: int,
        boundary_threshold: float = 0.80,
        min_segment_duration: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Identify semantic boundaries by analyzing embedding similarity changes.

        Algorithm:
        1. Load all frame embeddings for scene (ordered by timestamp)
        2. Compute cosine similarity between consecutive frames
        3. When similarity drops below threshold, mark boundary
        4. Merge short segments (< min_duration)

        Use cases:
        - Auto-detect scene changes (camera cuts, location changes)
        - Generate chapter markers for long videos
        - Identify action sequences vs static moments

        Args:
            scene_id: Scene to segment
            boundary_threshold: Similarity drop below this = new segment
            min_segment_duration: Merge segments shorter than this

        Returns:
            List of segment dicts with start/end times, frame counts, etc.
        """
        # Load all frames and embeddings
        frames = self._load_all_frames_for_scene(scene_id)
        if len(frames) < 2:
            return []

        # Find boundaries by similarity drops
        boundaries = [0]  # Start of scene

        for i in range(1, len(frames)):
            prev_emb = np.array(frames[i - 1]["embedding"], dtype=np.float32)
            curr_emb = np.array(frames[i]["embedding"], dtype=np.float32)

            similarity = float(np.dot(prev_emb, curr_emb))

            # Boundary detected when similarity drops significantly
            if similarity < boundary_threshold:
                boundaries.append(i)

        boundaries.append(len(frames))  # End of scene

        # Build segments
        segments: list[dict[str, Any]] = []
        segment_index = 0

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            start_time = frames[start_idx]["timestamp"]
            end_time = frames[end_idx - 1]["timestamp"]
            duration = end_time - start_time

            # Skip short segments (except first)
            if duration < min_segment_duration and i > 0:
                continue

            # Average embeddings in segment
            segment_embs = [frames[j]["embedding"] for j in range(start_idx, end_idx)]
            avg_emb = self._average_embeddings(segment_embs)

            # Boundary score (similarity to next segment)
            boundary_score = 0.0
            if i < len(boundaries) - 2:
                next_start_idx = boundaries[i + 1]
                next_emb = np.array(frames[next_start_idx]["embedding"], dtype=np.float32)
                curr_avg = np.array(avg_emb, dtype=np.float32)
                boundary_score = float(np.dot(curr_avg, next_emb))

            segments.append(
                {
                    "segment_index": segment_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "frame_count": end_idx - start_idx,
                    "avg_embedding": avg_emb,
                    "boundary_score": boundary_score,
                }
            )

            segment_index += 1

        return segments

    def _average_embeddings(self, embeddings: list[list[float]]) -> list[float]:
        """
        Average multiple embeddings into a single embedding.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Averaged and normalized embedding
        """
        if not embeddings:
            raise ValueError("Cannot average empty list of embeddings")

        # Convert to numpy array and average
        emb_arr = np.array(embeddings, dtype=np.float32)
        avg = np.mean(emb_arr, axis=0)

        # Normalize to unit vector
        norm = float(np.linalg.norm(avg))
        if norm > 0:
            avg = avg / norm

        result: list[float] = avg.tolist()
        return result

    def store_scene_segment(
        self,
        scene_id: int,
        segment_index: int,
        start_timestamp: float,
        end_timestamp: float,
        start_frame: int,
        end_frame: int,
        avg_embedding: list[float],
        boundary_score: float,
        segment_type: str | None = None,
    ) -> None:
        """
        Store a pre-computed scene segment for the current model_key.

        Args:
            scene_id: Stash scene ID
            segment_index: 0-based segment number
            start_timestamp: Segment start in seconds
            end_timestamp: Segment end in seconds
            start_frame: Start frame index
            end_frame: End frame index
            avg_embedding: Average embedding for segment
            boundary_score: Similarity drop score (0-1)
            segment_type: Optional type (e.g., 'camera_change', 'scene_change')
        """
        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO scene_segments (
                scene_id, model_key, segment_index, start_timestamp,
                end_timestamp, start_frame, end_frame, avg_embedding,
                boundary_score, segment_type, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                scene_id,
                self.model_key,
                segment_index,
                start_timestamp,
                end_timestamp,
                start_frame,
                end_frame,
                self._pack_embedding(avg_embedding),
                boundary_score,
                segment_type,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def get_scene_segments(self, scene_id: int) -> list[SceneSegment]:
        """
        Retrieve all stored segments for a scene with current model_key.

        Args:
            scene_id: Stash scene ID

        Returns:
            List of SceneSegment records
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM scene_segments
            WHERE scene_id = ? AND model_key = ?
            ORDER BY segment_index
        """,
            (scene_id, self.model_key),
        )

        segments: list[SceneSegment] = []
        for row in cursor.fetchall():
            segments.append(
                cast(
                    "SceneSegment",
                    {
                        "scene_id": row["scene_id"],
                        "model_key": row["model_key"],
                        "segment_index": row["segment_index"],
                        "start_timestamp": row["start_timestamp"],
                        "end_timestamp": row["end_timestamp"],
                        "start_frame": row["start_frame"],
                        "end_frame": row["end_frame"],
                        "avg_embedding": self._unpack_embedding(row["avg_embedding"]),
                        "boundary_score": row["boundary_score"],
                        "segment_type": row["segment_type"],
                        "created_at": row["created_at"],
                    },
                )
            )

        conn.close()
        return segments

    def delete_frame_embeddings(self, scene_id: int) -> int:
        """
        Delete all frame embeddings for a scene with current model_key.

        Returns:
            Number of frames deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Delete frame embeddings
        cursor.execute(
            """
            DELETE FROM frame_embeddings
            WHERE scene_id = ? AND model_key = ?
        """,
            (scene_id, self.model_key),
        )
        deleted_frames = cursor.rowcount

        # Delete metadata
        cursor.execute(
            """
            DELETE FROM frame_embedding_metadata
            WHERE scene_id = ? AND model_key = ?
        """,
            (scene_id, self.model_key),
        )

        # Delete segments
        cursor.execute(
            """
            DELETE FROM scene_segments
            WHERE scene_id = ? AND model_key = ?
        """,
            (scene_id, self.model_key),
        )

        conn.commit()
        conn.close()
        return deleted_frames

    def get_frame_embedding_stats(self) -> dict[str, Any]:
        """Get frame embedding statistics for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Count scenes with frame embeddings
        cursor.execute(
            """
            SELECT COUNT(*) as count FROM frame_embedding_metadata
            WHERE model_key = ?
        """,
            (self.model_key,),
        )
        scene_count = cursor.fetchone()["count"]

        # Count total frames
        cursor.execute(
            """
            SELECT COUNT(*) as count FROM frame_embeddings
            WHERE model_key = ?
        """,
            (self.model_key,),
        )
        frame_count = cursor.fetchone()["count"]

        # Average dedup ratio
        cursor.execute(
            """
            SELECT AVG(dedup_ratio) as avg_dedup FROM frame_embedding_metadata
            WHERE model_key = ?
        """,
            (self.model_key,),
        )
        avg_dedup = cursor.fetchone()["avg_dedup"] or 0.0

        # Date range
        cursor.execute(
            """
            SELECT MIN(created_at) as oldest, MAX(updated_at) as newest
            FROM frame_embedding_metadata
            WHERE model_key = ?
        """,
            (self.model_key,),
        )
        dates = cursor.fetchone()

        conn.close()

        return {
            "model_key": self.model_key,
            "scenes_with_frames": scene_count,
            "total_frames": frame_count,
            "avg_frames_per_scene": frame_count / scene_count if scene_count > 0 else 0,
            "avg_dedup_ratio": avg_dedup,
            "oldest_frame_embedding": dates["oldest"],
            "newest_frame_embedding": dates["newest"],
        }

    # =========================================================================
    # Scene Cleanup Methods (for handling deleted scenes)
    # =========================================================================

    def get_all_model_keys(self) -> list[str]:
        """
        Get list of all model keys that have any data stored.

        Checks scene_embeddings, o_moment_embeddings, and frame_embeddings tables.

        Returns:
            List of distinct model_key strings
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get model keys from all tables
        cursor.execute(
            """
            SELECT DISTINCT model_key FROM scene_embeddings
            UNION
            SELECT DISTINCT model_key FROM o_moment_embeddings
            UNION
            SELECT DISTINCT model_key FROM frame_embeddings
            UNION
            SELECT DISTINCT model_key FROM frame_embedding_metadata
            ORDER BY model_key
        """
        )
        keys = [row["model_key"] for row in cursor.fetchall()]
        conn.close()
        return keys

    def delete_all_scene_data(self, scene_id: int) -> dict[str, int]:
        """
        Delete ALL data for a scene across ALL model keys.

        Removes embeddings, o-moments, frames, and segments for the scene.
        This is used when a scene is deleted from Stash.

        Args:
            scene_id: Stash scene ID to delete

        Returns:
            Dict with counts of deleted items by type
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check which tables exist (resilient to missing tables from partial migrations)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?, ?, ?)",
            (
                "scene_embeddings",
                "o_moment_embeddings",
                "frame_embeddings",
                "frame_embedding_metadata",
                "scene_segments",
            ),
        )
        existing_tables = {row["name"] for row in cursor.fetchall()}

        embeddings_deleted = 0
        o_moments_deleted = 0
        frames_deleted = 0
        metadata_deleted = 0
        segments_deleted = 0

        # Delete from scene_embeddings (all model keys)
        if "scene_embeddings" in existing_tables:
            cursor.execute(
                "DELETE FROM scene_embeddings WHERE scene_id = ?",
                (scene_id,),
            )
            embeddings_deleted = cursor.rowcount

        # Delete from o_moment_embeddings (all model keys)
        if "o_moment_embeddings" in existing_tables:
            cursor.execute(
                "DELETE FROM o_moment_embeddings WHERE scene_id = ?",
                (scene_id,),
            )
            o_moments_deleted = cursor.rowcount

        # Delete from frame_embeddings (all model keys)
        if "frame_embeddings" in existing_tables:
            cursor.execute(
                "DELETE FROM frame_embeddings WHERE scene_id = ?",
                (scene_id,),
            )
            frames_deleted = cursor.rowcount

        # Delete from frame_embedding_metadata (all model keys)
        if "frame_embedding_metadata" in existing_tables:
            cursor.execute(
                "DELETE FROM frame_embedding_metadata WHERE scene_id = ?",
                (scene_id,),
            )
            metadata_deleted = cursor.rowcount

        # Delete from scene_segments (all model keys)
        if "scene_segments" in existing_tables:
            cursor.execute(
                "DELETE FROM scene_segments WHERE scene_id = ?",
                (scene_id,),
            )
            segments_deleted = cursor.rowcount

        conn.commit()
        conn.close()

        return {
            "embeddings": embeddings_deleted,
            "o_moments": o_moments_deleted,
            "frames": frames_deleted,
            "metadata": metadata_deleted,
            "segments": segments_deleted,
        }

    def get_all_stored_scene_ids(self) -> list[int]:
        """
        Get list of all scene IDs that have any data stored (across all model keys).

        Returns:
            List of distinct scene IDs with stored data
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check which tables exist (resilient to missing tables from partial migrations)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?)",
            ("scene_embeddings", "o_moment_embeddings", "frame_embeddings"),
        )
        existing_tables = {row["name"] for row in cursor.fetchall()}

        if not existing_tables:
            conn.close()
            return []

        # Build UNION query only for existing tables
        queries = []
        if "scene_embeddings" in existing_tables:
            queries.append("SELECT DISTINCT scene_id FROM scene_embeddings")
        if "o_moment_embeddings" in existing_tables:
            queries.append("SELECT DISTINCT scene_id FROM o_moment_embeddings")
        if "frame_embeddings" in existing_tables:
            queries.append("SELECT DISTINCT scene_id FROM frame_embeddings")

        union_query = " UNION ".join(queries) + " ORDER BY scene_id"
        cursor.execute(union_query)
        ids = [row["scene_id"] for row in cursor.fetchall()]
        conn.close()
        return ids

    def get_orphaned_scene_ids(self, valid_scene_ids: list[int]) -> list[int]:
        """
        Find scene IDs that have embeddings but are not in the valid_scene_ids list.

        Args:
            valid_scene_ids: List of scene IDs that exist in Stash database

        Returns:
            List of scene IDs with embeddings but no corresponding Stash scene
        """
        stored_ids = set(self.get_all_stored_scene_ids())
        valid_ids = set(valid_scene_ids)

        orphaned = stored_ids - valid_ids
        return sorted(orphaned)

    def find_similar_validated(
        self,
        query_embedding: list[float],
        valid_scene_ids: list[int],
        limit: int = 10,
        offset: int = 0,
        exclude_scene_ids: list[int] | None = None,
        min_similarity: float = 0.0,
        visual_weight: float | None = None,
        query_visual_embedding: list[float] | None = None,
        query_metadata_embedding: list[float] | None = None,
    ) -> list[SimilarityResult]:
        """
        Find similar scenes, excluding those not in valid_scene_ids.

        Same as find_similar() but adds validation against Stash scene existence.

        Args:
            query_embedding: Query vector (composite, used as fallback)
            valid_scene_ids: List of scene IDs that exist in Stash
            limit: Maximum results to return
            offset: Number of results to skip (for pagination)
            exclude_scene_ids: Scene IDs to exclude from results
            min_similarity: Minimum similarity threshold
            visual_weight: Optional weight for dynamic embedding blend
            query_visual_embedding: Query scene's visual embedding
            query_metadata_embedding: Query scene's metadata embedding

        Returns:
            List of SimilarityResult sorted by similarity descending
        """
        # Combine exclusions: explicit excludes + invalid scenes
        valid_set = set(valid_scene_ids)
        exclude_set = set(exclude_scene_ids or [])

        # Get all stored scene IDs and find invalid ones
        stored_ids = set(self.get_embedded_scene_ids())
        invalid_ids = stored_ids - valid_set

        # Merge all exclusions
        all_excludes = list(exclude_set | invalid_ids)

        return self.find_similar(
            query_embedding=query_embedding,
            limit=limit,
            offset=offset,
            exclude_scene_ids=all_excludes,
            min_similarity=min_similarity,
            visual_weight=visual_weight,
            query_visual_embedding=query_visual_embedding,
            query_metadata_embedding=query_metadata_embedding,
        )

    def get_all_embeddings_validated(
        self, valid_scene_ids: list[int]
    ) -> list[tuple[int, list[float]]]:
        """
        Get all scene embeddings, filtering out scenes not in valid_scene_ids.

        Args:
            valid_scene_ids: List of scene IDs that exist in Stash

        Returns:
            List of (scene_id, embedding) tuples for valid scenes only
        """
        valid_set = set(valid_scene_ids)
        all_embeddings = self.get_all_embeddings()

        return [
            (scene_id, embedding) for scene_id, embedding in all_embeddings if scene_id in valid_set
        ]

    def get_embedded_scene_ids_validated(self, valid_scene_ids: list[int]) -> list[int]:
        """
        Get list of embedded scene IDs, filtering out deleted scenes.

        Args:
            valid_scene_ids: List of scene IDs that exist in Stash

        Returns:
            List of scene IDs that have embeddings AND exist in Stash
        """
        valid_set = set(valid_scene_ids)
        embedded_ids = self.get_embedded_scene_ids()

        return [sid for sid in embedded_ids if sid in valid_set]

    # =========================================================================
    # Performer Embedding Methods
    # =========================================================================

    def store_performer_embedding(
        self,
        performer_id: int,
        embedding: list[float],
        contributing_scenes: int,
        total_engagement_score: float,
        scene_count: int,
        visual_description: str | None = None,
        top_tags: str | None = None,
    ) -> None:
        """
        Store or update a performer embedding for the current model_key.

        Args:
            performer_id: Stash performer ID
            embedding: Aggregated embedding vector from performer's scenes
            contributing_scenes: Number of scenes used to build embedding
            total_engagement_score: Sum of engagement scores from contributing scenes
            scene_count: Total number of scenes featuring this performer
            visual_description: AI-generated visual description
            top_tags: JSON array of most common tags
        """
        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if record exists to preserve created_at
        cursor.execute(
            "SELECT created_at FROM performer_embeddings WHERE performer_id = ? AND model_key = ?",
            (performer_id, self.model_key),
        )
        existing = cursor.fetchone()
        created_at = existing["created_at"] if existing else now

        cursor.execute(
            """
            INSERT OR REPLACE INTO performer_embeddings (
                performer_id, model_key, embedding, contributing_scenes,
                total_engagement_score, visual_description, top_tags,
                scene_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                performer_id,
                self.model_key,
                self._pack_embedding(embedding),
                contributing_scenes,
                total_engagement_score,
                visual_description,
                top_tags,
                scene_count,
                created_at,
                now,
            ),
        )

        conn.commit()
        conn.close()

    def get_performer_embedding(self, performer_id: int) -> dict[str, Any] | None:
        """
        Retrieve performer embedding for current model_key.

        Args:
            performer_id: Stash performer ID

        Returns:
            Dict with performer embedding data or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM performer_embeddings
            WHERE performer_id = ? AND model_key = ?
        """,
            (performer_id, self.model_key),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "performer_id": row["performer_id"],
            "model_key": row["model_key"],
            "embedding": self._unpack_embedding(row["embedding"]),
            "contributing_scenes": row["contributing_scenes"],
            "total_engagement_score": row["total_engagement_score"],
            "visual_description": row["visual_description"],
            "top_tags": row["top_tags"],
            "scene_count": row["scene_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def has_performer_embedding(self, performer_id: int) -> bool:
        """Check if a performer has an embedding for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM performer_embeddings WHERE performer_id = ? AND model_key = ?",
            (performer_id, self.model_key),
        )
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def delete_performer_embedding(self, performer_id: int) -> bool:
        """Delete performer embedding for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM performer_embeddings WHERE performer_id = ? AND model_key = ?",
            (performer_id, self.model_key),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def get_all_performer_embeddings(
        self,
    ) -> list[tuple[int, list[float]]]:
        """
        Get all performer IDs and their embeddings for current model_key.

        Returns:
            List of (performer_id, embedding) tuples
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT performer_id, embedding FROM performer_embeddings
            WHERE model_key = ?
        """,
            (self.model_key,),
        )

        results = [
            (row["performer_id"], self._unpack_embedding(row["embedding"]))
            for row in cursor.fetchall()
        ]

        conn.close()
        return results

    def get_embedded_performer_ids(self) -> list[int]:
        """Get list of all performer IDs with embeddings for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT performer_id FROM performer_embeddings WHERE model_key = ? ORDER BY performer_id",
            (self.model_key,),
        )
        ids = [row["performer_id"] for row in cursor.fetchall()]
        conn.close()
        return ids

    def find_similar_performers(
        self,
        query_embedding: list[float],
        limit: int = 10,
        exclude_performer_ids: list[int] | None = None,
        min_similarity: float = 0.0,
    ) -> list[tuple[int, float]]:
        """
        Find performers most similar to query embedding for current model_key.

        Uses cosine similarity (assumes normalized embeddings).

        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            exclude_performer_ids: Performer IDs to exclude
            min_similarity: Minimum similarity threshold

        Returns:
            List of (performer_id, similarity) tuples sorted by similarity descending
        """
        exclude_set = set(exclude_performer_ids or [])

        query_arr = np.array(query_embedding, dtype=np.float32)
        query_norm: float = float(np.linalg.norm(query_arr))
        if query_norm > 0:
            query_arr = query_arr / query_norm

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT performer_id, embedding FROM performer_embeddings
            WHERE model_key = ?
        """,
            (self.model_key,),
        )

        similarities: list[tuple[int, float]] = []

        for row in cursor.fetchall():
            performer_id: int = row["performer_id"]
            if performer_id in exclude_set:
                continue

            stored_emb = self._unpack_embedding(row["embedding"])
            stored_arr = np.array(stored_emb, dtype=np.float32)

            similarity = float(np.dot(query_arr, stored_arr))

            if similarity >= min_similarity:
                similarities.append((performer_id, similarity))

        conn.close()

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:limit]

    def get_performer_stats(self) -> dict[str, Any]:
        """Get performer embedding statistics for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='performer_embeddings'"
        )
        if not cursor.fetchone():
            conn.close()
            return {
                "model_key": self.model_key,
                "total_performers": 0,
                "total_contributing_scenes": 0,
                "avg_scenes_per_performer": 0.0,
                "oldest_embedding": None,
                "newest_embedding": None,
            }

        cursor.execute(
            "SELECT COUNT(*) as count FROM performer_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        count = cursor.fetchone()["count"]

        cursor.execute(
            """
            SELECT SUM(contributing_scenes) as total_scenes,
                   AVG(contributing_scenes) as avg_scenes
            FROM performer_embeddings
            WHERE model_key = ?
        """,
            (self.model_key,),
        )
        scene_stats = cursor.fetchone()

        cursor.execute(
            """
            SELECT MIN(created_at) as oldest, MAX(updated_at) as newest
            FROM performer_embeddings
            WHERE model_key = ?
        """,
            (self.model_key,),
        )
        dates = cursor.fetchone()

        conn.close()

        return {
            "model_key": self.model_key,
            "total_performers": count,
            "total_contributing_scenes": scene_stats["total_scenes"] or 0,
            "avg_scenes_per_performer": scene_stats["avg_scenes"] or 0.0,
            "oldest_embedding": dates["oldest"],
            "newest_embedding": dates["newest"],
        }

    def clear_all_performer_embeddings(self) -> int:
        """Delete all performer embeddings for current model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='performer_embeddings'"
        )
        if not cursor.fetchone():
            conn.close()
            return 0

        cursor.execute(
            "DELETE FROM performer_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def update_performer_description(
        self,
        performer_id: int,
        visual_description: str,
        top_tags: str | None = None,
    ) -> bool:
        """
        Update the visual description and tags for a performer embedding.

        Args:
            performer_id: Stash performer ID
            visual_description: AI-generated description
            top_tags: Optional JSON array of top tags

        Returns:
            True if updated, False if performer embedding not found
        """
        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        if top_tags is not None:
            cursor.execute(
                """
                UPDATE performer_embeddings
                SET visual_description = ?, top_tags = ?, updated_at = ?
                WHERE performer_id = ? AND model_key = ?
            """,
                (visual_description, top_tags, now, performer_id, self.model_key),
            )
        else:
            cursor.execute(
                """
                UPDATE performer_embeddings
                SET visual_description = ?, updated_at = ?
                WHERE performer_id = ? AND model_key = ?
            """,
                (visual_description, now, performer_id, self.model_key),
            )

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    # ========================================================================
    # Taste Map Methods
    # ========================================================================

    def save_taste_clusters(self, clusters: list[Any], model_key: str) -> None:
        """Save taste clusters, replacing any existing for this model_key."""
        import json
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        # Clear existing clusters for this model
        cursor.execute("DELETE FROM taste_clusters WHERE model_key = ?", (model_key,))

        for cluster in clusters:
            tag_matches_json = json.dumps(cluster.tag_matches)
            scene_ids_json = json.dumps(cluster.scene_ids)
            cursor.execute(
                """INSERT INTO taste_clusters
                (cluster_id, model_key, centroid, scene_ids, engagement_total,
                 engagement_share, auto_label, user_label, weight_override,
                 excluded, tag_matches, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cluster.cluster_id,
                    model_key,
                    self._pack_embedding(cluster.centroid.tolist()),
                    scene_ids_json,
                    cluster.engagement_total,
                    cluster.engagement_share,
                    cluster.auto_label,
                    cluster.user_label,
                    cluster.weight_override,
                    1 if cluster.excluded else 0,
                    tag_matches_json,
                    now,
                ),
            )
        conn.commit()
        conn.close()

    def get_taste_clusters(self, model_key: str) -> list[dict[str, Any]]:
        """Load taste clusters for a model_key."""
        import json

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM taste_clusters WHERE model_key = ? ORDER BY cluster_id",
            (model_key,),
        )
        rows = cursor.fetchall()

        clusters = []
        for row in rows:
            clusters.append(
                {
                    "cluster_id": row["cluster_id"],
                    "model_key": row["model_key"],
                    "centroid": self._unpack_embedding(row["centroid"]),
                    "scene_ids": json.loads(row["scene_ids"]),
                    "engagement_total": row["engagement_total"],
                    "engagement_share": row["engagement_share"],
                    "auto_label": row["auto_label"],
                    "user_label": row["user_label"],
                    "weight_override": row["weight_override"],
                    "excluded": bool(row["excluded"]),
                    "tag_matches": json.loads(row["tag_matches"]),
                    "created_at": row["created_at"],
                }
            )

        conn.close()
        return clusters

    def update_taste_cluster(self, cluster_id: int, model_key: str, **kwargs: object) -> None:
        """Update specific fields of a taste cluster."""
        allowed = {"user_label", "weight_override", "excluded"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return

        if "excluded" in updates:
            updates["excluded"] = 1 if updates["excluded"] else 0

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [cluster_id, model_key]

        conn = self._get_connection()
        conn.execute(
            f"UPDATE taste_clusters SET {set_clause} WHERE cluster_id = ? AND model_key = ?",
            values,
        )
        conn.commit()
        conn.close()

    # --- UMAP Coordinate Methods ---

    def save_umap_coords(
        self,
        coords: dict[int, tuple[float, float, float]],
        cluster_assignments: dict[int, int],
        model_key: str,
    ) -> None:
        """Save UMAP 3D coordinates for scenes."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        # Clear existing coords for this model
        cursor.execute("DELETE FROM scene_umap_coords WHERE model_key = ?", (model_key,))

        for scene_id, (x, y, z) in coords.items():
            cluster_id = cluster_assignments.get(scene_id)
            cursor.execute(
                """INSERT INTO scene_umap_coords
                (scene_id, model_key, x, y, z, cluster_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (scene_id, model_key, x, y, z, cluster_id, now),
            )
        conn.commit()
        conn.close()

    def get_umap_coords(self, model_key: str) -> list[dict[str, int | float | None]]:
        """Load all UMAP 3D coordinates for a model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT scene_id, x, y, z, cluster_id FROM scene_umap_coords WHERE model_key = ?",
            (model_key,),
        )
        rows = cursor.fetchall()

        result: list[dict[str, int | float | None]] = [
            {
                "scene_id": r["scene_id"],
                "x": r["x"],
                "y": r["y"],
                "z": r["z"],
                "cluster_id": r["cluster_id"],
            }
            for r in rows
        ]

        conn.close()
        return result

    # --- Tag Embedding Methods ---

    def save_tag_embedding(
        self, text: str, model_key: str, embedding: list[float], source: str
    ) -> None:
        """Save a tag/phrase text embedding."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO tag_embeddings
            (text, model_key, embedding, source, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (text, model_key, self._pack_embedding(embedding), source, now),
        )
        conn.commit()
        conn.close()

    def save_tag_embeddings_batch(
        self,
        entries: list[tuple[str, list[float], str]],
        model_key: str,
    ) -> None:
        """Batch save tag embeddings. Each entry is (text, embedding, source)."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        for text, embedding, source in entries:
            conn.execute(
                """INSERT OR REPLACE INTO tag_embeddings
                (text, model_key, embedding, source, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (text, model_key, self._pack_embedding(embedding), source, now),
            )
        conn.commit()
        conn.close()

    def get_all_tag_embeddings(self, model_key: str) -> list[dict[str, Any]]:
        """Load all tag embeddings for a model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT text, embedding, source FROM tag_embeddings WHERE model_key = ?",
            (model_key,),
        )
        rows = cursor.fetchall()

        result = [
            {
                "text": r["text"],
                "embedding": self._unpack_embedding(r["embedding"]),
                "source": r["source"],
            }
            for r in rows
        ]

        conn.close()
        return result

    def get_tag_embedding(self, text: str, model_key: str) -> list[float] | None:
        """Get embedding for a specific tag/phrase."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT embedding FROM tag_embeddings WHERE text = ? AND model_key = ?",
            (text, model_key),
        )
        row = cursor.fetchone()

        conn.close()
        if row:
            return self._unpack_embedding(row["embedding"])
        return None

    def get_tag_embedding_count(self, model_key: str) -> int:
        """Count how many tag embeddings exist for a model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) as cnt FROM tag_embeddings WHERE model_key = ?",
            (model_key,),
        )
        row = cursor.fetchone()

        conn.close()
        return row["cnt"] if row else 0

    def get_stash_tag_names(self, model_key: str) -> set[str]:
        """Get set of tag names that are actual Stash tags (not curated phrases).

        Args:
            model_key: The embedding model key.

        Returns:
            Set of lowercase tag names with source='stash_tag'.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT text FROM tag_embeddings WHERE model_key = ? AND source = 'stash_tag'",
            (model_key,),
        )
        rows = cursor.fetchall()

        conn.close()
        return {r["text"].lower() for r in rows}

    # ── Frame Tag Coverage Methods ─────────────────────────────────────

    def save_frame_tag_coverage_batch(self, rows: list[FrameTagCoverageRecord]) -> None:
        """Batch insert or replace frame tag coverage records.

        Args:
            rows: List of FrameTagCoverageRecord dicts to upsert.
        """
        if not rows:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.executemany(
            """
            INSERT OR REPLACE INTO frame_tag_coverage (
                scene_id, frame_index, model_key,
                best_tag, best_similarity, is_covered
            ) VALUES (
                :scene_id, :frame_index, :model_key,
                :best_tag, :best_similarity, :is_covered
            )
        """,
            rows,
        )

        conn.commit()
        conn.close()

    def get_scene_tag_coverage(self, scene_id: int) -> list[FrameTagCoverageRecord]:
        """Get all tag coverage rows for a scene, ordered by frame index.

        Args:
            scene_id: The scene to query.

        Returns:
            List of FrameTagCoverageRecord dicts ordered by frame_index.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT scene_id, frame_index, model_key,
                   best_tag, best_similarity, is_covered
            FROM frame_tag_coverage
            WHERE scene_id = ? AND model_key = ?
            ORDER BY frame_index
        """,
            (scene_id, self.model_key),
        )
        rows = cursor.fetchall()

        conn.close()
        return [
            FrameTagCoverageRecord(
                scene_id=r["scene_id"],
                frame_index=r["frame_index"],
                model_key=r["model_key"],
                best_tag=r["best_tag"],
                best_similarity=r["best_similarity"],
                is_covered=bool(r["is_covered"]),
            )
            for r in rows
        ]

    def get_coverage_summary(self) -> list[dict[str, Any]]:
        """Get per-scene coverage summary, ordered by most uncovered first.

        Returns:
            List of dicts with keys: scene_id, total_frames,
            uncovered_frames, coverage_ratio. Ordered by coverage_ratio
            ascending (most uncovered scenes first).
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                scene_id,
                COUNT(*) AS total_frames,
                SUM(CASE WHEN NOT is_covered THEN 1 ELSE 0 END)
                    AS uncovered_frames,
                1.0 - (
                    CAST(
                        SUM(CASE WHEN NOT is_covered THEN 1 ELSE 0 END)
                        AS REAL
                    ) / COUNT(*)
                ) AS coverage_ratio
            FROM frame_tag_coverage
            WHERE model_key = ?
            GROUP BY scene_id
            ORDER BY coverage_ratio ASC
        """,
            (self.model_key,),
        )
        rows = cursor.fetchall()

        conn.close()
        return [
            {
                "scene_id": r["scene_id"],
                "total_frames": r["total_frames"],
                "uncovered_frames": r["uncovered_frames"],
                "coverage_ratio": r["coverage_ratio"],
            }
            for r in rows
        ]

    def get_uncovered_frame_embeddings(self, scene_id: int) -> NDArray[np.float32]:
        """Get embeddings for frames that are not covered by any tag.

        Joins frame_tag_coverage with frame_embeddings to retrieve the
        actual embedding vectors for uncovered frames.

        Args:
            scene_id: The scene to query.

        Returns:
            Numpy array of shape (N, dims) with uncovered frame embeddings.
            Returns an empty (0,) array if no uncovered frames exist.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT fe.embedding
            FROM frame_tag_coverage ftc
            JOIN frame_embeddings fe
                ON ftc.scene_id = fe.scene_id
                AND ftc.frame_index = fe.frame_index
                AND ftc.model_key = fe.model_key
            WHERE ftc.scene_id = ?
                AND ftc.model_key = ?
                AND NOT ftc.is_covered
            ORDER BY ftc.frame_index
        """,
            (scene_id, self.model_key),
        )
        rows = cursor.fetchall()

        conn.close()

        if not rows:
            return np.array([], dtype=np.float32)

        embeddings = [self._unpack_embedding(r["embedding"]) for r in rows]
        return np.array(embeddings, dtype=np.float32)

    def get_scene_frame_embeddings(self, scene_id: int) -> NDArray[np.float32]:
        """Get all frame embeddings for a scene.

        Args:
            scene_id: The scene to query.

        Returns:
            Numpy array of shape (N, dims) with all frame embeddings.
            Returns an empty (0,) array if no frames exist.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT embedding
            FROM frame_embeddings
            WHERE scene_id = ? AND model_key = ?
            ORDER BY frame_index
        """,
            (scene_id, self.model_key),
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return np.array([], dtype=np.float32)

        embeddings = [self._unpack_embedding(r["embedding"]) for r in rows]
        return np.array(embeddings, dtype=np.float32)

    def get_scenes_with_tag_coverage(self) -> list[int]:
        """Get all scene IDs that have tag coverage data.

        Returns:
            List of distinct scene IDs with coverage records
            for the current model_key.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT scene_id
            FROM frame_tag_coverage
            WHERE model_key = ?
        """,
            (self.model_key,),
        )
        rows = cursor.fetchall()

        conn.close()
        return [r["scene_id"] for r in rows]

    def delete_scene_tag_coverage(self, scene_id: int) -> None:
        """Delete all tag coverage records for a scene.

        Args:
            scene_id: The scene whose coverage data should be removed.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM frame_tag_coverage
            WHERE scene_id = ? AND model_key = ?
        """,
            (scene_id, self.model_key),
        )

        conn.commit()
        conn.close()

    def update_coverage_threshold(self, threshold: float) -> int:
        """Re-evaluate coverage using a new similarity threshold.

        Updates is_covered for all rows: frames with best_similarity
        >= threshold are marked as covered, others as uncovered.

        Args:
            threshold: The new similarity threshold.

        Returns:
            Number of rows updated.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE frame_tag_coverage
            SET is_covered = CASE
                WHEN best_similarity >= ? THEN 1
                ELSE 0
            END
            WHERE model_key = ?
        """,
            (threshold, self.model_key),
        )
        rowcount = cursor.rowcount

        conn.commit()
        conn.close()
        return rowcount

    def save_dismissed_tag(self, scene_id: int, tag_id: int) -> None:
        """Record that a tag suggestion was dismissed for a scene."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO dismissed_tag_suggestions
            (scene_id, tag_id, dismissed_at)
            VALUES (?, ?, ?)
            """,
            (scene_id, tag_id, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def get_dismissed_tags(self, scene_id: int) -> set[int]:
        """Get all dismissed tag IDs for a scene."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT tag_id FROM dismissed_tag_suggestions
            WHERE scene_id = ?
            """,
            (scene_id,),
        )
        result = {row["tag_id"] for row in cursor.fetchall()}
        conn.close()
        return result

    def clear_dismissed_tags(self, scene_id: int) -> int:
        """Clear all dismissals for a scene. Returns count deleted."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM dismissed_tag_suggestions
            WHERE scene_id = ?
            """,
            (scene_id,),
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count

    # ── Tag Merge Dismissal Methods ───────────────────────────────────

    def save_dismissed_tag_merge(self, tag_a_name: str, tag_b_name: str) -> None:
        """Record that a tag merge pair was dismissed."""
        from datetime import timezone

        # Normalize order so (A,B) and (B,A) are the same dismissal
        names = sorted([tag_a_name.lower(), tag_b_name.lower()])
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO dismissed_tag_merges
            (tag_a_name, tag_b_name, dismissed_at)
            VALUES (?, ?, ?)
            """,
            (names[0], names[1], datetime.now(tz=timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    def get_dismissed_tag_merges(self) -> set[tuple[str, str]]:
        """Get all dismissed tag merge pairs as a set of (name_a, name_b) tuples."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT tag_a_name, tag_b_name FROM dismissed_tag_merges")
        result = {(row["tag_a_name"], row["tag_b_name"]) for row in cursor.fetchall()}
        conn.close()
        return result

    def delete_tag_embedding(self, text: str, model_key: str) -> None:
        """Delete a tag embedding by text and model_key."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM tag_embeddings WHERE text = ? AND model_key = ?",
            (text.lower(), model_key),
        )
        conn.commit()
        conn.close()

    # ── Labeling Session & Annotation Methods ─────────────────────────

    def create_labeling_session(
        self,
        sampling_method: str,
        batch_size: int,
        total_frames: int,
        config_json: str | None = None,
    ) -> str:
        """Create a new labeling session.

        Args:
            sampling_method: How frames were sampled (e.g., 'random', 'stratified').
            batch_size: Number of frames per batch.
            total_frames: Total frames available for labeling.
            config_json: Optional JSON string with additional configuration.

        Returns:
            The session_id (UUID string).
        """
        from datetime import timezone

        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO labeling_sessions
            (session_id, created_at, updated_at, status, sampling_method,
             batch_size, total_frames, labeled_count, skipped_count, config_json)
            VALUES (?, ?, ?, 'active', ?, ?, ?, 0, 0, ?)
            """,
            (session_id, now, now, sampling_method, batch_size, total_frames, config_json),
        )
        conn.commit()
        conn.close()
        return session_id

    def get_labeling_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a labeling session by ID.

        Args:
            session_id: The session UUID.

        Returns:
            Session dict or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM labeling_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        return dict(row)

    def update_labeling_session(self, session_id: str, **kwargs: Any) -> None:
        """Update allowed fields on a labeling session.

        Allowed fields: status, labeled_count, skipped_count, config_json.
        Also updates updated_at automatically.

        Args:
            session_id: The session UUID.
            **kwargs: Field=value pairs to update.
        """
        from datetime import timezone

        allowed = {"status", "labeled_count", "skipped_count", "config_json"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return

        updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        values.append(session_id)

        conn = self._get_connection()
        conn.execute(
            f"UPDATE labeling_sessions SET {set_clause} WHERE session_id = ?",
            values,
        )
        conn.commit()
        conn.close()

    def list_labeling_sessions(self, status: str | None = None) -> list[dict[str, Any]]:
        """List labeling sessions, optionally filtered by status.

        Args:
            status: If provided, filter to sessions with this status.

        Returns:
            List of session dicts ordered by created_at descending.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if status is not None:
            cursor.execute(
                "SELECT * FROM labeling_sessions WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        else:
            cursor.execute(
                "SELECT * FROM labeling_sessions ORDER BY created_at DESC"
            )

        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def save_annotations(self, session_id: str, annotations: list[dict[str, Any]]) -> None:
        """Bulk save frame annotations for a session.

        Each annotation dict should contain: scene_id, frame_index, tag_text,
        tag_source, label, and optionally similarity_score.

        Args:
            session_id: The session UUID.
            annotations: List of annotation dicts.
        """
        from datetime import timezone

        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        for ann in annotations:
            conn.execute(
                """
                INSERT OR REPLACE INTO frame_annotations
                (session_id, scene_id, frame_index, tag_text, tag_source,
                 label, similarity_score, labeled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    ann["scene_id"],
                    ann["frame_index"],
                    ann["tag_text"],
                    ann["tag_source"],
                    ann["label"],
                    ann.get("similarity_score"),
                    now,
                ),
            )
        conn.commit()
        conn.close()

    def get_annotations(self, session_id: str) -> list[dict[str, Any]]:
        """Get all annotations for a session.

        Args:
            session_id: The session UUID.

        Returns:
            List of annotation dicts.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM frame_annotations WHERE session_id = ?",
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_all_confirmed_annotations(self) -> list[dict[str, Any]]:
        """Get all confirmed annotations across all sessions.

        Returns:
            List of annotation dicts where label='confirmed'.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM frame_annotations WHERE label = 'confirmed'"
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def update_labeling_progress(
        self, session_id: str, scene_id: int, frame_index: int, status: str
    ) -> None:
        """Insert or update labeling progress for a frame.

        Args:
            session_id: The session UUID.
            scene_id: The scene ID.
            frame_index: The frame index within the scene.
            status: Progress status (e.g., 'pending', 'labeled', 'skipped').
        """
        conn = self._get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO labeling_progress
            (scene_id, frame_index, session_id, status)
            VALUES (?, ?, ?, ?)
            """,
            (scene_id, frame_index, session_id, status),
        )
        conn.commit()
        conn.close()

    def get_labeled_frame_keys(self) -> set[tuple[int, int]]:
        """Get all (scene_id, frame_index) pairs with status='labeled'.

        Returns:
            Set of (scene_id, frame_index) tuples.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT scene_id, frame_index FROM labeling_progress WHERE status = 'labeled'"
        )
        rows = cursor.fetchall()
        conn.close()
        return {(row["scene_id"], row["frame_index"]) for row in rows}

    def get_unembedded_manual_tags(self, model_key: str) -> list[str]:
        """Get manual tag texts that don't have embeddings yet.

        Finds confirmed annotations with tag_source='manual' whose tag_text
        does not appear in tag_embeddings for the given model_key.

        Args:
            model_key: The embedding model key to check against.

        Returns:
            List of tag text strings without embeddings.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT fa.tag_text
            FROM frame_annotations fa
            WHERE fa.tag_source = 'manual'
              AND fa.label = 'confirmed'
              AND fa.tag_text NOT IN (
                  SELECT te.text FROM tag_embeddings te WHERE te.model_key = ?
              )
            """,
            (model_key,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [row["tag_text"] for row in rows]
