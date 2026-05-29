"""FAISS-based frame-level semantic search index."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from stash_ai.embeddings.storage import EmbeddingStorage


@dataclass
class FrameMetadata:
    """Metadata for a single frame in the index."""

    scene_id: int
    frame_index: int
    timestamp: float


@dataclass
class FrameMatch:
    """A single frame match from FAISS search."""

    scene_id: int
    frame_index: int
    timestamp: float
    similarity: float


@dataclass
class SceneMatch:
    """Aggregated scene match (best frame per scene)."""

    scene_id: int
    best_frame_index: int
    best_timestamp: float
    similarity: float


@dataclass
class IndexInfo:
    """Metadata about a built index."""

    model_key: str
    frame_count: int
    scene_count: int
    dimensions: int
    created_at: str


class FrameSearchIndex:
    """FAISS-based frame-level search index.

    Builds and queries a FAISS index over all frame embeddings,
    enabling fast similarity search across millions of frames.

    Uses memory-mapped loading and binary metadata for fast startup.
    """

    def __init__(self, assets_dir: str, model_key: str = "siglip"):
        """Initialize the frame search index.

        Args:
            assets_dir: Path to assets directory for index storage
            model_key: Embedding model key (e.g., "siglip", "openclip:ViT-H-14")
        """
        self.assets_dir = Path(assets_dir)
        self.model_key = model_key

        # Sanitize model_key for filename (replace : with -)
        safe_key = model_key.replace(":", "-").replace("/", "-")
        self.index_path = self.assets_dir / f"frame_search_{safe_key}.index"
        self.meta_path = self.assets_dir / f"frame_search_{safe_key}_meta.npz"
        self.info_path = self.assets_dir / f"frame_search_{safe_key}_info.json"
        # Legacy JSON path for migration
        self._legacy_meta_path = self.assets_dir / f"frame_search_{safe_key}_meta.json"

        # Lazy-loaded index and metadata arrays
        self._index: faiss.IndexFlatIP | None = None
        self._scene_ids: NDArray[np.int64] | None = None
        self._frame_indices: NDArray[np.int32] | None = None
        self._timestamps: NDArray[np.float32] | None = None
        self._info: IndexInfo | None = None

    @property
    def exists(self) -> bool:
        """Check if index files exist on disk."""
        # Support both new binary format and legacy JSON format
        has_meta = self.meta_path.exists() or self._legacy_meta_path.exists()
        return self.index_path.exists() and has_meta

    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded in memory."""
        return self._index is not None

    def build(
        self,
        storage: "EmbeddingStorage",
        batch_size: int = 10000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexInfo:
        """Build FAISS index from all frame embeddings.

        Args:
            storage: EmbeddingStorage instance to read frames from
            batch_size: Number of frames to process at a time
            progress_callback: Optional callback(current, total) for progress

        Returns:
            IndexInfo with build statistics
        """

        # Ensure assets directory exists
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Get all frame embeddings for this model
        conn = storage._get_connection()
        cursor = conn.cursor()

        # First, count total frames
        cursor.execute(
            "SELECT COUNT(*) FROM frame_embeddings WHERE model_key = ?",
            (self.model_key,),
        )
        total_frames = cursor.fetchone()[0]

        if total_frames == 0:
            conn.close()
            raise ValueError(
                f"No frame embeddings found for model '{self.model_key}'. "
                "Run 'Embed All Scenes' task first."
            )

        # Get dimensions from first embedding
        cursor.execute(
            "SELECT embedding FROM frame_embeddings WHERE model_key = ? LIMIT 1",
            (self.model_key,),
        )
        first_row = cursor.fetchone()
        first_embedding = storage._unpack_embedding(first_row[0])
        dimensions = len(first_embedding)

        # Create FAISS index (Inner Product = cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dimensions)

        # Pre-allocate metadata arrays for efficiency
        scene_ids = np.zeros(total_frames, dtype=np.int64)
        frame_indices = np.zeros(total_frames, dtype=np.int32)
        timestamps = np.zeros(total_frames, dtype=np.float32)
        unique_scene_ids: set[int] = set()

        # Process in batches
        cursor.execute(
            """
            SELECT scene_id, frame_index, timestamp, embedding
            FROM frame_embeddings
            WHERE model_key = ?
            ORDER BY scene_id, frame_index
            """,
            (self.model_key,),
        )

        batch_embeddings = []
        processed = 0

        for row in cursor:
            scene_id, frame_index, timestamp, embedding_blob = row
            embedding = storage._unpack_embedding(embedding_blob)

            batch_embeddings.append(embedding)
            scene_ids[processed] = scene_id
            frame_indices[processed] = frame_index
            timestamps[processed] = timestamp
            unique_scene_ids.add(scene_id)
            processed += 1

            # Add batch to index when full
            if len(batch_embeddings) >= batch_size:
                vectors = np.array(batch_embeddings, dtype=np.float32)
                index.add(vectors)
                batch_embeddings = []

                if progress_callback:
                    progress_callback(processed, total_frames)

        # Add remaining embeddings
        if batch_embeddings:
            vectors = np.array(batch_embeddings, dtype=np.float32)
            index.add(vectors)

        conn.close()

        if progress_callback:
            progress_callback(total_frames, total_frames)

        # Build info
        info = IndexInfo(
            model_key=self.model_key,
            frame_count=total_frames,
            scene_count=len(unique_scene_ids),
            dimensions=dimensions,
            created_at=datetime.utcnow().isoformat() + "Z",
        )

        # Save index to disk
        faiss.write_index(index, str(self.index_path))

        # Save metadata as binary NumPy arrays (MUCH faster than JSON)
        np.savez(
            self.meta_path,
            scene_ids=scene_ids,
            frame_indices=frame_indices,
            timestamps=timestamps,
        )

        # Save lightweight info as JSON (tiny file)
        info_dict = {
            "model_key": info.model_key,
            "frame_count": info.frame_count,
            "scene_count": info.scene_count,
            "dimensions": info.dimensions,
            "created_at": info.created_at,
        }
        with open(self.info_path, "w") as f:
            json.dump(info_dict, f)

        # Remove legacy JSON metadata if it exists
        if self._legacy_meta_path.exists():
            self._legacy_meta_path.unlink()

        # Save raw vectors as numpy file for fast preference scoring.
        # This avoids the FAISS mmap overhead (~35s) by using numpy's
        # simpler format (instant mmap open, ~0.5s scoring).
        vectors_npy_path = (
            self.assets_dir
            / f"frame_vectors_{self.model_key.replace(':', '-').replace('/', '-')}.npy"
        )
        vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(
            index.ntotal, index.d
        )
        np.save(vectors_npy_path, np.array(vectors, dtype=np.float32))

        # Update instance state
        self._index = index
        self._scene_ids = scene_ids
        self._frame_indices = frame_indices
        self._timestamps = timestamps
        self._info = info

        return info

    def load(self, mmap: bool = True) -> IndexInfo:
        """Load index from disk into memory.

        Args:
            mmap: Use memory-mapped loading for FAISS index (faster startup,
                  lower memory usage, but slightly slower queries). Default True.

        Returns:
            IndexInfo with index statistics

        Raises:
            FileNotFoundError: If index files don't exist
        """
        if not self.exists:
            raise FileNotFoundError(
                f"Frame search index not found for model '{self.model_key}'. "
                "Run 'Build Frame Search Index' task first."
            )

        # Load FAISS index with memory mapping for fast startup
        if mmap:
            self._index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)
        else:
            self._index = faiss.read_index(str(self.index_path))

        # Load metadata - try binary format first, fall back to legacy JSON
        if self.meta_path.exists():
            # Fast binary loading
            data = np.load(self.meta_path)
            self._scene_ids = data["scene_ids"]
            self._frame_indices = data["frame_indices"]
            self._timestamps = data["timestamps"]
        elif self._legacy_meta_path.exists():
            # Legacy JSON loading (slower, but maintains compatibility)
            with open(self._legacy_meta_path) as f:
                meta_dict = json.load(f)

            frames = meta_dict["frames"]
            self._scene_ids = np.array([m["scene_id"] for m in frames], dtype=np.int64)
            self._frame_indices = np.array([m["frame_index"] for m in frames], dtype=np.int32)
            self._timestamps = np.array([m["timestamp"] for m in frames], dtype=np.float32)
        else:
            raise FileNotFoundError(f"Metadata file not found for model '{self.model_key}'")

        # Load info
        self._info = self._load_info()
        if self._info is None:
            # Reconstruct from loaded data
            self._info = IndexInfo(
                model_key=self.model_key,
                frame_count=len(self._scene_ids),
                scene_count=len(np.unique(self._scene_ids)),
                dimensions=self._index.d,
                created_at="unknown",
            )

        return self._info

    def _load_info(self) -> IndexInfo | None:
        """Load just the index info (lightweight)."""
        # Try new info file first
        if self.info_path.exists():
            with open(self.info_path) as f:
                meta_dict = json.load(f)
            return IndexInfo(
                model_key=meta_dict["model_key"],
                frame_count=meta_dict["frame_count"],
                scene_count=meta_dict["scene_count"],
                dimensions=meta_dict["dimensions"],
                created_at=meta_dict["created_at"],
            )

        # Fall back to legacy JSON (but only parse the header, not frames)
        if self._legacy_meta_path.exists():
            with open(self._legacy_meta_path) as f:
                # Read just enough to get the header fields
                content = f.read(4096)  # First 4KB should have header
                # Find where "frames" starts and truncate
                frames_pos = content.find('"frames"')
                if frames_pos > 0:
                    content = content[:frames_pos] + '"frames": []}'
                try:
                    meta_dict = json.loads(content)
                    return IndexInfo(
                        model_key=meta_dict.get("model_key", self.model_key),
                        frame_count=meta_dict.get("frame_count", 0),
                        scene_count=meta_dict.get("scene_count", 0),
                        dimensions=meta_dict.get("dimensions", 0),
                        created_at=meta_dict.get("created_at", "unknown"),
                    )
                except json.JSONDecodeError:
                    pass

        return None

    def ensure_loaded(self) -> IndexInfo:
        """Ensure index is loaded, loading from disk if needed."""
        if not self.is_loaded:
            return self.load()
        return self._info  # type: ignore

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 1000,
    ) -> list[FrameMatch]:
        """Search for similar frames.

        Args:
            query_embedding: Query vector (must match index dimensions)
            top_k: Number of top matches to return

        Returns:
            List of FrameMatch sorted by similarity (descending)
        """
        self.ensure_loaded()
        # Assert arrays are loaded (ensure_loaded guarantees this)
        assert self._index is not None
        assert self._scene_ids is not None
        assert self._frame_indices is not None
        assert self._timestamps is not None

        # Reshape query to 2D array for FAISS
        query = np.array([query_embedding], dtype=np.float32)

        # Search (D = distances/similarities, I = indices)
        similarities, indices = self._index.search(query, top_k)

        # Build results using vectorized access
        results: list[FrameMatch] = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for not enough results
                break

            results.append(
                FrameMatch(
                    scene_id=int(self._scene_ids[idx]),
                    frame_index=int(self._frame_indices[idx]),
                    timestamp=float(self._timestamps[idx]),
                    similarity=float(sim),
                )
            )

        return results

    def aggregate_to_scenes(
        self,
        frame_matches: list[FrameMatch],
    ) -> list[SceneMatch]:
        """Aggregate frame matches to scene-level using max-score.

        For each scene, keeps only the best-matching frame.

        Args:
            frame_matches: List of frame matches from search()

        Returns:
            List of SceneMatch sorted by similarity (descending)
        """
        # Group by scene, keep max
        scene_best: dict[int, FrameMatch] = {}

        for match in frame_matches:
            if (
                match.scene_id not in scene_best
                or match.similarity > scene_best[match.scene_id].similarity
            ):
                scene_best[match.scene_id] = match

        # Convert to SceneMatch and sort
        results = [
            SceneMatch(
                scene_id=m.scene_id,
                best_frame_index=m.frame_index,
                best_timestamp=m.timestamp,
                similarity=m.similarity,
            )
            for m in scene_best.values()
        ]

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results

    def get_info(self) -> IndexInfo | None:
        """Get index info without loading the full index.

        Returns:
            IndexInfo if metadata file exists, None otherwise
        """
        if self._info is not None:
            return self._info

        return self._load_info()
