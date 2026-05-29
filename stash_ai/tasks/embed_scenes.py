"""Task for generating scene embeddings."""

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from ..config import LLMConfig
from ..embeddings.base import BaseImageEmbeddingProvider
from ..embeddings.config import EmbeddingConfig
from ..embeddings.provider import get_embedding_provider
from ..embeddings.storage import EmbeddingStorage
from ..llm import get_provider
from ..prompts.loader import get_prompt
from ..tools.database import get_readonly_connection, get_stash_db_path
from .frame_extractor import (
    DEFAULT_FPS_RATE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAMES,
    FrameExtractionConfig,
    FrameExtractor,
)

if TYPE_CHECKING:
    from ..stash_client import StashClient


@dataclass
class EmbedConfig:
    """Configuration for embedding generation."""

    # Visual embedding weight (0-1), metadata gets 1-visual_weight
    visual_weight: float = 1.0

    # Reuse existing vision descriptions if available
    use_cached_descriptions: bool = True

    # Frame extraction settings (only used if generating new descriptions)
    # Typically uses cached frames from vision analysis
    fps_rate: float = DEFAULT_FPS_RATE
    min_frames: int = DEFAULT_MIN_FRAMES
    max_frames: int = DEFAULT_MAX_FRAMES

    # Metadata components to include
    include_tags: bool = True
    include_performers: bool = True
    include_studio: bool = True
    include_title: bool = False  # Titles are often auto-generated

    # Direct image embedding (CLIP/OpenCLIP/SigLIP)
    # If True and embedder supports images, embeds frames directly
    # If False, uses VLM → text description → text embedding
    use_direct_image_embedding: bool = True

    # Representative frame selection
    # If True, uses k-means clustering to select diverse frames before averaging
    # This reduces redundancy from similar consecutive frames
    use_representative_selection: bool = False
    representative_n_frames: int = 8  # Number of representative frames to select
    representative_method: str = "kmeans"  # kmeans, maximin, or coverage

    # Parallel processing
    # Number of worker threads for parallel embedding (0 = auto based on CPU count)
    # Frame extraction uses single-pass FFmpeg, GPU embedding is sequential (locked)
    # Recommended: 2-4 workers for GPU systems, higher for CPU-only
    num_workers: int = 2


# Fallback prompt (used if YAML file not found)
VISUAL_DESCRIPTION_PROMPT = """Describe this video scene in detail. Focus on:
- Visual appearance and physical characteristics of people
- Actions and activities taking place
- Setting, environment, and atmosphere
- Camera angles and shot composition
- Mood and tone

Be descriptive and factual. Output 1-2 paragraphs."""


def _get_visual_description_prompt() -> str:
    """Load visual description prompt from YAML with fallback to hardcoded."""
    try:
        return get_prompt("embed", "visual_description", "visual_description")
    except (FileNotFoundError, KeyError):
        return VISUAL_DESCRIPTION_PROMPT


class EmbedScenesTask:
    """
    Task for generating hybrid embeddings for scenes.

    Workflow:
    1. Extract frames from video (or use cached)
    2. Generate visual description using VLM (or use cached from vision analysis)
    3. Collect metadata (tags, performers, studio)
    4. Generate text embeddings for both visual and metadata
    5. Combine into composite embedding with configurable weighting
    6. Store in local SQLite database
    """

    def __init__(
        self,
        stash: "StashClient",
        vlm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        image_embedding_config: EmbeddingConfig | None = None,
        embed_config: EmbedConfig | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Initialize the embedding task.

        Args:
            stash: StashClient instance
            vlm_config: Config for vision model (description generation)
            embedding_config: Config for text embedding model (metadata)
            image_embedding_config: Optional config for image embedding (CLIP/OpenCLIP/SigLIP)
            embed_config: Optional embedding-specific configuration
            log_callback: Optional logging callback
            progress_callback: Optional progress callback
        """
        self.stash = stash
        self.vlm_config = vlm_config
        self.embedding_config = embedding_config
        self.image_embedding_config = image_embedding_config
        self.config = embed_config or EmbedConfig()
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

        # Initialize providers lazily
        self._vlm: Any | None = None
        self._embedder: Any | None = None
        self._image_embedder: Any | None = None

        # Lock for GPU operations (embedder calls) to prevent resource contention
        self._gpu_lock = Lock()

        # Initialize storage with model_key from image embedding config
        # This allows multiple embedding models to coexist without overwriting
        model_key = image_embedding_config.model_key if image_embedding_config else "siglip"
        self.storage = EmbeddingStorage(model_key=model_key)
        self.log(f"Using embedding model key: {model_key}", "debug")

        # Setup frame extractor
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cache_dir = os.path.join(plugin_dir, "assets", "embedded_frames")
        self.frame_extractor = FrameExtractor(
            config=FrameExtractionConfig(
                fps_rate=self.config.fps_rate,
                min_frames=self.config.min_frames,
                max_frames=self.config.max_frames,
            ),
            cache_dir=cache_dir,
            log_callback=self.log,
        )

    @property
    def vlm(self) -> Any:
        """Lazy-load VLM provider."""
        if self._vlm is None:
            self._vlm = get_provider(self.vlm_config)
        return self._vlm

    @property
    def embedder(self) -> Any:
        """Lazy-load text embedding provider (for metadata)."""
        if self._embedder is None:
            self._embedder = get_embedding_provider(self.embedding_config)
        return self._embedder

    @property
    def image_embedder(self) -> Any | None:
        """Lazy-load image embedding provider (CLIP/OpenCLIP/SigLIP)."""
        if self._image_embedder is None and self.image_embedding_config is not None:
            self._image_embedder = get_embedding_provider(self.image_embedding_config)
        return self._image_embedder

    def cleanup(self) -> None:
        """Release GPU resources held by this task.

        Called automatically when the task completes or is interrupted.
        Safe to call multiple times.
        """
        # Release lazy-loaded providers
        if self._image_embedder is not None:
            if hasattr(self._image_embedder, "cleanup"):
                try:
                    self._image_embedder.cleanup()
                except Exception:
                    pass
            self._image_embedder = None

        if self._embedder is not None:
            if hasattr(self._embedder, "cleanup"):
                try:
                    self._embedder.cleanup()
                except Exception:
                    pass
            self._embedder = None

        if self._vlm is not None:
            if hasattr(self._vlm, "cleanup"):
                try:
                    self._vlm.cleanup()
                except Exception:
                    pass
            self._vlm = None

    def __del__(self) -> None:
        """Destructor - ensure resources are freed."""
        self.cleanup()

    def embed_scene(
        self,
        scene_id: int,
        force: bool = False,
        success_tag: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate and store embedding for a single scene.

        Args:
            scene_id: Stash scene ID
            force: If True, regenerate even if embedding exists
            success_tag: Optional tag name to apply after successful embedding

        Returns:
            Dict with success status and embedding info
        """
        # Check if already embedded
        if not force and self.storage.has_embedding(scene_id):
            self.log(f"Scene {scene_id} already has embedding, skipping", "debug")
            return {"success": True, "scene_id": scene_id, "cached": True}

        # Get scene info from database (safe to do in parallel)
        scene_info = self._get_scene_info(scene_id)
        if not scene_info:
            return {"success": False, "error": f"Scene {scene_id} not found"}

        # Check if we MIGHT use direct image embedding based on config
        # (actual embedder initialization happens inside the GPU lock)
        might_use_direct = (
            self.config.use_direct_image_embedding and self.image_embedding_config is not None
        )

        # Pre-extract frames outside the GPU lock (I/O bound, can run in parallel)
        # We extract frames if we might use direct embedding
        frame_paths: list[str] | None = None
        if might_use_direct:
            frame_paths = self._prepare_frames_for_embedding(scene_id, scene_info)

        # Build metadata text outside the lock (CPU only)
        metadata_text = self._build_metadata_text(scene_info)

        # === ALL GPU OPERATIONS HAPPEN INSIDE THIS LOCK ===
        # This prevents CUDA context corruption from concurrent GPU access
        with self._gpu_lock:
            # Now safely check if embedder supports images (inside lock)
            use_direct = (
                might_use_direct
                and self.image_embedder is not None
                and self.image_embedder.supports_images
            )

            return self._embed_scene_gpu_locked(
                scene_id=scene_id,
                scene_info=scene_info,
                use_direct=use_direct,
                frame_paths=frame_paths,
                metadata_text=metadata_text,
                success_tag=success_tag,
            )

    def _prepare_frames_for_embedding(
        self,
        scene_id: int,
        scene_info: dict[str, Any],
    ) -> list[str] | None:
        """Extract and prepare frames for embedding (I/O bound, no GPU)."""
        video_path = scene_info.get("video_path")
        duration = scene_info.get("duration")

        if not video_path or not duration:
            return None

        try:
            # Extract frames (or use cached) - this is I/O bound
            frames = self.frame_extractor.get_or_extract_frames(
                str(scene_id),
                video_path,
                duration,
            )

            if not frames:
                return None

            # Get actual file paths for the extracted frames
            return self.frame_extractor.get_frame_paths(str(scene_id))

        except Exception as e:
            self.log(f"Failed to prepare frames for scene {scene_id}: {e}", "warning")
            return None

    def _embed_scene_gpu_locked(
        self,
        scene_id: int,
        scene_info: dict[str, Any],
        use_direct: bool,
        frame_paths: list[str] | None,
        metadata_text: str | None,
        success_tag: str | None,
    ) -> dict[str, Any]:
        """
        Perform all GPU operations for embedding a scene.

        This method must be called while holding the GPU lock.
        """
        visual_embedding: list[float] | None = None
        visual_description: str | None = None
        visual_model: str | None = None

        # Stage 1: Get visual embedding
        if use_direct:
            # Direct image embedding (CLIP/OpenCLIP/SigLIP)
            image_model = (
                self.image_embedding_config.model if self.image_embedding_config else "unknown"
            )
            self.log(f"Scene {scene_id}: Using IMAGE embedding ({image_model})", "info")

            if frame_paths and self.image_embedder is not None:
                try:
                    self.log(f"Embedding {len(frame_paths)} frames for scene {scene_id}", "debug")
                    results = self.image_embedder.embed_images(cast("list[Any]", frame_paths))

                    if results:
                        embeddings = [r["embedding"] for r in results]

                        # Store individual frame embeddings for smart selection
                        self._store_frame_embeddings(
                            scene_id=scene_id,
                            frame_paths=frame_paths,
                            embeddings=embeddings,
                            duration=scene_info.get("duration", 0.0),
                        )

                        visual_embedding = self._average_embeddings(embeddings)
                        visual_model = image_model
                except Exception as e:
                    self.log(f"Failed to embed frames: {e}", "warning")
        else:
            # VLM description → text embedding
            self.log(
                f"Scene {scene_id}: Using TEXT embedding (VLM: {self.vlm_config.model})", "info"
            )
            visual_description = self._get_visual_description(scene_id, scene_info)
            if visual_description:
                try:
                    visual_result = self.embedder.embed_text(visual_description)
                    visual_embedding = visual_result["embedding"]
                    visual_model = self.vlm_config.model
                except Exception as e:
                    self.log(f"Failed to embed visual description: {e}", "warning")

        # Stage 2: Generate metadata embedding
        metadata_embedding: list[float] | None = None

        if metadata_text:
            try:
                if use_direct and self.image_embedder is not None:
                    # Use image embedder for text (no Ollama needed)
                    metadata_result = self.image_embedder.embed_text(metadata_text)
                    model_name = (
                        self.image_embedding_config.model
                        if self.image_embedding_config
                        else "unknown"
                    )
                    self.log(f"Embedded metadata using {model_name}", "debug")
                else:
                    # Fall back to text embedder (Ollama)
                    metadata_result = self.embedder.embed_text(metadata_text)
                metadata_embedding = metadata_result["embedding"]
            except Exception as e:
                self.log(f"Failed to embed metadata: {e}", "warning")

        # Stage 4: Combine embeddings
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

        # Stage 5: Store
        self.storage.store_embedding(
            scene_id=scene_id,
            composite_embedding=composite,
            text_model=self.embedding_config.model,
            visual_embedding=visual_embedding,
            metadata_embedding=metadata_embedding,
            visual_model=visual_model,
            visual_description=visual_description,
            metadata_text=metadata_text,
        )

        self.log(f"Stored embedding for scene {scene_id} ({len(composite)} dims)", "info")

        # Apply success tag if specified
        tagged = False
        if success_tag:
            tag_id = self._get_tag_id(success_tag)
            if not tag_id:
                tag_id = self._create_tag(success_tag)
            if tag_id:
                tagged = self._add_tag_to_scene(scene_id, tag_id)
                if tagged:
                    self.log(f"Tagged scene {scene_id} as '{success_tag}'", "debug")

        return {
            "success": True,
            "scene_id": scene_id,
            "dimensions": len(composite),
            "has_visual": visual_embedding is not None,
            "has_metadata": metadata_embedding is not None,
            "tagged": tagged,
        }

    def embed_all(
        self,
        force: bool = False,
        batch_size: int = 10,
        success_tag: str = "Embedded",
    ) -> dict[str, Any]:
        """
        Generate embeddings for all scenes in the library.

        Uses parallel processing: frame extraction runs in parallel threads,
        while GPU embedding is batched to avoid memory issues.

        Args:
            force: If True, regenerate all embeddings
            batch_size: Number of scenes per progress update
            success_tag: Tag name to apply after successful embedding (default "Embedded")

        Returns:
            Summary of embedding generation
        """
        try:
            return self._embed_all_impl(force, batch_size, success_tag)
        finally:
            # Ensure GPU resources are freed on completion or interruption
            self.cleanup()

    def _embed_all_impl(
        self,
        force: bool,
        batch_size: int,
        success_tag: str,
    ) -> dict[str, Any]:
        """Internal implementation of embed_all."""
        # Get all scene IDs
        scene_ids = self._get_all_scene_ids()
        total = len(scene_ids)

        # Determine number of workers
        num_workers = self.config.num_workers
        if num_workers <= 0:
            import multiprocessing

            num_workers = min(multiprocessing.cpu_count(), 8)

        self.log(f"Embedding {total} scenes using {num_workers} workers...", "info")

        # Get or create the success tag
        success_tag_id: int | None = None
        if success_tag:
            success_tag_id = self._get_tag_id(success_tag)
            if not success_tag_id:
                success_tag_id = self._create_tag(success_tag)
                if success_tag_id:
                    self.log(f"Created tag '{success_tag}' with ID {success_tag_id}", "info")
                else:
                    self.log(f"Could not create tag '{success_tag}', will skip tagging", "warning")

        # Thread-safe counters
        counter_lock = Lock()
        counters = {
            "success": 0,
            "skip": 0,
            "error": 0,
            "tagged": 0,
            "processed": 0,
        }
        errors: list[str] = []

        def process_scene(scene_id: int) -> None:
            """Process a single scene (thread-safe)."""
            nonlocal counters, errors

            try:
                result = self.embed_scene(scene_id, force=force)

                with counter_lock:
                    counters["processed"] += 1
                    if result.get("success"):
                        if result.get("cached"):
                            counters["skip"] += 1
                        else:
                            counters["success"] += 1

                        # Apply success tag
                        if success_tag_id:
                            if self._add_tag_to_scene(scene_id, success_tag_id):
                                counters["tagged"] += 1
                    else:
                        counters["error"] += 1
                        if len(errors) < 50:  # Limit error collection
                            errors.append(f"Scene {scene_id}: {result.get('error')}")

                    # Update progress periodically
                    if counters["processed"] % batch_size == 0:
                        self.progress(counters["processed"], total)

            except Exception as e:
                with counter_lock:
                    counters["processed"] += 1
                    counters["error"] += 1
                    if len(errors) < 50:
                        errors.append(f"Scene {scene_id}: {e!s}")
                self.log(f"Error embedding scene {scene_id}: {e}", "error")

        # Process scenes in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_scene, scene_id): scene_id for scene_id in scene_ids}

            # Wait for completion (as_completed allows for better progress tracking)
            for _future in as_completed(futures):
                # Exceptions are handled inside process_scene
                pass

        self.progress(total, total)

        if success_tag_id:
            self.log(f"Tagged {counters['tagged']} scenes as '{success_tag}'", "info")

        return {
            "total": total,
            "success": counters["success"],
            "skipped": counters["skip"],
            "errors": counters["error"],
            "tagged": counters["tagged"],
            "error_details": errors[:10],  # Limit error list in response
        }

    def _get_scene_info(self, scene_id: int) -> dict[str, Any] | None:
        """Fetch scene info from Stash database."""
        db_path = get_stash_db_path()
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

    def _get_visual_description(
        self,
        scene_id: int,
        scene_info: dict[str, Any],
    ) -> str | None:
        """
        Get visual description for a scene.

        First checks for cached vision analysis, then generates new if needed.
        """
        # Check for cached vision history
        if self.config.use_cached_descriptions:
            plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            history_file = os.path.join(
                plugin_dir, "assets", "scene_vision", f"vision_history_{scene_id}.json"
            )

            if os.path.exists(history_file):
                try:
                    with open(history_file) as f:
                        history = json.load(f)
                    if history.get("description"):
                        self.log(
                            f"Using cached vision description for scene {scene_id}",
                            "debug",
                        )
                        return str(history["description"])
                except (OSError, json.JSONDecodeError):
                    pass

        # Generate new description
        video_path = scene_info.get("video_path")
        if not video_path:
            return None

        duration = scene_info.get("duration")
        if not duration:
            return None

        try:
            # Extract frames
            frames = self.frame_extractor.get_or_extract_frames(
                str(scene_id),
                video_path,
                duration,
            )

            if not frames:
                self.log(f"No frames extracted for scene {scene_id}", "warning")
                return None

            # Get frames as base64
            frames_base64 = self.frame_extractor.get_frames_base64(str(scene_id))

            if not frames_base64:
                self.log(f"Failed to load frames as base64 for scene {scene_id}", "warning")
                return None

            # Check if VLM supports vision
            if not self.vlm.supports_vision:
                self.log(
                    f"VLM model {self.vlm_config.model} does not support vision",
                    "warning",
                )
                return None

            # Generate description using VLM (hot-reloaded prompt)
            description = self.vlm.complete(
                prompt=_get_visual_description_prompt(),
                images=frames_base64,
                temperature=0.5,
            )

            return cast("str", description)

        except Exception as e:
            self.log(f"Failed to generate visual description: {e}", "warning")
            return None

    def _select_representative_embeddings(
        self,
        embeddings: list[list[float]],
        frames: list[Any],
        scene_id: int,
    ) -> list[list[float]]:
        """
        Select diverse, representative frame embeddings using clustering.

        Args:
            embeddings: All frame embeddings
            frames: Frame metadata
            scene_id: Scene ID for logging

        Returns:
            Subset of embeddings for representative frames
        """
        from typing import Literal, cast

        from .frame_analysis import FrameAnalysisConfig, FrameAnalyzer, FrameEmbedding

        n_select = self.config.representative_n_frames
        method_str = self.config.representative_method

        # Validate method
        valid_methods = ("kmeans", "maximin", "coverage")
        if method_str not in valid_methods:
            method_str = "kmeans"
        method = cast("Literal['kmeans', 'maximin', 'coverage']", method_str)

        config = FrameAnalysisConfig(
            n_representative=n_select,
            selection_method=method,
        )

        # Create frame embedding objects
        frame_embeddings = [
            FrameEmbedding(
                index=f.index if hasattr(f, "index") else i + 1,
                timestamp=f.timestamp if hasattr(f, "timestamp") else 0.0,
                embedding=emb,
            )
            for i, (f, emb) in enumerate(zip(frames, embeddings))
        ]

        # Create analyzer (embedder not needed for selection, just pass existing)
        if self.image_embedder is None:
            raise ValueError("image_embedder required for frame selection")
        analyzer = FrameAnalyzer(
            embedder=cast("BaseImageEmbeddingProvider", self.image_embedder),
            config=config,
        )

        # Select representative frames
        result = analyzer.select_representative_frames(frame_embeddings, n_select, method)

        self.log(
            f"Scene {scene_id}: Selected {len(result['selected_indices'])} of "
            f"{len(embeddings)} frames (method={method}, "
            f"diversity={result['diversity_score']:.3f})",
            "debug",
        )

        # Return embeddings for selected frames (convert 1-based to 0-based indices)
        selected_indices_0based = [idx - 1 for idx in result["selected_indices"]]
        return [embeddings[i] for i in selected_indices_0based]

    def _average_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> list[float] | None:
        """Average multiple embeddings into a single embedding."""
        if not embeddings:
            return None

        if len(embeddings) == 1:
            return embeddings[0]

        # Stack and average
        arr: NDArray[np.float32] = np.array(embeddings, dtype=np.float32)
        averaged = np.mean(arr, axis=0)

        # Re-normalize for cosine similarity
        norm: float = float(np.linalg.norm(averaged))
        if norm > 0:
            averaged = averaged / norm

        return cast("list[float]", averaged.tolist())

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

    def _combine_embeddings(
        self,
        visual: list[float] | None,
        metadata: list[float] | None,
        visual_weight: float,
    ) -> list[float] | None:
        """
        Combine visual and metadata embeddings with weighting.

        If only one is available, returns that one.
        If both are available, returns weighted average (requires same dimensions).
        """
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

        return cast("list[float]", combined.tolist())

    def _get_all_scene_ids(self) -> list[int]:
        """Get all scene IDs from the database."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return []

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM scenes ORDER BY id")
        ids = [row["id"] for row in cursor.fetchall()]
        conn.close()
        return ids

    def _get_tag_id(self, tag_name: str) -> int | None:
        """Get tag ID by name from database."""
        db_path = get_stash_db_path()
        if not db_path.exists():
            return None

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM tags WHERE LOWER(name) = LOWER(?)",
            (tag_name,),
        )
        row = cursor.fetchone()
        conn.close()

        return row["id"] if row else None

    def _create_tag(self, tag_name: str) -> int | None:
        """Create a new tag via GraphQL."""
        if not self.stash:
            return None

        try:
            result = self.stash.call_GQL(
                """
                mutation TagCreate($input: TagCreateInput!) {
                    tagCreate(input: $input) {
                        id
                    }
                }
                """,
                {"input": {"name": tag_name}},
            )
            if result and "tagCreate" in result:
                return int(result["tagCreate"]["id"])
        except Exception as e:
            self.log(f"Failed to create tag '{tag_name}': {e}", "error")

        return None

    def _add_tag_to_scene(self, scene_id: int, tag_id: int) -> bool:
        """
        Add a tag to a scene (without removing any).

        Uses GraphQL to update the scene.
        """
        if not self.stash:
            return False

        try:
            # First, get current tags for the scene
            result = self.stash.call_GQL(
                """
                query FindScene($id: ID!) {
                    findScene(id: $id) {
                        tags {
                            id
                        }
                    }
                }
                """,
                {"id": str(scene_id)},
            )

            if not result or "findScene" not in result:
                return False

            current_tag_ids = [int(t["id"]) for t in result["findScene"]["tags"]]

            # Check if tag already exists
            if tag_id in current_tag_ids:
                return True  # Already has the tag

            # Add the new tag
            new_tag_ids = current_tag_ids + [tag_id]

            # Update the scene
            self.stash.call_GQL(
                """
                mutation SceneUpdate($input: SceneUpdateInput!) {
                    sceneUpdate(input: $input) {
                        id
                    }
                }
                """,
                {
                    "input": {
                        "id": str(scene_id),
                        "tag_ids": [str(tid) for tid in new_tag_ids],
                    }
                },
            )

            return True

        except Exception as e:
            self.log(f"Failed to add tag to scene {scene_id}: {e}", "warning")
            return False

    def _store_frame_embeddings(
        self,
        scene_id: int,
        frame_paths: list[str],
        embeddings: list[list[float]],
        duration: float,
    ) -> None:
        """
        Store individual frame embeddings for smart frame selection.

        Frame paths follow pattern: .../scene_123/frame_0001.jpg
        Frame 0001 = timestamp 0s (1fps extraction)

        Args:
            scene_id: Stash scene ID
            frame_paths: List of frame file paths
            embeddings: List of embedding vectors (same order as frame_paths)
            duration: Scene duration in seconds
        """
        import re

        if not frame_paths or not embeddings:
            return

        if len(frame_paths) != len(embeddings):
            self.log(
                f"Frame/embedding count mismatch: {len(frame_paths)} vs {len(embeddings)}",
                "warning",
            )
            return

        # Extract frame numbers from filenames (frame_0001.jpg -> 1)
        frame_pattern = re.compile(r"frame_(\d+)\.(?:jpg|png)$", re.IGNORECASE)

        stored_count = 0
        first_timestamp: float | None = None
        last_timestamp: float | None = None

        for frame_path, embedding in zip(frame_paths, embeddings):
            match = frame_pattern.search(frame_path)
            if not match:
                continue

            frame_num = int(match.group(1))
            # 1fps extraction: frame_0001 = timestamp 0, frame_0002 = timestamp 1, etc.
            timestamp = float(frame_num - 1)

            self.storage.store_frame_embedding(
                scene_id=scene_id,
                frame_index=stored_count,
                timestamp=timestamp,
                embedding=embedding,
            )

            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp
            stored_count += 1

        if stored_count > 0:
            # Store metadata for the scene
            composite = self._average_embeddings(embeddings)
            if composite:
                self.storage.store_frame_metadata(
                    scene_id=scene_id,
                    frame_count=stored_count,
                    total_frames_extracted=len(frame_paths),
                    duration=duration,
                    sampling_rate=1.0,  # 1fps
                    composite_embedding=composite,
                    dedup_ratio=0.0,  # No dedup at this stage
                    first_frame_timestamp=first_timestamp,
                    last_frame_timestamp=last_timestamp,
                )

            self.log(
                f"Stored {stored_count} frame embeddings for scene {scene_id}",
                "debug",
            )
