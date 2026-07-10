"""Find scenes visually similar to the frame playing at a given timestamp.

Migrated from the inline entry-point handler under #4, commit 4. Result-producing:
the handler routes ``run()``'s dict through the seam's ``ResultStore`` as
``frame_search_{request_id|latest}.json``. Extracts a single frame at the given
timestamp, embeds it with the configured image provider, and searches the FAISS
frame index for visually similar frames across the library.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import paths
from ..embeddings.config import EmbeddingConfig
from ..embeddings.frame_search import FrameSearchIndex
from ..embeddings.provider import get_embedding_provider
from ..tools.database import get_readonly_connection, get_stash_db_path
from ..tools.scene_details import get_scene_details_batch
from .frame_extractor import FrameExtractionConfig, FrameExtractor

if TYPE_CHECKING:
    from .dispatch import TaskContext


class FindSimilarByFrameTask:
    """Find scenes similar to the frame at a playback timestamp via frame embeddings."""

    result_key = "frame_search"

    def __init__(
        self,
        scene_id: str = "",
        timestamp_str: str = "0",
        limit_raw: Any = 20,
        request_id: str = "",
        image_provider: str | None = None,
        image_model: str | None = None,
        image_device: str = "auto",
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.scene_id = scene_id
        self.timestamp_str = timestamp_str
        self.limit_raw = limit_raw
        self.request_id = request_id
        self.image_provider = image_provider
        self.image_model = image_model
        self.image_device = image_device
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> FindSimilarByFrameTask:
        """Build from a standard :class:`TaskContext` — resolves scene/timestamp + image settings."""
        args = ctx.args
        ps = ctx.plugin_settings
        return cls(
            scene_id=args.get("scene_id", "").strip(),
            timestamp_str=args.get("timestamp", "0"),
            limit_raw=args.get("limit", 20),
            request_id=args.get("request_id", ""),
            image_provider=ps.get("image_embedding_provider"),
            image_model=ps.get("image_embedding_model"),
            image_device=ps.get("image_embedding_device") or "auto",
            log_callback=ctx.log,
        )

    def run(self) -> dict[str, Any]:
        """Run the frame search and return the result dict (or an error dict)."""
        request_id = self.request_id
        try:
            scene_id = self.scene_id

            try:
                timestamp = float(self.timestamp_str)
                limit = int(self.limit_raw)
            except ValueError as e:
                return {
                    "status": "error",
                    "error": f"Invalid parameter: {e}",
                    "request_id": request_id,
                }

            self.log(
                f"Frame search: scene={scene_id}, timestamp={timestamp:.1f}s, limit={limit}", "info"
            )

            # Step 1: Resolve video file path from Stash SQLite
            db_path = get_stash_db_path()
            conn = get_readonly_connection(db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT fo.path || '/' || f.basename as video_path
                    FROM scenes s
                    JOIN scenes_files sf ON s.id = sf.scene_id AND sf."primary" = 1
                    JOIN files f ON sf.file_id = f.id
                    JOIN folders fo ON f.parent_folder_id = fo.id
                    JOIN video_files vf ON f.id = vf.file_id
                    WHERE s.id = ?
                    """,
                    (int(scene_id),),
                )
                row = cursor.fetchone()
            finally:
                conn.close()

            if not row:
                return {
                    "status": "error",
                    "error": f"Could not find video file for scene {scene_id}",
                    "request_id": request_id,
                }

            video_path = row["video_path"]

            # Step 2: Extract frame at timestamp (ephemeral - no disk caching)
            extractor = FrameExtractor(
                config=FrameExtractionConfig(),
                cache_dir=str(paths.frames_cache_dir()),
                log_callback=self.log,
            )

            frame_bytes = extractor.extract_frame_at_timestamp(video_path, timestamp)
            if frame_bytes is None:
                return {
                    "status": "error",
                    "error": f"Failed to extract frame at {timestamp:.1f}s",
                    "request_id": request_id,
                }

            self.log(f"Extracted frame: {len(frame_bytes)} bytes", "debug")

            # Step 3: Embed the frame
            if not self.image_provider or not self.image_model:
                return {
                    "status": "error",
                    "error": "No image embedding provider configured. Set up in Plugin Settings.",
                    "request_id": request_id,
                }

            embedding_config = EmbeddingConfig(
                provider=self.image_provider,
                model=self.image_model,
                device=self.image_device,
            )
            model_key = embedding_config.model_key

            embedder = get_embedding_provider(embedding_config)
            if not hasattr(embedder, "embed_image"):
                return {
                    "status": "error",
                    "error": f"Provider '{self.image_provider}' does not support image embedding.",
                    "request_id": request_id,
                }
            result = embedder.embed_image(frame_bytes)
            query_embedding = np.array(result["embedding"], dtype=np.float32)

            self.log(f"Embedded frame: {result['dimensions']} dims", "debug")

            # Step 4: Load frame search index
            frame_index = FrameSearchIndex(model_key=model_key)

            if not frame_index.exists:
                return {
                    "status": "error",
                    "error": f"Frame search index not found for model '{model_key}'. Run 'Build Frame Search Index' task first.",
                    "request_id": request_id,
                }

            # Step 5: Search for similar frames
            # Over-fetch frames since many top matches may belong to the same scene.
            # After aggregate_to_scenes(), we need enough unique scenes to fill `limit`.
            frame_matches = frame_index.search(query_embedding, top_k=2000)

            # Step 6: Filter out query scene's own frames
            frame_matches = [m for m in frame_matches if m.scene_id != int(scene_id)]

            # Step 7: Aggregate to best match per scene
            scene_matches = frame_index.aggregate_to_scenes(frame_matches)

            # Step 8: Truncate to limit
            scene_matches = scene_matches[:limit]

            # Step 9: Fetch scene details
            scene_details = get_scene_details_batch([m.scene_id for m in scene_matches], self.log)

            # Step 10: Build result data
            result_data = []
            for m in scene_matches:
                scene = scene_details.get(m.scene_id, {})
                frame_path = (
                    f"embedded_frames/scene_{m.scene_id}/frame_{m.best_frame_index:04d}.jpg"
                )
                result_data.append(
                    {
                        "scene_id": m.scene_id,
                        "similarity": m.similarity,
                        "matched_timestamp": m.best_timestamp,
                        "matched_frame_index": m.best_frame_index,
                        "frame_path": frame_path,
                        "scene": scene,
                    }
                )

            self.log(f"Frame search complete: {len(result_data)} scenes found", "info")
            return {
                "status": "complete",
                "query_scene_id": int(scene_id),
                "query_timestamp": timestamp,
                "model_key": model_key,
                "results": result_data,
                "limit": limit,
                "request_id": request_id,
            }

        except Exception as e:
            self.log(f"Frame search error: {e}", "error")
            return {"status": "error", "error": str(e), "request_id": request_id}
