"""Build the FAISS index for frame-level semantic search.

Migrated from the inline entry-point handler under #4, commit 4. Log-only: the
index is written to disk by ``FrameSearchIndex.build`` and the task only logs
progress + a final summary; it declares no ``result_key`` (nothing is polled).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..embeddings.config import EmbeddingConfig
from ..embeddings.frame_search import FrameSearchIndex
from ..embeddings.storage import EmbeddingStorage

if TYPE_CHECKING:
    from .dispatch import TaskContext


class BuildFrameIndexTask:
    """Build (and persist) the frame-level FAISS index for an embedding model."""

    def __init__(
        self,
        model_key: str = "",
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        # Empty model_key signals "image embedding provider not configured".
        self.model_key = model_key
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda cur, total: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> BuildFrameIndexTask:
        """Build from a standard :class:`TaskContext` — resolves the embedding model key.

        An explicit ``model_key`` arg wins; otherwise it derives from the
        configured image-embedding provider/model. An empty result means the
        provider is not configured (``run`` logs an error and stops).
        """
        args = ctx.args
        ps = ctx.plugin_settings

        requested_model_key = args.get("model_key", "").strip()
        model_key = ""
        if requested_model_key:
            model_key = requested_model_key
        else:
            image_provider = ps.get("image_embedding_provider")
            image_model = ps.get("image_embedding_model")
            if image_provider and image_model:
                model_key = EmbeddingConfig(
                    provider=image_provider,
                    model=image_model,
                    device="cpu",  # Not used for indexing
                ).model_key

        return cls(
            model_key=model_key,
            log_callback=ctx.log,
            progress_callback=ctx.progress,
        )

    def run(self) -> None:
        """Build the FAISS index, logging progress and a final summary."""
        if not self.model_key:
            self.log(
                "Image embedding provider not configured. Set up in Plugin Settings first.",
                "error",
            )
            return

        self.log(f"Building frame search index for model: {self.model_key}", "info")

        storage = EmbeddingStorage(model_key=self.model_key)
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        frame_index = FrameSearchIndex(assets_dir=assets_dir, model_key=self.model_key)

        def progress_callback(current: int, total: int) -> None:
            self.progress(current, total)
            if current % 50000 == 0 or current == total:
                self.log(f"Indexed {current:,} / {total:,} frames", "info")

        info = frame_index.build(storage=storage, progress_callback=progress_callback)

        self.log(
            f"Frame search index built successfully:\n"
            f"  Model: {info.model_key}\n"
            f"  Frames: {info.frame_count:,}\n"
            f"  Scenes: {info.scene_count:,}\n"
            f"  Dimensions: {info.dimensions}",
            "info",
        )
