"""Semantic scene search by natural-language text query.

Migrated from the inline entry-point handler under #4, commit 4. Result-producing:
the handler routes ``run()``'s dict through the seam's ``ResultStore`` as
``search_results_{request_id|latest}.json``. Supports both whole-scene text→image
search and frame-level FAISS search (``frame_search``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..embeddings.config import EmbeddingConfig
from ..embeddings.provider import get_embedding_provider
from ..embeddings.storage import EmbeddingStorage
from ..tools.scene_details import get_scene_details_batch

if TYPE_CHECKING:
    from .dispatch import TaskContext


class SearchByTextTask:
    """Search scenes by a text query via text→image embedding similarity."""

    result_key = "search_results"

    def __init__(
        self,
        query: str = "",
        limit: int = 24,
        offset: int = 0,
        request_id: str = "",
        requested_model_key: str = "",
        frame_search: bool = False,
        image_device: str = "auto",
        image_provider: str | None = None,
        image_model: str | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.query = query
        self.limit = limit
        self.offset = offset
        self.request_id = request_id
        self.requested_model_key = requested_model_key
        self.frame_search = frame_search
        self.image_device = image_device
        self.image_provider = image_provider
        self.image_model = image_model
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> SearchByTextTask:
        """Build from a standard :class:`TaskContext` — resolves the query + search args."""
        args = ctx.args
        ps = ctx.plugin_settings

        query = args.get("query", "").strip()
        limit = int(args.get("limit", 24))
        offset = int(args.get("offset", 0))
        frame_search = args.get("frame_search", "").lower() == "true"

        ctx.log(
            f"Searching scenes for: '{query}' (limit={limit}, offset={offset}, "
            f"frame_search={frame_search})",
            "info",
        )

        return cls(
            query=query,
            limit=limit,
            offset=offset,
            request_id=args.get("request_id", ""),
            requested_model_key=args.get("model_key", "").strip(),
            frame_search=frame_search,
            image_device=ps.get("image_embedding_device") or "auto",
            image_provider=ps.get("image_embedding_provider"),
            image_model=ps.get("image_embedding_model"),
            log_callback=ctx.log,
        )

    def run(self) -> dict[str, Any]:
        """Run the search and return the result dict (or an error dict)."""
        try:
            query = self.query
            limit = self.limit
            offset = self.offset
            request_id = self.request_id

            # Resolve the embedding model/config: explicit model_key wins, else settings.
            if self.requested_model_key:
                embedding_config = EmbeddingConfig.from_model_key(
                    self.requested_model_key, device=self.image_device
                )
                model_key = self.requested_model_key
                self.log(f"Using requested model: {model_key}", "info")
            else:
                if not self.image_provider or not self.image_model:
                    return {
                        "status": "error",
                        "error": "Image embedding provider not configured. Set up in Plugin Settings.",
                    }
                embedding_config = EmbeddingConfig(
                    provider=self.image_provider,
                    model=self.image_model,
                    device=self.image_device,
                )
                model_key = embedding_config.model_key

            embedder = get_embedding_provider(embedding_config)
            storage = EmbeddingStorage(model_key=model_key)

            stats = storage.get_stats()
            if stats["total_embeddings"] == 0:
                return {
                    "status": "error",
                    "error": "No scene embeddings found. Run 'Embed All Scenes' task first.",
                }

            # Frame-level search using the FAISS index.
            if self.frame_search:
                import numpy as np

                from ..embeddings.frame_search import FrameSearchIndex

                frame_index = FrameSearchIndex(model_key=model_key)
                if not frame_index.exists:
                    return {
                        "status": "error",
                        "error": f"Frame search index not built for model '{model_key}'. Run 'Build Frame Search Index' task first.",
                    }

                try:
                    result = embedder.embed_text(query)
                    query_embedding = np.array(result["embedding"], dtype=np.float32)
                except Exception as e:
                    return {"status": "error", "error": f"Failed to embed query: {e!s}"}

                frame_matches = frame_index.search(query_embedding, top_k=2000)
                scene_matches = frame_index.aggregate_to_scenes(frame_matches)
                paginated = scene_matches[offset : offset + limit]

                scene_details = get_scene_details_batch([m.scene_id for m in paginated], self.log)

                result_data = []
                for m in paginated:
                    scene = scene_details.get(m.scene_id, {})
                    frame_path = (
                        f"embedded_frames/scene_{m.scene_id}/frame_{m.best_frame_index:04d}.jpg"
                    )
                    result_data.append(
                        {
                            "scene_id": m.scene_id,
                            "similarity": m.similarity,
                            "best_frame_index": m.best_frame_index,
                            "best_timestamp": m.best_timestamp,
                            "frame_path": frame_path,
                            "scene": scene,
                        }
                    )

                has_more = len(scene_matches) > (offset + limit)
                self.log(f"Frame search complete: {len(result_data)} scenes for '{query}'", "info")
                return {
                    "status": "complete",
                    "query": query,
                    "model_key": model_key,
                    "frame_search": True,
                    "results": result_data,
                    "offset": offset,
                    "limit": limit,
                    "has_more": has_more,
                    "request_id": request_id,
                    "total_scenes": len(scene_matches),
                }

            # Whole-scene text→image search.
            try:
                result = embedder.embed_text(query)
                text_query_embedding = result["embedding"]
            except Exception as e:
                return {"status": "error", "error": f"Failed to embed query: {e!s}"}

            # Text-to-image similarities are low (0.01-0.10); no minimum threshold.
            results = storage.find_similar(
                query_embedding=text_query_embedding,
                limit=limit,
                offset=offset,
                min_similarity=0.0,
            )

            scene_details = get_scene_details_batch([r.scene_id for r in results], self.log)
            result_data = []
            for r in results:
                scene = scene_details.get(r.scene_id, {})
                result_data.append(
                    {"scene_id": r.scene_id, "similarity": r.similarity, "scene": scene}
                )

            has_more = len(results) == limit
            self.log(f"Search complete: {len(result_data)} results for '{query}'", "info")
            return {
                "status": "complete",
                "query": query,
                "model_key": model_key,
                "results": result_data,
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
                "request_id": request_id,
                "total_embeddings": stats["total_embeddings"],
            }

        except Exception as e:
            self.log(f"Search error: {e}", "error")
            return {"status": "error", "error": str(e)}
