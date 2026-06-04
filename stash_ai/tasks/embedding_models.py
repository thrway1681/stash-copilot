"""Task for listing the embedding models that have stored embeddings.

Result-producing: writes ``assets/embedding_models_{request_id|latest}.json``
itself (declares ``result_key`` for the dispatch-seam guard; the frontend polls
that file). Owns its own error handling because, unlike most tasks, it must write
a ``{"status": "error"}`` result file on failure so the UI stops polling — so
``EmbeddingStorage`` is imported inside ``run()`` and exceptions are caught there
(matching the old handler exactly), rather than relying on ``dispatch``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dispatch import TaskContext


class GetEmbeddingModelsTask:
    """List embedding models (keys, counts, dimensions, timestamps) + the current one."""

    result_key = "embedding_models"

    def __init__(
        self,
        plugin_settings: dict[str, Any],
        request_id: str = "",
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.plugin_settings = plugin_settings
        self.request_id = request_id
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> GetEmbeddingModelsTask:
        """Build from the standard :class:`TaskContext` (reads plugin settings + request_id)."""
        return cls(
            plugin_settings=ctx.plugin_settings,
            request_id=str(ctx.args.get("request_id", "")),
            log_callback=ctx.log,
        )

    def _write_result(self, data: dict[str, Any]) -> None:
        """Write the result file the frontend polls (``_latest`` when no request_id)."""
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets_dir = os.path.join(plugin_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        filename = f"embedding_models_{self.request_id or 'latest'}.json"
        result_file = os.path.join(assets_dir, filename)
        try:
            with open(result_file, "w") as f:
                json.dump(data, f)
            self.log(f"Wrote embedding models to: {result_file}", "debug")
        except Exception as e:
            self.log(f"Failed to write embedding models file: {e}", "error")

    def run(self) -> dict[str, Any]:
        """Collect model stats and persist the result; write an error file on failure."""
        try:
            from stash_ai.embeddings.storage import EmbeddingStorage

            self.log("Fetching available embedding models...", "info")

            # model_key doesn't matter for listing all models.
            storage = EmbeddingStorage(model_key="siglip")
            model_keys = storage.get_available_model_keys()

            models_data = []
            for model_key in model_keys:
                model_storage = EmbeddingStorage(model_key=model_key)
                stats = model_storage.get_stats()
                models_data.append(
                    {
                        "model_key": model_key,
                        "count": stats["total_embeddings"],
                        "dimensions": list(stats["dimensions_distribution"].keys())[0]
                        if stats["dimensions_distribution"]
                        else None,
                        "oldest": stats["oldest_embedding"],
                        "newest": stats["newest_embedding"],
                    }
                )

            current_provider = self.plugin_settings.get("image_embedding_provider", "")
            current_model = self.plugin_settings.get("image_embedding_model", "")
            current_model_key = None
            if current_provider and current_model:
                if current_provider == "siglip":
                    current_model_key = "siglip"
                else:
                    current_model_key = f"{current_provider}:{current_model}"

            result_data: dict[str, Any] = {
                "status": "complete",
                "models": models_data,
                "current_model_key": current_model_key,
                "request_id": self.request_id,
            }
            self._write_result(result_data)
            self.log(f"Found {len(models_data)} embedding models", "info")
            return result_data

        except ImportError as e:
            self.log(f"Failed to import embedding modules: {e}", "error")
            error_data: dict[str, Any] = {
                "status": "error",
                "error": f"Failed to import embedding modules: {e}",
            }
            self._write_result(error_data)
            return error_data
        except Exception as e:
            self.log(f"Error getting embedding models: {e}", "error")
            error_data = {"status": "error", "error": str(e)}
            self._write_result(error_data)
            return error_data
