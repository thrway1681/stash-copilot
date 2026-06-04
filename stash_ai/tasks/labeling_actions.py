"""Labeling-session task handlers (dispatch seam, #4, commit 4).

The four labeling modes migrated as a group: prepare a session, sync annotations
back, export the labeled dataset, and list sessions. Each is a *task-internal
result writer* — it writes its own ``labeling_*_{request_id}.json`` (preserving
the original per-mode formatting and asymmetric error handling: ``prepare`` and
``export`` write a specific error dict on failure, while ``sync`` and
``get_sessions`` only log) and declares ``result_key`` for the dispatch-seam
guard rather than routing through ``ResultStore`` (same pattern as
``RecommendationsTask`` / ``TasteMapTask``).

These modes are slated for removal under #13; they are migrated here only so the
cross-stack guard test treats every registered mode uniformly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..stash_client import StashClient
    from .dispatch import TaskContext


def _assets_dir() -> Path:
    """Repo-root ``assets/`` directory (same target as the old entry-point handlers)."""
    return Path(__file__).resolve().parents[2] / "assets"


class PrepareLabelingSessionTask:
    """Prepare a labeling session with uncertainty-sampled frames."""

    result_key = "labeling_session"

    def __init__(
        self,
        stash: StashClient,
        plugin_settings: dict[str, Any],
        request_id: str = "",
        batch_size: int = 200,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.stash = stash
        self.plugin_settings = plugin_settings
        self.request_id = request_id
        self.batch_size = batch_size
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> PrepareLabelingSessionTask:
        return cls(
            stash=ctx.stash,
            plugin_settings=ctx.plugin_settings,
            request_id=ctx.args.get("request_id", ""),
            batch_size=int(ctx.args.get("batch_size", 200)),
            log_callback=ctx.log,
        )

    def run(self) -> None:
        request_id = self.request_id
        self.log(
            f"Preparing labeling session (batch_size={self.batch_size}), request_id={request_id}",
            "info",
        )
        try:
            from ..embeddings.config import EmbeddingConfig
            from ..embeddings.storage import EmbeddingStorage
            from ..embeddings.tag_vocabulary import TagVocabulary
            from .labeling import LabelingTask
            from .labeling_types import LabelingConfig

            plugin_settings = self.plugin_settings

            image_provider = plugin_settings.get("image_embedding_provider")
            image_model = plugin_settings.get("image_embedding_model")
            image_device = plugin_settings.get("image_embedding_device") or "auto"

            if image_provider and image_model:
                model_key = EmbeddingConfig(
                    provider=image_provider, model=image_model, device=image_device
                ).model_key
            else:
                model_key = "siglip"

            storage = EmbeddingStorage(model_key=model_key)

            # Sync tag vocabulary before preparing session
            self.log("Syncing tag vocabulary...", "info")
            tag_vocab = TagVocabulary(storage=storage, model_key=model_key, log_callback=self.log)
            stash_tags = [t["name"] for t in self.stash.find_tags(f={})]
            tag_vocab.ensure_embeddings(stash_tags=stash_tags)

            config = LabelingConfig(
                batch_size=self.batch_size,
                uncertainty_low=float(plugin_settings.get("label_uncertainty_low", 0.25)),
                uncertainty_high=float(plugin_settings.get("label_uncertainty_high", 0.35)),
                max_suggested_tags=int(plugin_settings.get("label_suggested_tags", 10)),
                caption_template=plugin_settings.get(
                    "label_caption_template", "a scene featuring {tags}"
                ),
            )

            task = LabelingTask(
                stash=self.stash,
                storage=storage,
                log_callback=self.log,
                model_key=model_key,
            )

            result = task.prepare_session(config)

            assets_dir = _assets_dir()
            assets_dir.mkdir(exist_ok=True)
            result_file = assets_dir / f"labeling_session_{request_id}.json"
            result_file.write_text(json.dumps(result, indent=2))

            self.log(f"Labeling session written to {result_file}", "info")

        except Exception as e:
            self.log(f"Error preparing labeling session: {e}", "error")
            result_file = _assets_dir() / f"labeling_session_{request_id}.json"
            result_file.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "session_id": "",
                        "batch": [],
                        "vocabulary": [],
                        "error": str(e),
                    }
                )
            )


class SyncLabelingAnnotationsTask:
    """Sync annotations back from the labeling UI."""

    result_key = "labeling_sync"

    def __init__(
        self,
        stash: StashClient,
        request_id: str = "",
        payload_json: str = "{}",
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.stash = stash
        self.request_id = request_id
        self.payload_json = payload_json
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> SyncLabelingAnnotationsTask:
        return cls(
            stash=ctx.stash,
            request_id=ctx.args.get("request_id", ""),
            payload_json=ctx.args.get("payload", "{}"),
            log_callback=ctx.log,
        )

    def run(self) -> None:
        try:
            from ..embeddings.storage import EmbeddingStorage
            from .labeling import LabelingTask

            payload = json.loads(self.payload_json)
            storage = EmbeddingStorage()

            task = LabelingTask(
                stash=self.stash,
                storage=storage,
                log_callback=self.log,
            )

            task.sync_annotations(payload)

            result_file = _assets_dir() / f"labeling_sync_{self.request_id}.json"
            result_file.write_text(json.dumps({"status": "complete"}))

        except Exception as e:
            self.log(f"Error syncing annotations: {e}", "error")


class ExportLabelingDatasetTask:
    """Export labeled data as a WebDataset."""

    result_key = "labeling_export"

    def __init__(
        self,
        stash: StashClient,
        plugin_settings: dict[str, Any],
        request_id: str = "",
        include_negatives: bool = True,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.stash = stash
        self.plugin_settings = plugin_settings
        self.request_id = request_id
        self.include_negatives = include_negatives
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> ExportLabelingDatasetTask:
        return cls(
            stash=ctx.stash,
            plugin_settings=ctx.plugin_settings,
            request_id=ctx.args.get("request_id", ""),
            include_negatives=ctx.args.get("include_negatives", "true").lower() == "true",
            log_callback=ctx.log,
        )

    def run(self) -> None:
        request_id = self.request_id
        self.log(f"Exporting labeling dataset, request_id={request_id}", "info")
        try:
            from ..embeddings.storage import EmbeddingStorage
            from .labeling import LabelingTask
            from .labeling_types import LabelingConfig

            storage = EmbeddingStorage()
            config = LabelingConfig.from_plugin_settings(self.plugin_settings)

            task = LabelingTask(
                stash=self.stash,
                storage=storage,
                log_callback=self.log,
            )

            result = task.export_dataset(config, include_negatives=self.include_negatives)

            result_file = _assets_dir() / f"labeling_export_{request_id}.json"
            result_file.write_text(json.dumps(result, indent=2))

            self.log(f"Export result written to {result_file}", "info")

        except Exception as e:
            self.log(f"Error exporting dataset: {e}", "error")
            result_file = _assets_dir() / f"labeling_export_{request_id}.json"
            result_file.write_text(
                json.dumps(
                    {
                        "status": "error",
                        "export_path": "",
                        "total_images": 0,
                        "total_tags": 0,
                        "error": str(e),
                    }
                )
            )


class GetLabelingSessionsTask:
    """List existing labeling sessions."""

    result_key = "labeling_sessions"

    def __init__(
        self,
        request_id: str = "",
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.request_id = request_id
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> GetLabelingSessionsTask:
        return cls(
            request_id=ctx.args.get("request_id", ""),
            log_callback=ctx.log,
        )

    def run(self) -> None:
        try:
            from ..embeddings.storage import EmbeddingStorage

            storage = EmbeddingStorage()
            sessions = storage.list_labeling_sessions()

            result_file = _assets_dir() / f"labeling_sessions_{self.request_id}.json"
            result_file.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "sessions": sessions,
                    },
                    indent=2,
                )
            )

        except Exception as e:
            self.log(f"Error listing sessions: {e}", "error")
