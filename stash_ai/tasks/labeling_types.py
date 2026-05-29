"""Type definitions for the image labeling task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class FrameSuggestion(TypedDict):
    """A suggested tag for a specific frame."""
    tag_text: str
    tag_source: str  # "stash_tag" | "curated" | "user"
    similarity: float


class LabelingFrameItem(TypedDict):
    """A single frame in the labeling batch."""
    scene_id: int
    frame_index: int
    frame_path: str
    timestamp: str  # "MM:SS" format
    uncertainty_score: float
    suggested_tags: list[FrameSuggestion]
    scene_tags: list[str]
    scene_title: str


class LabelingSessionResult(TypedDict):
    """Result from PrepareSession task."""
    status: str  # "complete" | "error" | "no_embeddings"
    session_id: str
    batch: list[LabelingFrameItem]
    vocabulary: list[str]
    error: str | None


class AnnotationPayload(TypedDict):
    """Payload sent from JS to sync annotations."""
    session_id: str
    annotations: list[dict[str, Any]]
    progress: list[dict[str, Any]]


class ExportResult(TypedDict):
    """Result from ExportDataset task."""
    status: str
    export_path: str
    total_images: int
    total_tags: int
    error: str | None


@dataclass
class LabelingConfig:
    """Configuration for labeling sessions."""
    batch_size: int = 200
    uncertainty_low: float = 0.25
    uncertainty_high: float = 0.35
    max_suggested_tags: int = 10
    caption_template: str = "a scene featuring {tags}"
    max_candidates: int = 50_000  # Cap on embeddings loaded for uncertainty ranking

    @classmethod
    def from_plugin_settings(cls, settings: dict[str, Any]) -> LabelingConfig:
        """Create from raw Stash plugin settings."""
        return cls(
            batch_size=int(settings.get("label_batch_size", 200)),
            uncertainty_low=float(settings.get("label_uncertainty_low", 0.25)),
            uncertainty_high=float(settings.get("label_uncertainty_high", 0.35)),
            max_suggested_tags=int(settings.get("label_suggested_tags", 10)),
            caption_template=settings.get(
                "label_caption_template", "a scene featuring {tags}"
            ),
            max_candidates=int(settings.get("label_max_candidates", 50_000)),
        )
