"""Type definitions for performer embeddings and similarity search."""

from dataclasses import dataclass
from typing import TypedDict


class PerformerEmbeddingRecord(TypedDict):
    """Stored performer embedding record."""

    performer_id: int
    model_key: str
    embedding: list[float]
    contributing_scenes: int  # Number of scenes used to build embedding
    total_engagement_score: float  # Sum of engagement scores from scenes
    visual_description: str | None  # AI-generated description
    top_tags: str | None  # JSON array of most common tags
    scene_count: int  # Total scenes featuring this performer
    created_at: str  # ISO datetime
    updated_at: str  # ISO datetime


class PerformerData(TypedDict):
    """Performer data from Stash database."""

    id: int
    name: str
    disambiguation: str | None
    gender: str | None
    birthdate: str | None
    ethnicity: str | None
    country: str | None
    height: int | None  # cm
    weight: int | None  # kg
    measurements: str | None
    fake_tits: str | None
    tattoos: str | None
    piercings: str | None
    aliases: str | None
    favorite: bool
    rating: int | None  # rating100 (0-100)
    details: str | None
    image_blob: str | None


class PerformerSceneData(TypedDict):
    """Data for a scene featuring a performer."""

    scene_id: int
    performer_id: int
    title: str | None
    date: str | None
    play_count: int
    o_count: int
    play_duration: float  # seconds
    rating: int | None  # rating100
    tags: list[str]  # Tag names


class PerformerEngagementData(TypedDict):
    """Aggregated engagement data for a performer's scenes."""

    performer_id: int
    total_scenes: int
    scenes_with_embeddings: int
    total_play_count: int
    total_o_count: int
    total_play_duration: float  # hours
    avg_rating: float | None  # Average rating100
    top_tags: list[str]  # Most common tags


@dataclass
class PerformerSimilarityResult:
    """Result from performer similarity search."""

    performer_id: int
    similarity: float
    name: str
    scene_count: int
    visual_description: str | None = None


class PerformerDetails(TypedDict):
    """Performer details for API responses."""

    id: int
    name: str
    disambiguation: str | None
    gender: str | None
    image_path: str | None
    scene_count: int
    favorite: bool
    rating100: int | None
    country: str | None


class SimilarPerformerResult(TypedDict):
    """Result from similar performer search."""

    performer_id: int
    similarity_score: float
    performer: PerformerDetails


class PerformerProfileInfo(TypedDict):
    """Profile info for performer-based recommendations."""

    performer_id: int
    performer_name: str
    contributing_scenes: int
    total_engagement_score: float
    created_at: str


@dataclass
class PerformerEmbeddingConfig:
    """Configuration for performer embedding generation."""

    # Minimum scenes required to create embedding
    min_scenes: int = 2

    # Maximum scenes to use for embedding (uses top engaged)
    max_scenes: int = 50

    # Weight for engagement scoring
    use_engagement_weighting: bool = True

    # Include scene embeddings without engagement data (weight = 1.0)
    include_unwatched: bool = True


@dataclass
class PerformerDescriptionConfig:
    """Configuration for AI-generated performer descriptions."""

    # Number of representative frames to use per scene
    frames_per_scene: int = 4

    # Maximum scenes to analyze
    max_scenes: int = 10

    # Include tag analysis
    include_tags: bool = True

    # Generate attributes (body type, hair, etc.)
    generate_attributes: bool = True
