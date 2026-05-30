"""Type definitions for the recommendation system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class RecommendationMode(Enum):
    """Recommendation modes."""

    DISCOVER_NEW = "discover_new"  # Unwatched scenes similar to profile
    REWATCH_FAVORITES = "rewatch"  # Watched scenes by engagement + similarity
    O_MOMENTS = "o_moments"  # Find scenes with similar peak moments
    PERFORMER_PREFERENCE = "performer_preference"  # Scenes similar to favorite performers


class EngagementScoringMethod(Enum):
    """Methods for scoring scene engagement."""

    BASE_WEIGHTED = "base_weighted"  # Canonical: o_count*20 + replays*2 + stars*1.5
    TIME_DECAYED = "time_decayed"  # Recent engagement weighs more


class SceneEngagementData(TypedDict):
    """Engagement data for a single scene."""

    scene_id: int
    view_count: int
    o_count: int
    play_duration: float  # In seconds
    last_played: str | None  # ISO datetime string
    first_played: str | None  # ISO datetime string
    rating: int | None  # rating100 from Stash (0-100 scale, None if unrated)


class EngagementWeights(TypedDict):
    """Configurable weights for engagement scoring."""

    o_count: float  # Default: 20.0 (median scene plays between o_counts)
    view_count: float  # Default: 2.0 (per replay, views beyond the first)
    rating: float  # Default: 1.5 (per star on 5-star scale, only adds if rated)


class TimeDecayConfig(TypedDict):
    """Configuration for time-based decay scoring."""

    half_life_days: float  # Default: 30 (engagement halves every 30 days)
    min_weight: float  # Default: 0.1 (minimum decay multiplier)
    min_score_threshold: float  # Default: 0.0 (minimum decayed score to include in profile)


@dataclass
class EngagementScore:
    """Calculated engagement score for a scene."""

    scene_id: int
    raw_score: float  # Base weighted score
    time_decayed_score: float  # After time decay
    components: dict[str, float]  # Breakdown: {o_count: x, view_count: y, rating: z}


@dataclass
class RecommendationConfig:
    """Configuration for recommendation generation."""

    mode: RecommendationMode = RecommendationMode.DISCOVER_NEW
    scoring_method: EngagementScoringMethod = EngagementScoringMethod.BASE_WEIGHTED

    # Profile building
    top_scenes_for_profile: int = 20  # Number of top scenes to build profile from

    # Engagement weights
    weights: EngagementWeights = field(
        default_factory=lambda: cast(
            "EngagementWeights",
            {
                "o_count": 20.0,  # Median scene plays between o_counts
                "view_count": 2.0,  # Per replay (views beyond the first)
                "rating": 1.5,  # Per star (5-star scale), only adds if scene is rated
            },
        )
    )

    # Time decay settings
    time_decay: TimeDecayConfig = field(
        default_factory=lambda: cast(
            "TimeDecayConfig",
            {
                "half_life_days": 30.0,
                "min_weight": 0.1,
                "min_score_threshold": 0.0,
            },
        )
    )

    # Output settings
    limit: int = 120  # Number of recommendations (10 pages x 12 per page)
    per_page: int = 12  # Results per page for pagination metadata
    min_similarity: float = 0.1  # Minimum cosine similarity threshold (lower = more results)

    # Seed scene settings (for scene-specific recommendations)
    seed_scene_id: int | None = None  # Scene to boost similarity toward
    seed_weight: float = 0.3  # How much to weight seed similarity (0-1)

    # Rewatch mode: balance between engagement and similarity
    # 0.0 = pure similarity, 1.0 = pure engagement
    engagement_weight: float = 0.6  # Default: 60% engagement, 40% similarity


@dataclass
class UserPreferenceProfile:
    """Computed user preference profile."""

    profile_embedding: list[float]
    contributing_scenes: list[int]  # Scene IDs used to build profile
    total_engagement_score: float
    created_at: str  # ISO datetime
    scoring_method: EngagementScoringMethod


class FileDetails(TypedDict):
    """File details for a scene."""

    path: str | None
    size: int | None
    duration: float | None
    height: int | None
    width: int | None
    fingerprints: list[dict[str, str]]


class SceneDetails(TypedDict):
    """Scene details for recommendation results."""

    id: int
    title: str | None
    date: str | None
    rating100: int | None
    studio: dict[str, Any] | None
    performers: list[dict[str, Any]]
    tags: list[dict[str, Any]]
    # Additional fields for interactive features
    files: list[FileDetails]
    play_count: int
    o_counter: int
    interactive: bool


class RecommendationResult(TypedDict):
    """A single recommendation result."""

    scene_id: int
    similarity_score: float  # Cosine similarity to profile
    engagement_score: float  # For rewatch mode
    combined_score: float  # Weighted combination
    scene: SceneDetails  # Scene details (title, performers, etc.)


class ProfileInfo(TypedDict):
    """Profile summary info for response."""

    contributing_scenes: list[int]
    total_engagement_score: float
    scene_count: int


class PaginationInfo(TypedDict):
    """Pagination metadata for recommendation results."""

    total_results: int
    per_page: int
    total_pages: int


class RecommendationResponse(TypedDict):
    """Full response from recommendation task."""

    status: str  # "complete", "error"
    mode: str  # "discover_new" or "rewatch"
    scoring_method: str  # "base_weighted" or "time_decayed"
    profile: ProfileInfo
    results: list[RecommendationResult]
    pagination: PaginationInfo  # Pagination metadata
    generated_at: str  # ISO datetime
    request_id: str  # For frontend tracking


# ============================================================================
# O-Moment Embeddings Types
# ============================================================================


class OMomentMarker(TypedDict):
    """O marker data from Stash scene_markers table."""

    marker_id: int
    scene_id: int
    seconds: float  # Exact playback position from scene_markers.seconds
    end_seconds: float | None  # Optional end time
    created_at: str | None  # ISO datetime


class OMomentData(TypedDict):
    """O-moment data with marker information for embedding."""

    scene_id: int
    marker: OMomentMarker
    o_event_index: int  # Which O event for this scene (0-indexed)


@dataclass
class OMomentEmbedding:
    """Embedding derived from frames around an O-moment marker."""

    scene_id: int
    o_event_index: int  # Which O event for this scene (0-indexed)
    marker_id: int  # Links to scene_markers.id
    center_timestamp: float  # Center of extraction window (seconds)
    window_seconds: float  # Total window size (e.g., 120 for +/- 60s)
    frame_count: int  # How many frames were averaged
    model_key: str  # Embedding model used
    created_at: str  # ISO datetime


@dataclass
class OMomentExtractionConfig:
    """Configuration for O-moment frame extraction."""

    window_seconds: float = 120.0  # Total window (default: +/- 60s)
    frames_per_window: int = 12  # Frames to extract per window
    o_tag_name: str = "O"  # Tag name for O markers


@dataclass
class OMomentProfileInfo:
    """Profile info for O-moment based recommendations."""

    contributing_moments: list[int]  # Marker IDs used in profile
    contributing_scenes: list[int]  # Unique scene IDs
    total_moments: int
    created_at: str  # ISO datetime


# ============================================================================
# Taste Map Types
# ============================================================================


class TagMatch(TypedDict):
    """A vocabulary phrase matched to a cluster centroid."""

    text: str
    similarity: float
    source: str  # 'stash_tag' | 'curated' | 'user'


class TasteClusterData(TypedDict):
    """Serializable cluster data for JSON output."""

    cluster_id: int
    auto_label: str
    scene_ids: list[int]
    engagement_total: float
    engagement_share: float
    representative_scenes: list[int]
    tag_matches: list[TagMatch]


class TasteMapSceneData(TypedDict):
    """Per-scene data for the 3D taste map visualization."""

    scene_id: int
    x: float
    y: float
    z: float
    cluster_id: int | None  # None for non-profile scenes
    engagement_score: float
    is_profile: bool
    title: str | None
    thumbnail: str | None
    play_count: int
    o_counter: int


class TasteMapResponse(TypedDict):
    """Full taste map response saved to JSON."""

    status: str  # 'complete' | 'error'
    optimal_k: int
    silhouette_score: float
    clusters: list[TasteClusterData]
    scenes: list[TasteMapSceneData]
    error: str | None


@dataclass
class TasteCluster:
    """Runtime cluster with embedding data (not serialized to JSON)."""

    cluster_id: int
    centroid: NDArray[np.float32]
    scene_ids: list[int]
    engagement_total: float
    engagement_share: float
    auto_label: str
    user_label: str | None
    weight_override: float | None
    excluded: bool
    tag_matches: list[TagMatch]


@dataclass
class TasteProfile:
    """Complete taste profile with all clusters."""

    clusters: list[TasteCluster]
    optimal_k: int
    silhouette_score: float
    model_key: str
