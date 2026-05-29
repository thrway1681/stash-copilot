"""Recommendation system for personalized scene suggestions."""

from .engagement import EngagementCalculator
from .engine import RecommendationEngine
from .performer_profile import PerformerProfileBuilder
from .performer_types import (
    PerformerData,
    PerformerDescriptionConfig,
    PerformerDetails,
    PerformerEmbeddingConfig,
    PerformerEmbeddingRecord,
    PerformerEngagementData,
    PerformerProfileInfo,
    PerformerSceneData,
    PerformerSimilarityResult,
    SimilarPerformerResult,
)
from .profile import UserProfileBuilder
from .types import (
    EngagementScore,
    EngagementScoringMethod,
    EngagementWeights,
    ProfileInfo,
    RecommendationConfig,
    RecommendationMode,
    RecommendationResponse,
    RecommendationResult,
    SceneDetails,
    SceneEngagementData,
    TimeDecayConfig,
    UserPreferenceProfile,
)

__all__ = [
    "EngagementCalculator",
    "EngagementScore",
    "EngagementScoringMethod",
    "EngagementWeights",
    "PerformerData",
    "PerformerDescriptionConfig",
    "PerformerDetails",
    "PerformerEmbeddingConfig",
    "PerformerEmbeddingRecord",
    "PerformerEngagementData",
    "PerformerProfileBuilder",
    "PerformerProfileInfo",
    "PerformerSceneData",
    "PerformerSimilarityResult",
    "ProfileInfo",
    "RecommendationConfig",
    "RecommendationEngine",
    "RecommendationMode",
    "RecommendationResponse",
    "RecommendationResult",
    "SceneDetails",
    "SceneEngagementData",
    "SimilarPerformerResult",
    "TimeDecayConfig",
    "UserPreferenceProfile",
    "UserProfileBuilder",
]
