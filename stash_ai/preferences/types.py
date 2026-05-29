"""Type definitions for the preference learning system."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Phase threshold defaults (fallback when scene count is unknown)
# ---------------------------------------------------------------------------

_PHASE_BROAD_MIN: int = 10
"""Absolute minimum comparisons for BROAD phase."""

_PHASE_REFINE_MIN: int = 28
"""Absolute minimum comparisons for REFINE phase."""

_PHASE_BROAD_SCALE: float = 4.0
"""Multiplier for sqrt(n_scenes) to compute BROAD threshold.

For a 12,800 scene library: sqrt(12800) * 4.0 = ~450 comparisons.
This ensures extensive exploration (~3.5% of library) before focusing.
"""

_PHASE_REFINE_SCALE: float = 6.0
"""Multiplier for sqrt(n_scenes) to compute REFINE threshold.

For a 12,800 scene library: sqrt(12800) * 6.0 = ~680 comparisons.
Refining phase runs for ~230 additional comparisons after BROAD.
"""


def compute_phase_thresholds(n_scenes: int = 0) -> tuple[int, int]:
    """Compute dynamic phase thresholds scaled by embedding count.

    For small libraries the minimums apply.  For large libraries the
    thresholds grow with ``sqrt(n_scenes)`` so the model has enough
    exploration budget before transitioning.

    Args:
        n_scenes: Number of embedded scenes.  Pass 0 for static defaults.

    Returns:
        ``(broad_max, refine_max)`` comparison counts.
    """
    if n_scenes <= 0:
        return _PHASE_BROAD_MIN, _PHASE_REFINE_MIN
    root = math.sqrt(n_scenes)
    broad = max(_PHASE_BROAD_MIN, int(root * _PHASE_BROAD_SCALE))
    refine = max(_PHASE_REFINE_MIN, int(root * _PHASE_REFINE_SCALE))
    return broad, refine


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SwipeDirection(Enum):
    """User action on a presented scene or pair."""

    LIKE = "like"
    DISLIKE = "dislike"
    SUPER_LIKE = "super_like"
    SKIP = "skip"


class ComparisonPhase(Enum):
    """Which phase of the progressive refinement protocol."""

    BROAD = "broad"  # Phase 1: inter-cluster representatives
    REFINE = "refine"  # Phase 2: intra-cluster similar pairs
    BOUNDARY = "boundary"  # Phase 3: uncertainty sampling at decision line


class PreferenceSignal(Enum):
    """Signal strength derived from swipe direction."""

    STRONG_POSITIVE = "strong_positive"  # super_like (3x weight)
    POSITIVE = "positive"  # like (1x weight)
    NEGATIVE = "negative"  # dislike (-1x weight)
    NEUTRAL = "neutral"  # skip (no signal)


# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------


@dataclass
class PreferenceSessionConfig:
    """Configuration for a preference training session."""

    mode: str = "swipe"  # 'swipe' | 'tournament' | 'quick_rate'
    phase: ComparisonPhase = ComparisonPhase.BROAD
    batch_size: int = 20  # Scenes per session
    model_key: str = "siglip"
    seed_scene_id: int | None = None  # Optional: bias pairs toward this scene
    exploration_rate: float = 0.2  # Epsilon for explore/exploit
    pure_random: bool = False  # Skip clustering, use uniform random sampling


# ---------------------------------------------------------------------------
# Comparison records (stored in DB)
# ---------------------------------------------------------------------------


class PreferenceComparison:
    """A single pairwise comparison record (stored in DB)."""

    __slots__ = (
        "created_at",
        "id",
        "model_key",
        "phase",
        "response_time_ms",
        "scene_a_id",
        "scene_b_id",
        "session_id",
        "winner_id",
    )

    def __init__(
        self,
        *,
        id: int | None = None,
        scene_a_id: int,
        scene_b_id: int,
        winner_id: int,
        phase: str,
        response_time_ms: int | None = None,
        session_id: str,
        model_key: str,
        created_at: str,
    ) -> None:
        self.id = id
        self.scene_a_id = scene_a_id
        self.scene_b_id = scene_b_id
        self.winner_id = winner_id
        self.phase = phase
        self.response_time_ms = response_time_ms
        self.session_id = session_id
        self.model_key = model_key
        self.created_at = created_at


# ---------------------------------------------------------------------------
# Model state (serialized to DB)
# ---------------------------------------------------------------------------


@dataclass
class PreferenceModelState:
    """Serializable state of the Bayesian preference model."""

    model_key: str
    preference_mean: NDArray[np.float32]  # (dims,)
    preference_covariance_diag: NDArray[np.float32]  # (dims,) diagonal approx
    n_comparisons: int
    noise_variance: float
    phase: str  # Current phase name
    updated_at: str  # ISO datetime


# ---------------------------------------------------------------------------
# Session data (stored in DB)
# ---------------------------------------------------------------------------


@dataclass
class PreferenceSessionData:
    """Metadata for a preference training session."""

    session_id: str
    started_at: str
    completed_at: str | None = None
    comparison_count: int = 0
    phase: str = "broad"
    convergence_avg_sigma: float | None = None


# ---------------------------------------------------------------------------
# Pair selection outputs
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    """A pair of scenes selected for comparison."""

    scene_a_id: int
    scene_b_id: int
    phase: ComparisonPhase
    predicted_probability: float = 0.5  # P(a preferred over b)
    information_gain: float = 0.0  # Expected info gain from this pair
    similarity: float = 0.0  # Cosine similarity between the two


# ---------------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceMetrics:
    """Track how well the preference model has converged."""

    avg_sigma: float  # Mean uncertainty across all scenes
    max_sigma_top50: float  # Worst uncertainty in top-50
    n_comparisons: int  # Total comparisons so far
    phase: ComparisonPhase  # Current phase

    @property
    def confidence_pct(self) -> float:
        """Overall confidence as a percentage (0-100).

        Combines two signals:
        - **Sigma factor**: avg_sigma maps from [3.0, 0.1] to [0%, 100%].
          Widened from the previous [3.0, 0.5] range so the model needs
          genuinely tight posteriors to reach high confidence.
        - **Count factor**: dampens confidence until ~50 comparisons have
          been recorded.  This prevents a handful of swipes from claiming
          near-100% confidence just because avg_sigma drops quickly in
          high-dimensional space.
        """
        # Sigma component: [3.0, 0.1] -> [0, 1]
        sigma_factor = max(0.0, min(1.0, (3.0 - self.avg_sigma) / 2.9))
        # Comparison-count dampener: need ~50 comparisons for full credit
        count_factor = min(1.0, self.n_comparisons / 50)
        return round(sigma_factor * count_factor * 100, 1)

    @property
    def is_converged(self) -> bool:
        """Whether the top-50 ranking is reliable enough to stop."""
        return self.max_sigma_top50 < 1.5 and self.avg_sigma < 2.5


# ---------------------------------------------------------------------------
# API response types
# ---------------------------------------------------------------------------


@dataclass
class PreferenceTrainerResponse:
    """Response from the preference trainer task."""

    status: str  # 'ready' | 'complete' | 'error'
    session_id: str
    pairs: list[PreferencePairData] = field(default_factory=list)
    convergence: ConvergenceData | None = None
    phase: str = "broad"
    n_comparisons: int = 0
    error: str | None = None
    model_surprise: float | None = None  # 0-1, how surprised the model was
    taste_profile: list[dict[str, object]] | None = None  # top likes/dislikes from tag embeddings


@dataclass
class PreferencePairData:
    """Serializable pair data for the frontend."""

    scene_a_id: int
    scene_b_id: int
    phase: str
    predicted_probability: float
    scene_a: dict[str, object] | None = None  # Scene details
    scene_b: dict[str, object] | None = None  # Scene details
    surprise_if_liked: float | None = None  # 0-1, pre-computed for instant toast
    surprise_if_disliked: float | None = None  # 0-1, pre-computed for instant toast


@dataclass
class ConvergenceData:
    """Serializable convergence metrics for the frontend."""

    confidence_pct: float
    n_comparisons: int
    phase: str
    is_converged: bool
    phase_progress_pct: float = 0.0  # 0-100, progress within current phase
