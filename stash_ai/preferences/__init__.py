"""Preference learning system for scene ranking via pairwise comparisons.

Uses Bayesian Bradley-Terry model with embedding-aware active pair selection
to efficiently learn user preferences from swipe/comparison interactions.
"""

from stash_ai.preferences.model import BayesianPreferenceModel
from stash_ai.preferences.pair_selector import PairSelectionEngine
from stash_ai.preferences.session import PreferenceSessionManager
from stash_ai.preferences.types import (
    ComparisonPhase,
    ConvergenceMetrics,
    PreferenceComparison,
    PreferenceModelState,
    PreferencePair,
    PreferenceSessionConfig,
    PreferenceSessionData,
    PreferenceSignal,
    PreferenceTrainerResponse,
    SwipeDirection,
)

__all__ = [
    "BayesianPreferenceModel",
    "ComparisonPhase",
    "ConvergenceMetrics",
    "PairSelectionEngine",
    "PreferenceComparison",
    "PreferenceModelState",
    "PreferencePair",
    "PreferenceSessionConfig",
    "PreferenceSessionData",
    "PreferenceSessionManager",
    "PreferenceSignal",
    "PreferenceTrainerResponse",
    "SwipeDirection",
]
