"""Preference session manager - orchestrates training sessions with DB persistence.

Manages the full lifecycle of preference training:
    1. Start session -> generate initial pairs
    2. Record swipe/comparison -> update model -> select next pairs
    3. End session -> save model state -> compute convergence

Results are saved to ``assets/preference_trainer_{session_id}.json``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from stash_ai.embeddings.storage import EmbeddingStorage
from stash_ai.embeddings.tag_vocabulary import TagVocabulary
from stash_ai.preferences.model import BayesianPreferenceModel
from stash_ai.preferences.pair_selector import ClusterInfo, PairSelectionEngine
from stash_ai.preferences.types import (
    ComparisonPhase,
    ConvergenceData,
    PreferenceComparison,
    PreferencePair,
    PreferencePairData,
    PreferenceSessionConfig,
    PreferenceSessionData,
    PreferenceTrainerResponse,
    SwipeDirection,
    compute_phase_thresholds,
)

if TYPE_CHECKING:
    from ..stash_client import StashClient


# ---------------------------------------------------------------------------
# Response-time -> signal-weight mapping
# ---------------------------------------------------------------------------

_RT_VERY_FAST_MS: int = 1000
_RT_FAST_MS: int = 3000
_RT_MEDIUM_MS: int = 5000

_WEIGHT_VERY_FAST: float = 1.5
_WEIGHT_FAST: float = 1.0
_WEIGHT_MEDIUM: float = 0.7
_WEIGHT_SLOW: float = 0.5


# ---------------------------------------------------------------------------
# Swipe direction -> signal weight multipliers
# ---------------------------------------------------------------------------

_SURPRISE_Z_AMPLIFICATION: float = 1.5

_SWIPE_SIGNAL_WEIGHTS: dict[SwipeDirection, float] = {
    SwipeDirection.SUPER_LIKE: 3.0,
    SwipeDirection.LIKE: 1.0,
    SwipeDirection.DISLIKE: 2.0,  # Boosted: comparing against mu is structurally
    # biased — the model always predicts P(mu beats scene) > 0.5, so dislike
    # gradients are always < 0.5.  The 2x multiplier compensates so dislikes
    # push mu away as strongly as likes pull it toward.
    SwipeDirection.SKIP: 0.0,
}


def _dataclass_to_dict(obj: object) -> Any:
    """Recursively convert a dataclass (or list/dict of dataclasses) to a dict.

    Falls back to the raw value for non-dataclass objects so that the result
    is JSON-serializable.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}  # type: ignore[call-overload]
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


class PreferenceSessionManager:
    """Manages preference training sessions with database persistence.

    Orchestrates the full lifecycle:

    1. ``start_session``  -- generate initial pairs
    2. ``record_comparison`` / ``record_swipe`` -- update model, select next pairs
    3. ``end_session``  -- save model state, compute convergence

    Results are saved to ``assets/preference_trainer_{session_id}.json``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        storage: EmbeddingStorage,
        stash: StashClient,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        model_key: str = "siglip",
    ) -> None:
        self.storage = storage
        self.stash = stash
        self.log = log_callback or (lambda _msg, _level: None)
        self.progress = progress_callback or (lambda _cur, _total: None)
        self.model_key = model_key

        # Subsampled frame embeddings per scene: (K, dims) arrays.
        # Loaded with k=16 at init for fast startup; individual scenes
        # are upgraded to full frames on demand via _get_best_frame().
        self._scene_frames: dict[int, NDArray[np.float32]] = {}

        # Scenes whose _scene_frames entry contains ALL frames (not subsampled).
        self._fully_loaded_scenes: set[int] = set()

        # Best-frame representative per scene (single dims-dim vector).
        # This keeps the existing dict[int, NDArray] interface for pair_selector.
        self._scene_embeddings: dict[int, NDArray[np.float32]] = {}

        self._load_scene_embeddings()

        # Load previously compared pairs from DB.
        self._compared_pairs: set[tuple[int, int]] = self._load_compared_pairs()

        # Load scene IDs that have already been swiped on (swipe mode).
        # These are excluded from future pair selection to avoid repetition.
        self._swiped_scene_ids: set[int] = self._load_swiped_scene_ids()

        # Load or create the Bayesian preference model.
        self._model: BayesianPreferenceModel = self._load_model()

        # Now that the model is loaded, recompute best-frame representatives
        # using the learned preference vector (initial load used middle frames).
        if self._model.n_comparisons > 0:
            self._refresh_scene_representatives()

        # Active sessions tracked in memory (session_id -> data).
        self._sessions: dict[str, PreferenceSessionData] = {}

        # Session settings (updated when start_session is called).
        self._exploration_rate: float = 0.2
        self._pure_random: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(
        self,
        config: PreferenceSessionConfig,
    ) -> PreferenceTrainerResponse:
        """Start a new preference training session.

        Generates a unique session ID, determines the current learning phase,
        selects an initial batch of pairs, enriches them with scene details,
        persists the session to the DB, and returns the response.
        """
        session_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # Store session settings for use in subsequent calls (record_swipe).
        self._exploration_rate = config.exploration_rate
        self._pure_random = config.pure_random

        phase = self._determine_phase()

        # Build cluster info for BROAD phase if available (skip if pure_random).
        clusters: list[ClusterInfo] | None = None
        if phase == ComparisonPhase.BROAD and not config.pure_random:
            clusters = self._load_clusters()

        # Select individual scenes for swipe mode (not pairs).
        eligible = self._eligible_embeddings()
        selector = PairSelectionEngine(
            model=self._model,
            embeddings=eligible,
            log_callback=self.log,
        )
        scene_ids = selector.select_swipe_scenes(
            n_scenes=config.batch_size,
            phase=phase,
            clusters=clusters,
            exploration_rate=config.exploration_rate,
            pure_random=config.pure_random,
        )

        # Mark all selected scenes as swiped so they won't appear in
        # future batches (when fetchMorePairs triggers a new session).
        for sid in scene_ids:
            self._swiped_scene_ids.add(sid)

        # Enrich with scene details.
        scene_details = self._get_scene_details(scene_ids)

        # Convert to API response format (scene_b_id=-1 for swipe mode).
        pair_data_list = self._scenes_to_swipe_data(scene_ids, scene_details, phase)

        # Build convergence snapshot.
        convergence = self._build_convergence_data(phase)

        # Persist session in DB.
        session_data = PreferenceSessionData(
            session_id=session_id,
            started_at=now_iso,
            completed_at=None,
            comparison_count=0,
            phase=phase.value,
            convergence_avg_sigma=(convergence.confidence_pct if convergence else None),
        )
        self._save_session_to_db(session_data)
        self._sessions[session_id] = session_data

        response = PreferenceTrainerResponse(
            status="ready",
            session_id=session_id,
            pairs=pair_data_list,
            convergence=convergence,
            phase=phase.value,
            n_comparisons=self._model.n_comparisons,
        )

        self.log(
            f"Session {session_id[:8]} started: phase={phase.value}, "
            f"scenes={len(scene_ids)}, total_comparisons={self._model.n_comparisons}, "
            f"swiped_excluded={len(self._swiped_scene_ids)}, "
            f"eligible_scenes={len(eligible)}",
            "info",
        )
        return response

    def record_comparison(
        self,
        session_id: str,
        scene_a_id: int,
        scene_b_id: int,
        winner_id: int,
        response_time_ms: int | None = None,
    ) -> PreferenceTrainerResponse:
        """Record a pairwise comparison and return updated state.

        Updates the Bayesian model, stores the comparison in the DB,
        selects 1-2 replacement pairs, and returns the updated response.
        """
        phase = self._determine_phase()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Validate embeddings exist and get best-frame representatives.
        emb_a = self._get_best_frame(scene_a_id)
        emb_b = self._get_best_frame(scene_b_id)
        if emb_a is None or emb_b is None:
            missing: list[str] = []
            if emb_a is None:
                missing.append(str(scene_a_id))
            if emb_b is None:
                missing.append(str(scene_b_id))
            return PreferenceTrainerResponse(
                status="error",
                session_id=session_id,
                phase=phase.value,
                n_comparisons=self._model.n_comparisons,
                error=f"Missing embeddings for scene(s): {', '.join(missing)}",
            )

        # Determine winner / loser embeddings.
        if winner_id == scene_a_id:
            e_winner, e_loser = emb_a, emb_b
        elif winner_id == scene_b_id:
            e_winner, e_loser = emb_b, emb_a
        else:
            return PreferenceTrainerResponse(
                status="error",
                session_id=session_id,
                phase=phase.value,
                n_comparisons=self._model.n_comparisons,
                error=(
                    f"winner_id {winner_id} must be one of "
                    f"scene_a_id ({scene_a_id}) or scene_b_id ({scene_b_id})"
                ),
            )

        signal_weight = self._signal_weight_from_response_time(response_time_ms)

        # Update Bayesian model.
        self._model.update(e_winner, e_loser, signal_weight)

        # Refresh best-frame representatives for both scenes after mu changed.
        for sid in (scene_a_id, scene_b_id):
            frames = self._scene_frames.get(sid)
            if frames is not None:
                new_scores = frames @ self._model.mu
                self._scene_embeddings[sid] = frames[int(np.argmax(new_scores))]

        # Record the canonical pair order for deduplication.
        canonical = (min(scene_a_id, scene_b_id), max(scene_a_id, scene_b_id))
        self._compared_pairs.add(canonical)

        # Persist comparison to DB.
        comparison = PreferenceComparison(
            scene_a_id=scene_a_id,
            scene_b_id=scene_b_id,
            winner_id=winner_id,
            phase=phase.value,
            response_time_ms=response_time_ms,
            session_id=session_id,
            model_key=self.model_key,
            created_at=now_iso,
        )
        self._save_comparison_to_db(comparison)

        # Update session metadata.
        self._increment_session_count(session_id)

        # Save model state.
        self._save_model()

        # Select 1-2 replacement pairs (excluding already-swiped scenes).
        new_phase = self._determine_phase()
        clusters = self._load_clusters() if new_phase == ComparisonPhase.BROAD else None
        eligible = self._eligible_embeddings()
        selector = PairSelectionEngine(
            model=self._model,
            embeddings=eligible,
            log_callback=self.log,
        )
        replacement_pairs = selector.select_pairs(
            n_pairs=2,
            phase=new_phase,
            clusters=clusters,
            compared_pairs=self._compared_pairs,
            exploration_rate=0.2,
        )

        # Enrich replacement pairs.
        replacement_scene_ids: list[int] = []
        for p in replacement_pairs:
            replacement_scene_ids.append(p.scene_a_id)
            replacement_scene_ids.append(p.scene_b_id)
        replacement_details = self._get_scene_details(list(set(replacement_scene_ids)))
        pair_data_list = self._pairs_to_data(replacement_pairs, replacement_details)

        convergence = self._build_convergence_data(new_phase)

        response = PreferenceTrainerResponse(
            status="ready",
            session_id=session_id,
            pairs=pair_data_list,
            convergence=convergence,
            phase=new_phase.value,
            n_comparisons=self._model.n_comparisons,
        )

        return response

    def record_swipe(
        self,
        session_id: str,
        scene_id: int,
        direction: SwipeDirection,
        response_time_ms: int | None = None,
    ) -> PreferenceTrainerResponse:
        """Record a single-scene swipe (Tinder mode).

        Swipe semantics:

        * **LIKE / SUPER_LIKE**: scene beats a synthetic opponent (the
          current preference mean).  Treated as a positive comparison.
        * **DISLIKE**: preference mean beats the scene.  Negative update.
        * **SKIP**: no model update; only recorded for history.
        """
        phase = self._determine_phase()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Get best-frame embedding for the swiped scene.
        emb = self._get_best_frame(scene_id)
        if emb is None:
            return PreferenceTrainerResponse(
                status="error",
                session_id=session_id,
                phase=phase.value,
                n_comparisons=self._model.n_comparisons,
                error=f"Missing embedding for scene {scene_id}",
            )

        signal_weight = self._signal_weight_from_response_time(response_time_ms)
        swipe_multiplier = _SWIPE_SIGNAL_WEIGHTS.get(direction, 0.0)
        combined_weight = signal_weight * swipe_multiplier

        # Compute model's predicted score AND uncertainty BEFORE the update.
        # Use best-frame scoring: max over frames for prediction.
        frames = self._scene_frames.get(scene_id)
        if frames is not None and self._model.n_comparisons > 0:
            frame_scores = frames @ self._model.mu
            best_idx = int(np.argmax(frame_scores))
            pre_score = float(frame_scores[best_idx])
            best_frame = frames[best_idx]
            pre_sigma = float(np.sqrt(np.sum(self._model.sigma_sq * best_frame**2)))
        else:
            pre_score, pre_sigma = self._model.predict_score(emb)

        # Construct a synthetic comparison against the preference mean.
        preference_mean = self._model.mu

        if direction in (SwipeDirection.LIKE, SwipeDirection.SUPER_LIKE):
            # Scene beats the mean -> scene is "winner", mean is "loser".
            self._model.update(emb, preference_mean, combined_weight)
            winner_id = scene_id
        elif direction == SwipeDirection.DISLIKE:
            # Mean beats the scene -> mean is "winner", scene is "loser".
            self._model.update(preference_mean, emb, combined_weight)
            winner_id = -1  # Sentinel: mean won
        else:
            # SKIP -- no model update.
            winner_id = 0  # Sentinel: skipped

        # Refresh this scene's representative after model update.
        if direction != SwipeDirection.SKIP and frames is not None:
            new_scores = frames @ self._model.mu
            self._scene_embeddings[scene_id] = frames[int(np.argmax(new_scores))]

        # Persist comparison record (scene_b_id = -1 for synthetic opponent).
        comparison = PreferenceComparison(
            scene_a_id=scene_id,
            scene_b_id=-1,  # Synthetic opponent (preference mean)
            winner_id=winner_id,
            phase=phase.value,
            response_time_ms=response_time_ms,
            session_id=session_id,
            model_key=self.model_key,
            created_at=now_iso,
        )
        self._save_comparison_to_db(comparison)
        self._increment_session_count(session_id)

        if direction != SwipeDirection.SKIP:
            self._save_model()

        # Track this scene as swiped so it won't be shown again.
        self._swiped_scene_ids.add(scene_id)

        # Select next scene to present (excluding already-swiped scenes).
        new_phase = self._determine_phase()
        clusters: list[ClusterInfo] | None = None
        if new_phase == ComparisonPhase.BROAD and not self._pure_random:
            clusters = self._load_clusters()
        eligible = self._eligible_embeddings()
        selector = PairSelectionEngine(
            model=self._model,
            embeddings=eligible,
            log_callback=self.log,
        )
        replacement_ids = selector.select_swipe_scenes(
            n_scenes=1,
            phase=new_phase,
            clusters=clusters,
            exploration_rate=self._exploration_rate,
            pure_random=self._pure_random,
        )

        # Mark replacement as swiped to prevent future repeats.
        for sid in replacement_ids:
            self._swiped_scene_ids.add(sid)

        replacement_details = self._get_scene_details(replacement_ids)
        pair_data_list = self._scenes_to_swipe_data(
            replacement_ids,
            replacement_details,
            new_phase,
        )

        convergence = self._build_convergence_data(new_phase)

        # Compute surprise: how much the swipe contradicted the model's
        # prediction.  Uses z-score (score / sigma) so that surprise is
        # suppressed when the model is uncertain and amplified when it
        # is genuinely confident and wrong.
        model_surprise: float | None = None
        if direction != SwipeDirection.SKIP:
            liked = direction in (SwipeDirection.LIKE, SwipeDirection.SUPER_LIKE)
            z = pre_score / max(pre_sigma, 1e-6)
            # If the user liked it, surprise is how negative z was.
            # If the user disliked it, surprise is how positive z was.
            raw_surprise_z = -z if liked else z
            model_surprise = round(
                float(1.0 / (1.0 + np.exp(-raw_surprise_z * _SURPRISE_Z_AMPLIFICATION))), 3
            )

        response = PreferenceTrainerResponse(
            status="ready",
            session_id=session_id,
            pairs=pair_data_list,
            convergence=convergence,
            phase=new_phase.value,
            n_comparisons=self._model.n_comparisons,
            model_surprise=model_surprise,
        )

        return response

    def end_session(self, session_id: str) -> PreferenceTrainerResponse:
        """End a preference training session.

        Marks the session as completed, computes final convergence metrics,
        saves model state, and returns the final response.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        phase = self._determine_phase()
        convergence = self._build_convergence_data(phase)

        # Update session in DB with completion timestamp.
        self._complete_session_in_db(
            session_id,
            completed_at=now_iso,
            convergence_avg_sigma=(convergence.confidence_pct if convergence else None),
        )

        # Ensure model is saved.
        self._save_model()

        # Clean up in-memory session tracking.
        self._sessions.pop(session_id, None)

        response = PreferenceTrainerResponse(
            status="complete",
            session_id=session_id,
            pairs=[],
            convergence=convergence,
            phase=phase.value,
            n_comparisons=self._model.n_comparisons,
            taste_profile=self._build_taste_profile(),
        )

        self.log(
            f"Session {session_id[:8]} ended: "
            f"comparisons={self._model.n_comparisons}, phase={phase.value}",
            "info",
        )
        return response

    def get_session_state(self, session_id: str) -> PreferenceTrainerResponse:
        """Return the current session state without making any changes."""
        phase = self._determine_phase()
        convergence = self._build_convergence_data(phase)

        return PreferenceTrainerResponse(
            status="ready",
            session_id=session_id,
            pairs=[],
            convergence=convergence,
            phase=phase.value,
            n_comparisons=self._model.n_comparisons,
        )

    def get_model_stats(self) -> PreferenceTrainerResponse:
        """Return current model stats without creating a session.

        Used by the frontend to display stats on the Train tab intro screen
        without side effects (no session created, no DB writes).
        """
        phase = self._determine_phase()
        convergence = self._build_convergence_data(phase)

        return PreferenceTrainerResponse(
            status="stats",
            session_id="",
            pairs=[],
            convergence=convergence,
            phase=phase.value,
            n_comparisons=self._model.n_comparisons,
            taste_profile=self._build_taste_profile(),
        )

    def reset_model(self) -> PreferenceTrainerResponse:
        """Delete all learned preference data for the current model_key.

        Removes:
          - ``preference_model_state`` row for this model_key
          - All ``preference_comparisons`` rows for this model_key
          - All ``preference_sessions`` that only contain comparisons
            from this model_key

        The in-memory model is replaced with a fresh warm-started instance.
        """
        conn = self.storage._get_connection()
        try:
            cursor = conn.cursor()

            # Find sessions that belong exclusively to this model_key.
            cursor.execute(
                """
                SELECT DISTINCT session_id
                FROM preference_comparisons
                WHERE model_key = ?
                """,
                (self.model_key,),
            )
            session_ids = [row["session_id"] for row in cursor.fetchall()]

            # Delete comparisons for this model_key.
            cursor.execute(
                "DELETE FROM preference_comparisons WHERE model_key = ?",
                (self.model_key,),
            )
            deleted_comparisons = cursor.rowcount

            # Delete sessions that no longer have any comparisons.
            deleted_sessions = 0
            for sid in session_ids:
                cursor.execute(
                    "SELECT COUNT(*) AS cnt FROM preference_comparisons WHERE session_id = ?",
                    (sid,),
                )
                if cursor.fetchone()["cnt"] == 0:
                    cursor.execute(
                        "DELETE FROM preference_sessions WHERE session_id = ?",
                        (sid,),
                    )
                    deleted_sessions += cursor.rowcount

            # Delete model state for this model_key.
            cursor.execute(
                "DELETE FROM preference_model_state WHERE model_key = ?",
                (self.model_key,),
            )

            conn.commit()
        finally:
            conn.close()

        self.log(
            f"Reset preference model '{self.model_key}': "
            f"removed {deleted_comparisons} comparisons, "
            f"{deleted_sessions} sessions",
            "info",
        )

        # Clear in-memory state.
        self._compared_pairs.clear()
        self._swiped_scene_ids.clear()

        # Re-create a fresh model with engagement warm-start.
        dims = self._infer_embedding_dims()
        if dims == 0:
            dims = 768
        self._model = BayesianPreferenceModel(dims=dims)
        self._warm_start_model(self._model)

        phase = self._determine_phase()
        convergence = self._build_convergence_data(phase)

        return PreferenceTrainerResponse(
            status="reset",
            session_id="",
            pairs=[],
            convergence=convergence,
            phase=phase.value,
            n_comparisons=0,
        )

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _load_model(self) -> BayesianPreferenceModel:
        """Load model from DB or create a fresh one with engagement warm-start.

        If a persisted model state exists in ``preference_model_state``,
        it is restored.  Otherwise a new model is initialised and optionally
        warm-started from engagement data.
        """
        conn = self.storage._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT preference_mean, preference_covariance_diag,
                       n_comparisons, noise_variance, phase
                FROM preference_model_state
                WHERE model_key = ?
                """,
                (self.model_key,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is not None:
            mean = np.array(
                self.storage._unpack_embedding(row["preference_mean"]),
                dtype=np.float32,
            )
            cov_diag = np.array(
                self.storage._unpack_embedding(row["preference_covariance_diag"]),
                dtype=np.float32,
            )
            model = BayesianPreferenceModel(
                dims=len(mean),
                noise_variance=row["noise_variance"],
            )
            model.mu = mean
            model.sigma_sq = cov_diag
            model.n_comparisons = row["n_comparisons"]
            self.log(
                f"Loaded preference model: dims={len(mean)}, "
                f"n_comparisons={row['n_comparisons']}, phase={row['phase']}",
                "info",
            )
            return model

        # No persisted state -- create from scratch.
        dims = self._infer_embedding_dims()
        if dims == 0:
            self.log(
                "No embeddings found; creating placeholder model with 768 dims",
                "warning",
            )
            dims = 768

        model = BayesianPreferenceModel(dims=dims)

        # Attempt warm-start from engagement data.
        self._warm_start_model(model)

        return model

    def _warm_start_model(self, model: BayesianPreferenceModel) -> None:
        """Initialise the model prior from engagement scores via ridge regression.

        Computes:
            mu_0 = (diag(X^T X) + lambda)^{-1} * (X^T y)

        using a diagonal approximation for tractability on high-dimensional
        embeddings, where X is the embedding matrix and y is the normalised
        engagement score vector.
        """
        from stash_ai.recommendations.engagement import EngagementCalculator

        calculator = EngagementCalculator(log_callback=self.log)
        engagement_data = calculator.get_all_scene_engagement()

        if not engagement_data:
            self.log("No engagement data available for warm-start", "debug")
            return

        # Build aligned arrays: X (n_scenes, dims), y (n_scenes,)
        scene_ids_ordered: list[int] = []
        scores: list[float] = []
        embeddings_list: list[NDArray[np.float32]] = []

        for sid, eng_data in engagement_data.items():
            emb = self._scene_embeddings.get(sid)
            if emb is None:
                continue
            score, _ = calculator.calculate_base_score(eng_data)
            if score <= 0:
                continue
            scene_ids_ordered.append(sid)
            scores.append(score)
            embeddings_list.append(emb)

        if len(embeddings_list) < 5:
            self.log(
                f"Insufficient data for warm-start "
                f"({len(embeddings_list)} scenes with engagement + embeddings), "
                f"skipping",
                "debug",
            )
            return

        X = np.array(embeddings_list, dtype=np.float32)  # (n, d)
        y_raw = np.array(scores, dtype=np.float32)  # (n,)

        # Normalise y to zero-mean, unit-variance.
        y_mean = float(np.mean(y_raw))
        y_std = float(np.std(y_raw))
        if y_std < 1e-8:
            self.log(
                "Engagement scores have zero variance; skipping warm-start",
                "debug",
            )
            return
        y = (y_raw - y_mean) / y_std

        # Ridge regression (diagonal approximation):
        #   mu_0 = (X^T y) / (diag(X^T X) + lambda)
        ridge_lambda: float = 1.0
        diag_XtX = np.sum(X * X, axis=0)  # (d,) -- sum of squared features
        XtY = X.T @ y  # (d,)
        mu_0 = XtY / (diag_XtX + ridge_lambda)

        # Normalise to unit vector.
        norm = float(np.linalg.norm(mu_0))
        if norm > 1e-8:
            mu_0 = mu_0 / norm

        # Set initial covariance from diagonal of (X^T X + lambda I)^{-1}.
        sigma_noise_sq = model.noise_variance
        cov_diag_0 = sigma_noise_sq / (diag_XtX + ridge_lambda)

        model.mu = mu_0.astype(np.float32)
        model.sigma_sq = cov_diag_0.astype(np.float32)

        self.log(
            f"Warm-started model from {len(embeddings_list)} engaged scenes "
            f"(mean_score={y_mean:.1f}, std={y_std:.1f})",
            "info",
        )

    def _save_model(self) -> None:
        """Persist the current model state to the DB."""
        now_iso = datetime.now(timezone.utc).isoformat()
        phase = self._determine_phase()

        mean_blob = self.storage._pack_embedding(self._model.mu.tolist())
        cov_blob = self.storage._pack_embedding(self._model.sigma_sq.tolist())

        conn = self.storage._get_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO preference_model_state
                    (model_key, preference_mean, preference_covariance_diag,
                     n_comparisons, noise_variance, phase, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.model_key,
                    mean_blob,
                    cov_blob,
                    self._model.n_comparisons,
                    self._model.noise_variance,
                    phase.value,
                    now_iso,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _load_scene_embeddings(self) -> None:
        """Load frame-level embeddings (preferred) with composite fallback.

        Loads K evenly-spaced frame embeddings per scene from the
        ``frame_embeddings`` table.  Scenes that only have composite
        embeddings (no per-frame data) fall back to treating the composite
        as a single "frame".

        After loading, ``_refresh_scene_representatives()`` picks the
        best-frame per scene for the ``_scene_embeddings`` dict used by
        the pair selector and convergence metrics.
        """
        # 1. Load frame-level embeddings (already numpy float32 arrays).
        self._scene_frames = self.storage.get_all_frame_representatives()

        # 2. Fall back to composite for scenes without frame embeddings.
        all_composites = self.storage.get_all_embeddings()
        fallback_count = 0
        for sid, emb in all_composites:
            if sid not in self._scene_frames:
                arr = np.array(emb, dtype=np.float32)
                self._scene_frames[sid] = arr.reshape(1, -1)
                fallback_count += 1

        # 3. Prune scenes that no longer exist in Stash (deleted by user).
        valid_ids = self._get_valid_scene_ids(set(self._scene_frames.keys()))
        pruned = len(self._scene_frames) - len(valid_ids)
        if pruned > 0:
            self._scene_frames = {
                sid: frames for sid, frames in self._scene_frames.items() if sid in valid_ids
            }

        # 4. Compute best-frame representatives.
        self._refresh_scene_representatives()

        self.log(
            f"Loaded {len(self._scene_frames)} scene embeddings "
            f"({len(self._scene_frames) - fallback_count} frame-level, "
            f"{fallback_count} composite fallback"
            f"{f', {pruned} deleted pruned' if pruned else ''})",
            "debug",
        )

    def _infer_embedding_dims(self) -> int:
        """Infer embedding dimensionality from loaded data."""
        for emb in self._scene_embeddings.values():
            return len(emb)
        return 0

    def _get_valid_scene_ids(self, candidate_ids: set[int]) -> set[int]:
        """Return the subset of candidate_ids that still exist in Stash.

        Queries the Stash SQLite database to check which scenes haven't
        been deleted.  Returns all candidates if the DB is unavailable.
        """
        from stash_ai.tools.database import get_readonly_connection, get_stash_db_path

        db_path = get_stash_db_path()
        if not db_path.exists():
            return candidate_ids

        try:
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(candidate_ids))
            cursor.execute(
                f"SELECT id FROM scenes WHERE id IN ({placeholders})",
                tuple(candidate_ids),
            )
            valid = {int(row["id"]) for row in cursor.fetchall()}
            conn.close()
            return valid
        except Exception:
            return candidate_ids

    def _refresh_scene_representatives(self) -> None:
        """Recompute best-frame representative per scene based on current mu.

        For trained models, picks the frame most aligned with the
        preference vector.  For fresh models (no comparisons yet) or
        during initial construction (before ``_model`` is loaded),
        uses the middle frame to avoid intro/outro bias.
        """
        has_trained_model = hasattr(self, "_model") and self._model.n_comparisons > 0
        for sid, frames in self._scene_frames.items():
            if has_trained_model:
                scores: NDArray[np.float32] = frames @ self._model.mu  # (K,)
                best_idx = int(np.argmax(scores))
            else:
                best_idx = len(frames) // 2
            self._scene_embeddings[sid] = frames[best_idx]

    def _get_best_frame(self, scene_id: int) -> NDArray[np.float32] | None:
        """Get the frame embedding most aligned with current preferences.

        Lazy-loads ALL frame embeddings for the scene on first access
        (the init-time load only kept k=16 subsampled frames for speed).
        Returns the frame with highest |dot(frame, mu)| so that both
        strong likes and strong dislikes select the most informative
        frame for the Bradley-Terry update.

        Falls back to the pre-computed representative if the scene has
        no frame-level data at all.
        """
        frames = self._scene_frames.get(scene_id)

        # Lazy-load full frame set for this scene if we only have
        # the subsampled representatives from init.
        if frames is not None and scene_id not in self._fully_loaded_scenes:
            all_frames = self.storage.get_scene_frames(scene_id)
            if all_frames is not None and len(all_frames) > len(frames):
                frames = all_frames
                self._scene_frames[scene_id] = frames
            self._fully_loaded_scenes.add(scene_id)

        if frames is not None and self._model.n_comparisons > 0:
            scores: NDArray[np.float32] = frames @ self._model.mu
            return cast("NDArray[np.float32]", frames[int(np.argmax(np.abs(scores)))])
        return self._scene_embeddings.get(scene_id)

    # ------------------------------------------------------------------
    # Compared-pairs tracking
    # ------------------------------------------------------------------

    def _load_compared_pairs(self) -> set[tuple[int, int]]:
        """Load all previously compared scene pairs from the DB.

        Returns canonical ``(min_id, max_id)`` tuples for deduplication.
        Excludes synthetic swipe comparisons (``scene_b_id = -1``).
        """
        conn = self.storage._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT scene_a_id, scene_b_id
                FROM preference_comparisons
                WHERE model_key = ? AND scene_b_id >= 0
                """,
                (self.model_key,),
            )
            pairs: set[tuple[int, int]] = set()
            for row in cursor.fetchall():
                a, b = int(row["scene_a_id"]), int(row["scene_b_id"])
                pairs.add((min(a, b), max(a, b)))
            return pairs
        finally:
            conn.close()

    def _load_swiped_scene_ids(self) -> set[int]:
        """Load all scene IDs that have been swiped on from the DB.

        In swipe mode, comparisons are stored with ``scene_b_id = -1``
        (synthetic opponent).  We track these separately so the pair
        selector can exclude already-seen scenes.
        """
        conn = self.storage._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT scene_a_id
                FROM preference_comparisons
                WHERE model_key = ? AND scene_b_id = -1
                """,
                (self.model_key,),
            )
            return {int(row["scene_a_id"]) for row in cursor.fetchall()}
        finally:
            conn.close()

    def _eligible_embeddings(self) -> dict[int, NDArray[np.float32]]:
        """Return embeddings excluding already-swiped scenes.

        Ensures the pair selector only considers scenes the user hasn't
        seen yet in swipe mode.
        """
        if not self._swiped_scene_ids:
            return self._scene_embeddings
        return {
            sid: emb
            for sid, emb in self._scene_embeddings.items()
            if sid not in self._swiped_scene_ids
        }

    # ------------------------------------------------------------------
    # DB persistence helpers
    # ------------------------------------------------------------------

    def _save_comparison_to_db(self, comparison: PreferenceComparison) -> None:
        """Insert a single comparison record into the DB."""
        conn = self.storage._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO preference_comparisons
                    (scene_a_id, scene_b_id, winner_id, phase,
                     response_time_ms, session_id, model_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comparison.scene_a_id,
                    comparison.scene_b_id,
                    comparison.winner_id,
                    comparison.phase,
                    comparison.response_time_ms,
                    comparison.session_id,
                    comparison.model_key,
                    comparison.created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_session_to_db(self, session: PreferenceSessionData) -> None:
        """Insert a new session record into the DB."""
        conn = self.storage._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO preference_sessions
                    (session_id, started_at, completed_at,
                     comparison_count, phase, convergence_avg_sigma)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.started_at,
                    session.completed_at,
                    session.comparison_count,
                    session.phase,
                    session.convergence_avg_sigma,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _increment_session_count(self, session_id: str) -> None:
        """Increment the comparison count for a session in the DB."""
        conn = self.storage._get_connection()
        try:
            conn.execute(
                """
                UPDATE preference_sessions
                SET comparison_count = comparison_count + 1
                WHERE session_id = ?
                """,
                (session_id,),
            )
            conn.commit()
        finally:
            conn.close()

        # Also update in-memory cache.
        if session_id in self._sessions:
            self._sessions[session_id].comparison_count += 1

    def _complete_session_in_db(
        self,
        session_id: str,
        completed_at: str,
        convergence_avg_sigma: float | None,
    ) -> None:
        """Mark a session as completed in the DB."""
        conn = self.storage._get_connection()
        try:
            conn.execute(
                """
                UPDATE preference_sessions
                SET completed_at = ?, convergence_avg_sigma = ?
                WHERE session_id = ?
                """,
                (completed_at, convergence_avg_sigma, session_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Scene details from Stash GraphQL
    # ------------------------------------------------------------------

    def _get_scene_details(self, scene_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch scene details (title, performers, tags, files) from SQLite.

        Uses direct database queries for reliable nested field retrieval
        (performer names, tag names, file metadata).
        """
        if not scene_ids:
            return {}

        from stash_ai.tools.database import get_readonly_connection, get_stash_db_path

        db_path = get_stash_db_path()
        if not db_path.exists():
            self.log(f"Database not found at {db_path}", "warning")
            return {}

        conn = get_readonly_connection(db_path)
        cursor = conn.cursor()

        try:
            placeholders = ",".join("?" * len(scene_ids))
            ids_tuple = tuple(scene_ids)

            # Fetch scene base data.
            cursor.execute(
                f"""
                SELECT s.id, s.title, s.date, s.rating,
                       st.name AS studio_name
                FROM scenes s
                LEFT JOIN studios st ON s.studio_id = st.id
                WHERE s.id IN ({placeholders})
                """,
                ids_tuple,
            )

            details: dict[int, dict[str, Any]] = {}
            for row in cursor.fetchall():
                sid = row["id"]
                details[sid] = {
                    "id": sid,
                    "title": row["title"],
                    "performers": [],
                    "tags": [],
                    "files": [],
                    "rating100": row["rating"],
                    "play_count": 0,
                    "o_counter": 0,
                    "interactive": False,
                    "date": row["date"],
                    "studio": row["studio_name"],
                }

            # Fetch file info.
            cursor.execute(
                f"""
                SELECT sf.scene_id, f.basename, vf.duration,
                       vf.width, vf.height, vf.interactive
                FROM scenes_files sf
                JOIN files f ON sf.file_id = f.id
                JOIN video_files vf ON f.id = vf.file_id
                WHERE sf.scene_id IN ({placeholders}) AND sf."primary" = 1
                """,
                ids_tuple,
            )

            for row in cursor.fetchall():
                sid = row["scene_id"]
                if sid in details:
                    details[sid]["files"].append(
                        {
                            "basename": row["basename"] or "",
                            "duration": row["duration"] or 0,
                            "width": row["width"] or 0,
                            "height": row["height"] or 0,
                        }
                    )
                    details[sid]["interactive"] = bool(row["interactive"])

            # Fill in title from basename when title is empty.
            for sid, scene in details.items():
                if not scene["title"] and scene["files"]:
                    scene["title"] = scene["files"][0].get("basename", f"Scene {sid}")
                elif not scene["title"]:
                    scene["title"] = f"Scene {sid}"

            # Fetch play counts.
            cursor.execute(
                f"""
                SELECT scene_id, COUNT(*) AS play_count
                FROM scenes_view_dates
                WHERE scene_id IN ({placeholders})
                GROUP BY scene_id
                """,
                ids_tuple,
            )

            for row in cursor.fetchall():
                if row["scene_id"] in details:
                    details[row["scene_id"]]["play_count"] = row["play_count"]

            # Fetch o counts.
            cursor.execute(
                f"""
                SELECT scene_id, COUNT(*) AS o_count
                FROM scenes_o_dates
                WHERE scene_id IN ({placeholders})
                GROUP BY scene_id
                """,
                ids_tuple,
            )

            for row in cursor.fetchall():
                if row["scene_id"] in details:
                    details[row["scene_id"]]["o_counter"] = row["o_count"]

            # Fetch performers.
            cursor.execute(
                f"""
                SELECT ps.scene_id, p.id, p.name
                FROM performers_scenes ps
                JOIN performers p ON ps.performer_id = p.id
                WHERE ps.scene_id IN ({placeholders})
                """,
                ids_tuple,
            )

            for row in cursor.fetchall():
                if row["scene_id"] in details:
                    details[row["scene_id"]]["performers"].append(
                        {
                            "id": str(row["id"]),
                            "name": row["name"],
                        }
                    )

            # Fetch tags.
            cursor.execute(
                f"""
                SELECT st.scene_id, t.id, t.name
                FROM scenes_tags st
                JOIN tags t ON st.tag_id = t.id
                WHERE st.scene_id IN ({placeholders})
                """,
                ids_tuple,
            )

            for row in cursor.fetchall():
                if row["scene_id"] in details:
                    details[row["scene_id"]]["tags"].append(
                        {
                            "id": str(row["id"]),
                            "name": row["name"],
                        }
                    )

        except Exception as e:
            self.log(f"Failed to fetch scene details: {e}", "warning")
        finally:
            conn.close()

        return details

    # ------------------------------------------------------------------
    # Phase determination
    # ------------------------------------------------------------------

    def _determine_phase(self) -> ComparisonPhase:
        """Auto-determine the current learning phase based on total comparisons.

        Thresholds scale with ``sqrt(n_scenes)`` via
        :func:`compute_phase_thresholds` so larger libraries get more
        exploration before transitioning.
        """
        n = self._model.n_comparisons
        n_scenes = len(self._scene_embeddings)
        broad_max, refine_max = compute_phase_thresholds(n_scenes)
        if n < broad_max:
            return ComparisonPhase.BROAD
        if n < refine_max:
            return ComparisonPhase.REFINE
        return ComparisonPhase.BOUNDARY

    # ------------------------------------------------------------------
    # Cluster loading
    # ------------------------------------------------------------------

    def _load_clusters(self) -> list[ClusterInfo] | None:
        """Load taste-map clusters from storage for BROAD-phase pair selection.

        Returns ``None`` if no clusters are available (taste map not yet built).
        """
        try:
            raw_clusters = self.storage.get_taste_clusters(self.model_key)
            if not raw_clusters:
                self.log(
                    "No taste clusters found; BROAD phase will use random selection",
                    "debug",
                )
                return None

            cluster_infos: list[ClusterInfo] = []
            for c in raw_clusters:
                if c.get("excluded"):
                    continue
                cluster_infos.append(
                    ClusterInfo(
                        cluster_id=c["cluster_id"],
                        centroid=np.array(c["centroid"], dtype=np.float32),
                        scene_ids=c["scene_ids"],
                        engagement_share=c.get("engagement_share", 0.0),
                    )
                )
            self.log(f"Loaded {len(cluster_infos)} taste clusters", "debug")
            return cluster_infos if cluster_infos else None
        except Exception as e:
            self.log(f"Failed to load clusters: {e}", "warning")
            return None

    # ------------------------------------------------------------------
    # Response-time mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_weight_from_response_time(
        response_time_ms: int | None,
    ) -> float:
        """Map response time to a signal confidence weight.

        Faster decisions indicate stronger gut reactions and therefore
        stronger preference signals.

        Returns:
            Weight multiplier:
                * < 1 s  -> 1.5
                * 1-3 s  -> 1.0
                * 3-5 s  -> 0.7
                * > 5 s  -> 0.5
        """
        if response_time_ms is None:
            return _WEIGHT_FAST  # Default to "clear preference"
        if response_time_ms < _RT_VERY_FAST_MS:
            return _WEIGHT_VERY_FAST
        if response_time_ms < _RT_FAST_MS:
            return _WEIGHT_FAST
        if response_time_ms < _RT_MEDIUM_MS:
            return _WEIGHT_MEDIUM
        return _WEIGHT_SLOW

    # ------------------------------------------------------------------
    # Taste profile
    # ------------------------------------------------------------------

    def _build_taste_profile(self, top_k: int = 5) -> list[dict[str, object]] | None:
        """Project the preference vector onto tag embeddings for human-readable labels.

        Computes cosine similarity between the learned preference mean (mu)
        and all tag embeddings.  Returns the top-k most aligned tags as
        "likes" (positive scores) and the top-k least aligned as "dislikes"
        (negative scores).

        Returns ``None`` if the model has not been trained yet.
        """
        if self._model.n_comparisons == 0:
            return None

        tag_vocab = TagVocabulary(self.storage, self.model_key, self.log)

        # Top-k likes: tags most aligned with preference direction
        likes = tag_vocab.match_cluster_centroid(self._model.mu, top_k=top_k)

        # Top-k dislikes: pass negated mu to find tags in the opposite direction
        negated_mu = -self._model.mu
        dislikes_raw = tag_vocab.match_cluster_centroid(negated_mu, top_k=top_k)

        if not likes and not dislikes_raw:
            return None

        result: list[dict[str, object]] = []
        like_texts: set[str] = set()

        for entry in likes:
            like_texts.add(str(entry["text"]))
            result.append(
                {
                    "text": entry["text"],
                    "score": entry["similarity"],
                    "source": entry["source"],
                }
            )

        for entry in dislikes_raw:
            if str(entry["text"]) in like_texts:
                continue
            # Force negative score for dislikes.  match_cluster_centroid(-mu)
            # returns cos(-mu, tag) = -cos(mu, tag).  In embedding spaces where
            # all similarities are positive, these values are negative and simply
            # negating them produces a positive number — hiding dislikes from the
            # frontend which filters on score < 0.  Using -abs() guarantees a
            # negative score regardless of the embedding-space geometry.
            result.append(
                {
                    "text": entry["text"],
                    "score": -abs(cast("float", entry["similarity"])),
                    "source": entry["source"],
                }
            )

        return result

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    def _build_convergence_data(self, phase: ComparisonPhase) -> ConvergenceData | None:
        """Build a convergence snapshot from the current model state.

        Returns ``None`` if no embeddings have been loaded yet.
        """
        if not self._scene_embeddings:
            return None

        n = self._model.n_comparisons
        phase_progress = self._compute_phase_progress(phase, n)

        try:
            metrics = self._model.get_convergence_metrics(
                embeddings=self._scene_embeddings,
            )
            return ConvergenceData(
                confidence_pct=metrics.confidence_pct,
                n_comparisons=metrics.n_comparisons,
                phase=phase.value,
                is_converged=metrics.is_converged,
                phase_progress_pct=phase_progress,
            )
        except Exception as e:
            self.log(f"Failed to compute convergence: {e}", "warning")
            return ConvergenceData(
                confidence_pct=0.0,
                n_comparisons=self._model.n_comparisons,
                phase=phase.value,
                is_converged=False,
                phase_progress_pct=phase_progress,
            )

    def _compute_phase_progress(self, phase: ComparisonPhase, n_comparisons: int) -> float:
        """Compute progress (0-100) within the current phase.

        - **BROAD**: linear from 0 to broad_max comparisons.
        - **REFINE**: linear from broad_max to refine_max comparisons.
        - **BOUNDARY**: confidence-based (0-90% confidence maps to 0-100%).
        """
        n_scenes = len(self._scene_embeddings)
        broad_max, refine_max = compute_phase_thresholds(n_scenes)

        if phase == ComparisonPhase.BROAD:
            if broad_max <= 0:
                return 100.0
            return round(min(100.0, (n_comparisons / broad_max) * 100), 1)

        if phase == ComparisonPhase.REFINE:
            span = refine_max - broad_max
            if span <= 0:
                return 100.0
            progress = n_comparisons - broad_max
            return round(min(100.0, (progress / span) * 100), 1)

        # BOUNDARY phase: use confidence as proxy since there's no upper bound
        try:
            metrics = self._model.get_convergence_metrics(
                embeddings=self._scene_embeddings,
            )
            return round(min(100.0, (metrics.confidence_pct / 90) * 100), 1)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Pair conversion
    # ------------------------------------------------------------------

    def _pairs_to_data(
        self,
        pairs: list[PreferencePair],
        scene_details: dict[int, dict[str, Any]],
    ) -> list[PreferencePairData]:
        """Convert internal ``PreferencePair`` objects to API-friendly data."""
        result: list[PreferencePairData] = []
        for p in pairs:
            result.append(
                PreferencePairData(
                    scene_a_id=p.scene_a_id,
                    scene_b_id=p.scene_b_id,
                    phase=p.phase.value,
                    predicted_probability=p.predicted_probability,
                    scene_a=scene_details.get(p.scene_a_id),
                    scene_b=scene_details.get(p.scene_b_id),
                )
            )
        return result

    def _scenes_to_swipe_data(
        self,
        scene_ids: list[int],
        scene_details: dict[int, dict[str, Any]],
        phase: ComparisonPhase,
    ) -> list[PreferencePairData]:
        """Convert individual scene IDs to swipe-mode API data.

        Wraps each scene as a ``PreferencePairData`` with ``scene_b_id=-1``
        (synthetic opponent) for compatibility with the existing frontend.

        ``predicted_probability`` is the model's directional confidence:
        ``sigmoid(|score|)`` ranges from 0.5 (no opinion) to 1.0 (very
        confident).  The sign is discarded so the frontend cannot infer
        whether the model expects a like or dislike.
        """
        result: list[PreferencePairData] = []
        for sid in scene_ids:
            # Compute direction-agnostic confidence from predicted score.
            # Use best-frame scoring: max over frames for prediction.
            frames = self._scene_frames.get(sid)
            if frames is not None and self._model.n_comparisons > 0:
                frame_scores = frames @ self._model.mu
                best_idx = int(np.argmax(frame_scores))
                score = float(frame_scores[best_idx])
                # sigmoid(|score|): 0.5 = no opinion, 1.0 = very confident
                confidence = float(1.0 / (1.0 + np.exp(-abs(score))))
                # Predictive uncertainty for best-scoring frame
                best_frame = frames[best_idx]
                score_sigma = float(np.sqrt(np.sum(self._model.sigma_sq * best_frame**2)))
                z = score / max(score_sigma, 1e-6)
                # Pre-compute surprise for both directions so the frontend
                # can show instant toast without waiting for the backend.
                # Z-score normalizes by uncertainty — suppresses toasts when
                # the model is unsure, amplifies when genuinely wrong.
                surprise_like = round(
                    float(1.0 / (1.0 + np.exp(z * _SURPRISE_Z_AMPLIFICATION))), 3
                )  # High when z is negative (model expected dislike)
                surprise_dislike = round(
                    float(1.0 / (1.0 + np.exp(-z * _SURPRISE_Z_AMPLIFICATION))), 3
                )  # High when z is positive (model expected like)
            else:
                emb = self._scene_embeddings.get(sid)
                if emb is not None and self._model.n_comparisons > 0:
                    score, sigma = self._model.predict_score(emb)
                    confidence = float(1.0 / (1.0 + np.exp(-abs(score))))
                    z = score / max(sigma, 1e-6)
                    surprise_like = round(
                        float(1.0 / (1.0 + np.exp(z * _SURPRISE_Z_AMPLIFICATION))), 3
                    )
                    surprise_dislike = round(
                        float(1.0 / (1.0 + np.exp(-z * _SURPRISE_Z_AMPLIFICATION))), 3
                    )
                else:
                    confidence = 0.5
                    surprise_like = None
                    surprise_dislike = None

            result.append(
                PreferencePairData(
                    scene_a_id=sid,
                    scene_b_id=-1,
                    phase=phase.value,
                    predicted_probability=round(confidence, 3),
                    scene_a=scene_details.get(sid),
                    scene_b=None,
                    surprise_if_liked=surprise_like,
                    surprise_if_disliked=surprise_dislike,
                )
            )
        return result
