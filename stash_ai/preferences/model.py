"""Bayesian Bradley-Terry preference model with diagonal covariance.

Implements online Bayesian learning of a preference vector in embedding space.
Each pairwise comparison (A preferred over B) updates the posterior via a
diagonal Laplace approximation, keeping memory and compute O(d) per update
rather than O(d^2) for a full covariance matrix.

The preference vector p lives on the unit sphere. Scene scores are inner
products: score(scene) = p^T @ embedding(scene). The Bradley-Terry model
gives comparison probabilities: P(A > B) = sigmoid(p^T (e_A - e_B)).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray

from stash_ai.preferences.types import (
    ComparisonPhase,
    ConvergenceMetrics,
    PreferenceModelState,
    compute_phase_thresholds,
)

# ---------------------------------------------------------------------------
# Numerically stable sigmoid
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function.

    Avoids overflow by branching on sign of x:
      x >= 0: 1 / (1 + exp(-x))
      x <  0: exp(x) / (1 + exp(x))
    """
    if x >= 0:
        z: float = float(np.exp(-x))
        return 1.0 / (1.0 + z)
    else:
        z = float(np.exp(x))
        return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Bayesian preference model
# ---------------------------------------------------------------------------


class BayesianPreferenceModel:
    """Bayesian Bradley-Terry preference model with diagonal covariance.

    Maintains posterior distribution p ~ N(mu, diag(sigma^2)) over the
    preference vector in embedding space.  Each comparison updates the
    posterior via Laplace approximation.

    Attributes:
        dims: Dimensionality of the embedding space.
        noise_variance: Observation noise (controls update step size).
        mu: Posterior mean of the preference vector, shape (dims,).
        sigma_sq: Posterior diagonal variance, shape (dims,).
        n_comparisons: Total number of comparisons incorporated.
    """

    def __init__(self, dims: int = 768, noise_variance: float = 1.0) -> None:
        """Initialise with an uninformative prior.

        Args:
            dims: Embedding dimensionality.
            noise_variance: Observation noise variance.  Lower values make
                each comparison update the model more aggressively.
        """
        self.dims: int = dims
        self.noise_variance: float = noise_variance

        # Uninformative prior: zero mean, broad isotropic variance
        self.mu: NDArray[np.float32] = np.zeros(dims, dtype=np.float32)
        self.sigma_sq: NDArray[np.float32] = np.full(
            dims, 9.0, dtype=np.float32
        )  # sigma = 3.0 -> sigma^2 = 9.0

        self.n_comparisons: int = 0

    # ------------------------------------------------------------------
    # Warm-start from engagement data
    # ------------------------------------------------------------------

    def warm_start_from_engagement(
        self,
        embeddings: dict[int, NDArray[np.float32]],
        engagement_scores: dict[int, float],
    ) -> None:
        """Initialise preference vector via ridge regression on engagement.

        Solves: mu = argmin_p  sum_i (score_i - p^T e_i)^2 + lambda ||p||^2
        where score_i are normalised engagement scores and e_i are L2-normalised
        embeddings.  This gives a closed-form warm start that points the
        preference vector in the direction of high-engagement content.

        Args:
            embeddings: Mapping of scene_id -> embedding vector.
            engagement_scores: Mapping of scene_id -> raw engagement score.
        """
        # Find scenes present in both maps
        common_ids: list[int] = [sid for sid in embeddings if sid in engagement_scores]
        if len(common_ids) < 3:
            # Too few overlapping scenes for meaningful regression
            return

        # Build design matrix (n_scenes x dims) with L2-normalised rows
        X: NDArray[np.float32] = np.stack([embeddings[sid] for sid in common_ids], axis=0).astype(
            np.float32
        )
        norms: NDArray[np.float32] = np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32)
        # Avoid division by zero for degenerate embeddings
        norms = np.maximum(norms, np.float32(1e-8))
        X = X / norms

        # Normalise engagement scores to zero-mean, unit-variance
        raw_scores: NDArray[np.float32] = np.array(
            [engagement_scores[sid] for sid in common_ids], dtype=np.float32
        )
        score_mean: float = float(np.mean(raw_scores))
        score_std: float = float(np.std(raw_scores))
        if score_std < 1e-8:
            # All scores identical -- no signal to learn from
            return
        y: NDArray[np.float32] = ((raw_scores - score_mean) / score_std).astype(np.float32)

        # Ridge regression: p = (X^T X + lambda I)^{-1} X^T y
        # Use diagonal approximation for efficiency (avoids dims x dims inverse)
        # For diagonal: p_j = (sum_i X_ij * y_i) / (sum_i X_ij^2 + lambda)
        ridge_lambda: float = 1.0
        XtX_diag: NDArray[np.float32] = np.sum(X**2, axis=0).astype(np.float32)
        Xty: NDArray[np.float32] = (X.T @ y).astype(np.float32)

        self.mu = (Xty / (XtX_diag + ridge_lambda)).astype(np.float32)

        # Normalise to unit sphere
        mu_norm: float = float(np.linalg.norm(self.mu))
        if mu_norm > 1e-8:
            self.mu = (self.mu / mu_norm).astype(np.float32)

        # Tighten variance around the warm-started estimate (but not too tight
        # -- we still want comparisons to refine it)
        self.sigma_sq = np.full(self.dims, 4.0, dtype=np.float32)  # sigma = 2.0

    # ------------------------------------------------------------------
    # Online Bayesian update
    # ------------------------------------------------------------------

    def update(
        self,
        e_winner: NDArray[np.float32],
        e_loser: NDArray[np.float32],
        signal_weight: float = 1.0,
    ) -> None:
        """Update posterior after observing a pairwise comparison.

        Uses diagonal Laplace approximation to the posterior.  The update
        equations are derived from the Bradley-Terry log-likelihood with a
        Gaussian prior.

        Args:
            e_winner: Embedding of the preferred scene.
            e_loser: Embedding of the non-preferred scene.
            signal_weight: Multiplier for this comparison's influence.
                Use >1 for super-likes, <1 for uncertain/skip signals.
        """
        # Difference vector
        d: NDArray[np.float32] = (e_winner - e_loser).astype(np.float32)

        # Current model prediction
        score_diff: float = float(self.mu @ d)
        prob: float = _sigmoid(score_diff)

        # Logistic variance (information content of this comparison)
        lambda_t: float = prob * (1.0 - prob)

        # Diagonal Laplace update
        # Precision update: new_precision = old_precision + lambda_t * d^2 / noise * weight
        d_sq: NDArray[np.float32] = (d**2).astype(np.float32)
        precision_update: NDArray[np.float32] = (
            lambda_t * d_sq / self.noise_variance * signal_weight
        ).astype(np.float32)
        old_precision: NDArray[np.float32] = (1.0 / self.sigma_sq).astype(np.float32)
        new_precision: NDArray[np.float32] = (old_precision + precision_update).astype(np.float32)

        # New variance
        self.sigma_sq = (1.0 / new_precision).astype(np.float32)

        # Mean update: mu_new = mu + sigma_new^2 * (1 - prob) * d / noise * weight
        gradient_scale: float = (1.0 - prob) * signal_weight / self.noise_variance
        self.mu = (self.mu + self.sigma_sq * gradient_scale * d).astype(np.float32)

        # Re-normalise mu to the unit sphere to keep scores interpretable
        mu_norm: float = float(np.linalg.norm(self.mu))
        if mu_norm > 1e-8:
            self.mu = (self.mu / mu_norm).astype(np.float32)

        self.n_comparisons += 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_score(self, embedding: NDArray[np.float32]) -> tuple[float, float]:
        """Predict preference score and uncertainty for a scene.

        Args:
            embedding: Scene embedding vector.

        Returns:
            Tuple of (mean_score, uncertainty).
            - mean_score: mu^T @ embedding (higher = more preferred).
            - uncertainty: sqrt(embedding^T diag(sigma^2) embedding),
              the posterior standard deviation of the score.
        """
        e: NDArray[np.float32] = embedding.astype(np.float32)
        mean_score: float = float(self.mu @ e)

        # Uncertainty: standard deviation of the score under the posterior
        # Var(p^T e) = e^T Sigma e = sum_j sigma_j^2 * e_j^2  (diagonal)
        score_variance: float = float(np.sum(self.sigma_sq * e**2))
        uncertainty: float = float(np.sqrt(max(score_variance, 0.0)))

        return mean_score, uncertainty

    def predict_comparison(
        self,
        e_a: NDArray[np.float32],
        e_b: NDArray[np.float32],
    ) -> tuple[float, float]:
        """Predict comparison outcome between two scenes.

        Args:
            e_a: Embedding of scene A.
            e_b: Embedding of scene B.

        Returns:
            Tuple of (probability_a_wins, confidence).
            - probability_a_wins: P(A preferred over B) in [0, 1].
            - confidence: How certain the model is, in [0, 1].
              1.0 means extremely certain, 0.0 means maximally uncertain
              (probability near 0.5).
        """
        d: NDArray[np.float32] = (e_a - e_b).astype(np.float32)
        score_diff: float = float(self.mu @ d)
        prob_a_wins: float = _sigmoid(score_diff)

        # Confidence: how far the probability is from 0.5
        # Map |prob - 0.5| from [0, 0.5] to [0, 1]
        confidence: float = abs(prob_a_wins - 0.5) * 2.0

        return prob_a_wins, confidence

    # ------------------------------------------------------------------
    # Convergence tracking
    # ------------------------------------------------------------------

    def get_convergence_metrics(
        self,
        embeddings: dict[int, NDArray[np.float32]],
        top_k: int = 50,
    ) -> ConvergenceMetrics:
        """Compute convergence metrics over the scene library.

        Evaluates the current posterior uncertainty projected onto each scene's
        embedding to determine how confident the ranking is.

        Args:
            embeddings: Mapping of scene_id -> embedding vector.
            top_k: Number of top-scored scenes to track for worst-case
                uncertainty.

        Returns:
            ConvergenceMetrics with avg/max sigma and phase information.
        """
        n_scenes = len(embeddings)
        if not embeddings:
            return ConvergenceMetrics(
                avg_sigma=3.0,
                max_sigma_top50=3.0,
                n_comparisons=self.n_comparisons,
                phase=self._infer_phase(n_scenes),
            )

        # Compute (score, uncertainty) for every scene
        scored: list[tuple[int, float, float]] = []
        for scene_id, emb in embeddings.items():
            score, sigma = self.predict_score(emb)
            scored.append((scene_id, score, sigma))

        # Global average uncertainty
        all_sigmas: list[float] = [s for _, _, s in scored]
        avg_sigma: float = float(np.mean(all_sigmas))

        # Worst uncertainty among the top-k by predicted score
        scored.sort(key=lambda t: t[1], reverse=True)
        top_k_actual: int = min(top_k, len(scored))
        top_sigmas: list[float] = [s for _, _, s in scored[:top_k_actual]]
        max_sigma_top50: float = max(top_sigmas) if top_sigmas else 3.0

        return ConvergenceMetrics(
            avg_sigma=avg_sigma,
            max_sigma_top50=max_sigma_top50,
            n_comparisons=self.n_comparisons,
            phase=self._infer_phase(n_scenes),
        )

    def _infer_phase(self, n_scenes: int = 0) -> ComparisonPhase:
        """Infer the current comparison phase from the number of comparisons.

        Thresholds scale with ``sqrt(n_scenes)`` so larger libraries get
        proportionally more exploration before transitioning.

        Args:
            n_scenes: Number of embedded scenes (0 for static defaults).
        """
        broad_max, refine_max = compute_phase_thresholds(n_scenes)
        if self.n_comparisons < broad_max:
            return ComparisonPhase.BROAD
        elif self.n_comparisons < refine_max:
            return ComparisonPhase.REFINE
        else:
            return ComparisonPhase.BOUNDARY

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def get_top_scenes(
        self,
        embeddings: dict[int, NDArray[np.float32]],
        limit: int = 50,
    ) -> list[tuple[int, float, float]]:
        """Rank all scenes by predicted preference score.

        Args:
            embeddings: Mapping of scene_id -> embedding vector.
            limit: Maximum number of scenes to return.

        Returns:
            List of (scene_id, score, uncertainty) sorted by score descending,
            truncated to *limit* entries.
        """
        scored: list[tuple[int, float, float]] = []
        for scene_id, emb in embeddings.items():
            score, sigma = self.predict_score(emb)
            scored.append((scene_id, score, sigma))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_state(
        self,
        model_key: str,
        n_scenes: int = 0,
    ) -> PreferenceModelState:
        """Serialise the model to a storable state object.

        Args:
            model_key: Identifier for the embedding model (e.g. ``"siglip"``).
            n_scenes: Number of embedded scenes for dynamic phase thresholds.

        Returns:
            PreferenceModelState suitable for database persistence.
        """
        return PreferenceModelState(
            model_key=model_key,
            preference_mean=self.mu.copy(),
            preference_covariance_diag=self.sigma_sq.copy(),
            n_comparisons=self.n_comparisons,
            noise_variance=self.noise_variance,
            phase=self._infer_phase(n_scenes).value,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def from_state(cls, state: PreferenceModelState) -> BayesianPreferenceModel:
        """Reconstruct a model from a previously stored state.

        Args:
            state: PreferenceModelState loaded from the database.

        Returns:
            Fully initialised BayesianPreferenceModel.
        """
        dims: int = len(state.preference_mean)
        model = cls(dims=dims, noise_variance=state.noise_variance)
        model.mu = state.preference_mean.astype(np.float32).copy()
        model.sigma_sq = state.preference_covariance_diag.astype(np.float32).copy()
        model.n_comparisons = state.n_comparisons
        return model

    # ------------------------------------------------------------------
    # Blending with engagement profile
    # ------------------------------------------------------------------

    def combine_with_engagement_profile(
        self,
        engagement_profile: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Blend the learned preference vector with an engagement-based profile.

        Uses a sigmoid schedule to transition from engagement-dominated to
        comparison-dominated as the number of comparisons grows:

            alpha = sigmoid((n_comparisons - 15) / 5)

        At 0 comparisons, alpha ~ 0.05 -> mostly engagement profile.
        At 15 comparisons, alpha = 0.5 -> equal blend.
        At 30 comparisons, alpha ~ 0.95 -> mostly learned preferences.

        Args:
            engagement_profile: L2-normalised engagement-weighted embedding
                (e.g. from ``UserProfileBuilder.build_profile``).

        Returns:
            L2-normalised blended preference vector.
        """
        alpha: float = _sigmoid((self.n_comparisons - 15.0) / 5.0)

        blended: NDArray[np.float32] = (
            alpha * self.mu + (1.0 - alpha) * engagement_profile.astype(np.float32)
        ).astype(np.float32)

        # Normalise to unit sphere
        norm: float = float(np.linalg.norm(blended))
        if norm > 1e-8:
            blended = (blended / norm).astype(np.float32)

        return blended
