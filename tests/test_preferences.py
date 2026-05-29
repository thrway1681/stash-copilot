"""Comprehensive test suite for the preference training system.

Tests the Bayesian Bradley-Terry preference model, pair selection engine,
convergence metrics, and signal weight behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from stash_ai.preferences.model import BayesianPreferenceModel, _sigmoid
from stash_ai.preferences.pair_selector import ClusterInfo, PairSelectionEngine
from stash_ai.preferences.types import (
    ComparisonPhase,
    ConvergenceMetrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_embeddings() -> dict[int, NDArray[np.float32]]:
    """Generate 100 random 128-dim embeddings (small dim for speed)."""
    rng = np.random.default_rng(42)
    embs: dict[int, NDArray[np.float32]] = {}
    for i in range(100):
        v = rng.standard_normal(128).astype(np.float32)
        v /= np.linalg.norm(v)
        embs[i] = v
    return embs


@pytest.fixture
def engagement_scores() -> dict[int, float]:
    """Engagement scores for 100 scenes.

    Higher scene ID implies higher engagement, giving a clear gradient for
    warm-start regression to learn.
    """
    return {i: float(i) for i in range(100)}


@pytest.fixture
def model() -> BayesianPreferenceModel:
    """Fresh 128-dim Bayesian preference model."""
    return BayesianPreferenceModel(dims=128, noise_variance=1.0)


@pytest.fixture
def clusters(random_embeddings: dict[int, NDArray[np.float32]]) -> list[ClusterInfo]:
    """Three simple clusters constructed from the random embeddings.

    Cluster 0: scenes 0-32
    Cluster 1: scenes 33-65
    Cluster 2: scenes 66-99
    """
    cluster_defs: list[tuple[int, int]] = [(0, 33), (33, 66), (66, 100)]
    result: list[ClusterInfo] = []
    for cluster_id, (lo, hi) in enumerate(cluster_defs):
        scene_ids = list(range(lo, hi))
        # Compute centroid from member embeddings
        member_embs = np.stack([random_embeddings[sid] for sid in scene_ids], axis=0)
        centroid = member_embs.mean(axis=0).astype(np.float32)
        result.append(
            ClusterInfo(
                cluster_id=cluster_id,
                scene_ids=scene_ids,
                centroid=centroid,
                engagement_share=1.0 / 3.0,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair_selector(
    model: BayesianPreferenceModel,
    embeddings: dict[int, NDArray[np.float32]],
) -> PairSelectionEngine:
    """Build a PairSelectionEngine from a model and embeddings.

    The pair selector uses the model's native API (predict_score with
    embedding, predict_comparison, sigma_sq, noise_variance) so no
    monkey-patching is needed.
    """
    return PairSelectionEngine(model=model, embeddings=embeddings)


# ===========================================================================
# TestBayesianPreferenceModel
# ===========================================================================


class TestBayesianPreferenceModel:
    """Tests for the Bayesian Bradley-Terry preference model."""

    # 1 ---------------------------------------------------------------
    def test_initialization(self, model: BayesianPreferenceModel) -> None:
        """Model starts with mu=zeros, sigma=3.0 for all dims."""
        np.testing.assert_array_equal(model.mu, np.zeros(128, dtype=np.float32))
        expected_sigma_sq = np.full(128, 9.0, dtype=np.float32)
        np.testing.assert_array_almost_equal(model.sigma_sq, expected_sigma_sq)
        assert model.n_comparisons == 0
        assert model.dims == 128
        assert model.noise_variance == 1.0

    # 2 ---------------------------------------------------------------
    def test_warm_start_from_engagement(
        self,
        model: BayesianPreferenceModel,
        random_embeddings: dict[int, NDArray[np.float32]],
        engagement_scores: dict[int, float],
    ) -> None:
        """After warm-start, the preference mean should point in the direction
        of high-engagement scenes more than low-engagement ones.
        """
        model.warm_start_from_engagement(random_embeddings, engagement_scores)

        # mu should no longer be zeros
        assert not np.allclose(model.mu, 0.0)

        # mu should be approximately unit-norm
        mu_norm = float(np.linalg.norm(model.mu))
        assert mu_norm == pytest.approx(1.0, abs=1e-3)

        # High-engagement scenes should score higher than low-engagement ones
        high_ids = list(range(80, 100))
        low_ids = list(range(0, 20))

        high_scores = [float(model.mu @ random_embeddings[sid]) for sid in high_ids]
        low_scores = [float(model.mu @ random_embeddings[sid]) for sid in low_ids]

        # On average, high-engagement scenes should score higher
        assert np.mean(high_scores) > np.mean(low_scores)

        # Sigma should have been tightened from 9.0 to 4.0
        np.testing.assert_array_almost_equal(model.sigma_sq, np.full(128, 4.0, dtype=np.float32))

    # 3 ---------------------------------------------------------------
    def test_update_moves_toward_winner(
        self,
        model: BayesianPreferenceModel,
    ) -> None:
        """After update(e_winner, e_loser), the winner's score should increase
        and the loser's score should decrease.
        """
        rng = np.random.default_rng(123)
        e_winner = rng.standard_normal(128).astype(np.float32)
        e_winner /= np.linalg.norm(e_winner)
        e_loser = rng.standard_normal(128).astype(np.float32)
        e_loser /= np.linalg.norm(e_loser)

        score_winner_before, _ = model.predict_score(e_winner)
        score_loser_before, _ = model.predict_score(e_loser)

        model.update(e_winner, e_loser)

        score_winner_after, _ = model.predict_score(e_winner)
        score_loser_after, _ = model.predict_score(e_loser)

        assert score_winner_after > score_winner_before
        assert score_loser_after < score_loser_before

    # 4 ---------------------------------------------------------------
    def test_update_reduces_uncertainty(
        self,
        model: BayesianPreferenceModel,
    ) -> None:
        """After update, sigma values should decrease (more confident)."""
        rng = np.random.default_rng(456)
        e_a = rng.standard_normal(128).astype(np.float32)
        e_a /= np.linalg.norm(e_a)
        e_b = rng.standard_normal(128).astype(np.float32)
        e_b /= np.linalg.norm(e_b)

        sigma_before = model.sigma_sq.copy()
        model.update(e_a, e_b)
        sigma_after = model.sigma_sq.copy()

        # sigma_sq should decrease (or stay equal on dimensions where d=0)
        assert np.all(sigma_after <= sigma_before + 1e-7)
        # At least some dimensions should have strictly decreased
        assert np.any(sigma_after < sigma_before - 1e-7)

    # 5 ---------------------------------------------------------------
    def test_predict_comparison_symmetry(
        self,
        model: BayesianPreferenceModel,
    ) -> None:
        """P(A>B) + P(B>A) should approximately equal 1.0."""
        rng = np.random.default_rng(789)
        e_a = rng.standard_normal(128).astype(np.float32)
        e_a /= np.linalg.norm(e_a)
        e_b = rng.standard_normal(128).astype(np.float32)
        e_b /= np.linalg.norm(e_b)

        prob_ab, _ = model.predict_comparison(e_a, e_b)
        prob_ba, _ = model.predict_comparison(e_b, e_a)

        assert prob_ab + prob_ba == pytest.approx(1.0, abs=1e-6)

    # 6 ---------------------------------------------------------------
    def test_predict_comparison_confidence(
        self,
        model: BayesianPreferenceModel,
    ) -> None:
        """After many consistent updates, confidence should increase.

        Note: Because the model renormalizes mu to the unit sphere after each
        update, the probability P(A>B) is bounded by the geometry of the
        embeddings. We test that confidence grows monotonically rather than
        requiring a specific threshold.
        """
        rng = np.random.default_rng(101)
        e_winner = rng.standard_normal(128).astype(np.float32)
        e_winner /= np.linalg.norm(e_winner)
        e_loser = rng.standard_normal(128).astype(np.float32)
        e_loser /= np.linalg.norm(e_loser)

        # Measure confidence at different stages
        prob_0, confidence_0 = model.predict_comparison(e_winner, e_loser)

        for _ in range(10):
            model.update(e_winner, e_loser)
        prob_10, confidence_10 = model.predict_comparison(e_winner, e_loser)

        for _ in range(40):
            model.update(e_winner, e_loser)
        prob_50, confidence_50 = model.predict_comparison(e_winner, e_loser)

        # Confidence should increase with more consistent comparisons
        assert confidence_50 > confidence_0
        assert confidence_10 > confidence_0

        # Winner should have higher probability throughout
        assert prob_10 > 0.5
        assert prob_50 > 0.5
        assert prob_50 > prob_0

    # 7 ---------------------------------------------------------------
    def test_super_like_weight(self) -> None:
        """update with signal_weight=3.0 should reduce uncertainty more than
        signal_weight=1.0, reflecting greater confidence from a super-like.

        The primary observable effect of higher signal_weight is a larger
        reduction in posterior variance (sigma_sq). Because mu is renormalized
        to the unit sphere after each update, the effect on directional
        preference is subtle, but the precision gain is guaranteed by the
        update equations: precision_update = lambda_t * d^2 / noise * weight.
        """
        rng = np.random.default_rng(202)
        e_winner = rng.standard_normal(128).astype(np.float32)
        e_winner /= np.linalg.norm(e_winner)
        e_loser = rng.standard_normal(128).astype(np.float32)
        e_loser /= np.linalg.norm(e_loser)

        # Model with normal weight
        model_normal = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        model_normal.update(e_winner, e_loser, signal_weight=1.0)

        # Model with super-like weight
        model_super = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        model_super.update(e_winner, e_loser, signal_weight=3.0)

        # Super-like should reduce uncertainty more (primary and guaranteed
        # effect from the precision update equation)
        avg_sigma_normal = float(np.mean(model_normal.sigma_sq))
        avg_sigma_super = float(np.mean(model_super.sigma_sq))
        assert avg_sigma_super < avg_sigma_normal

        # The uncertainty reduction should be proportional to the weight --
        # dimensions aligned with d=(e_winner-e_loser) should show the effect
        d = e_winner - e_loser
        # Find dimensions with the largest |d| (most affected)
        top_dims = np.argsort(np.abs(d))[-10:]
        avg_sigma_top_normal = float(np.mean(model_normal.sigma_sq[top_dims]))
        avg_sigma_top_super = float(np.mean(model_super.sigma_sq[top_dims]))

        # On the most-affected dimensions the difference should be clear
        assert avg_sigma_top_super < avg_sigma_top_normal

    # 8 ---------------------------------------------------------------
    def test_convergence_metrics(
        self,
        model: BayesianPreferenceModel,
        random_embeddings: dict[int, NDArray[np.float32]],
    ) -> None:
        """avg_sigma decreases as more comparisons are recorded."""
        metrics_before = model.get_convergence_metrics(random_embeddings)

        rng = np.random.default_rng(303)
        scene_ids = list(random_embeddings.keys())

        # Perform 20 comparisons
        for _ in range(20):
            idx_a, idx_b = rng.choice(len(scene_ids), size=2, replace=False)
            e_a = random_embeddings[scene_ids[idx_a]]
            e_b = random_embeddings[scene_ids[idx_b]]
            model.update(e_a, e_b)

        metrics_after = model.get_convergence_metrics(random_embeddings)

        assert metrics_after.avg_sigma < metrics_before.avg_sigma
        assert metrics_after.n_comparisons == 20
        assert metrics_before.n_comparisons == 0

    # 9 ---------------------------------------------------------------
    def test_get_top_scenes(
        self,
        model: BayesianPreferenceModel,
        random_embeddings: dict[int, NDArray[np.float32]],
    ) -> None:
        """Returns scenes sorted by predicted score in descending order."""
        rng = np.random.default_rng(404)

        # Warm-start so scores are not all zero
        engagement = {i: float(i) for i in range(100)}
        model.warm_start_from_engagement(random_embeddings, engagement)

        top = model.get_top_scenes(random_embeddings, limit=10)

        # Should return 10 results
        assert len(top) == 10

        # Each entry is (scene_id, score, uncertainty)
        for scene_id, score, uncertainty in top:
            assert isinstance(scene_id, int)
            assert isinstance(score, float)
            assert isinstance(uncertainty, float)
            assert uncertainty >= 0.0

        # Scores should be in descending order
        scores = [s for _, s, _ in top]
        assert scores == sorted(scores, reverse=True)

    # 10 --------------------------------------------------------------
    def test_serialization_roundtrip(
        self,
        random_embeddings: dict[int, NDArray[np.float32]],
    ) -> None:
        """to_state() -> from_state() preserves model state exactly."""
        model = BayesianPreferenceModel(dims=128, noise_variance=1.5)

        # Apply some updates so state is non-trivial
        rng = np.random.default_rng(505)
        scene_ids = list(random_embeddings.keys())
        for _ in range(10):
            idx_a, idx_b = rng.choice(len(scene_ids), size=2, replace=False)
            model.update(
                random_embeddings[scene_ids[idx_a]],
                random_embeddings[scene_ids[idx_b]],
            )

        state = model.to_state(model_key="test_siglip")
        restored = BayesianPreferenceModel.from_state(state)

        # All fields should match
        np.testing.assert_array_almost_equal(restored.mu, model.mu)
        np.testing.assert_array_almost_equal(restored.sigma_sq, model.sigma_sq)
        assert restored.n_comparisons == model.n_comparisons
        assert restored.dims == model.dims
        assert restored.noise_variance == model.noise_variance

        # Predictions should be identical
        test_emb = random_embeddings[0]
        score_orig, unc_orig = model.predict_score(test_emb)
        score_rest, unc_rest = restored.predict_score(test_emb)
        assert score_orig == pytest.approx(score_rest, abs=1e-6)
        assert unc_orig == pytest.approx(unc_rest, abs=1e-6)

    # 11 --------------------------------------------------------------
    def test_combine_with_engagement_profile(
        self,
        random_embeddings: dict[int, NDArray[np.float32]],
    ) -> None:
        """With 0 comparisons alpha~0.05 (engagement dominates), with 30
        comparisons alpha~0.95 (swipe dominates).
        """
        rng = np.random.default_rng(606)
        engagement_profile = rng.standard_normal(128).astype(np.float32)
        engagement_profile /= np.linalg.norm(engagement_profile)

        # --- 0 comparisons: engagement should dominate ---
        model_0 = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        # Give model_0 a non-zero mu so we can distinguish blending
        model_0.mu = rng.standard_normal(128).astype(np.float32)
        model_0.mu /= np.linalg.norm(model_0.mu)
        model_0.n_comparisons = 0

        blended_0 = model_0.combine_with_engagement_profile(engagement_profile)

        # alpha at 0 comparisons: sigmoid((0-15)/5) = sigmoid(-3) ~ 0.047
        alpha_0 = _sigmoid((0.0 - 15.0) / 5.0)
        assert alpha_0 == pytest.approx(0.047, abs=0.01)

        # Blended should be very close to engagement_profile
        cos_sim_with_eng = float(np.dot(blended_0, engagement_profile))
        cos_sim_with_mu = float(np.dot(blended_0, model_0.mu))
        assert cos_sim_with_eng > cos_sim_with_mu

        # --- 30 comparisons: swipe preferences should dominate ---
        model_30 = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        model_30.mu = rng.standard_normal(128).astype(np.float32)
        model_30.mu /= np.linalg.norm(model_30.mu)
        model_30.n_comparisons = 30

        blended_30 = model_30.combine_with_engagement_profile(engagement_profile)

        # alpha at 30 comparisons: sigmoid((30-15)/5) = sigmoid(3) ~ 0.953
        alpha_30 = _sigmoid((30.0 - 15.0) / 5.0)
        assert alpha_30 == pytest.approx(0.953, abs=0.01)

        # Blended should be very close to the model's own mu
        cos_sim_with_mu_30 = float(np.dot(blended_30, model_30.mu))
        cos_sim_with_eng_30 = float(np.dot(blended_30, engagement_profile))
        assert cos_sim_with_mu_30 > cos_sim_with_eng_30

        # Both results should be unit-normalised
        assert float(np.linalg.norm(blended_0)) == pytest.approx(1.0, abs=1e-5)
        assert float(np.linalg.norm(blended_30)) == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# TestPairSelectionEngine
# ===========================================================================


class TestPairSelectionEngine:
    """Tests for the intelligent pair selection engine."""

    # 12 --------------------------------------------------------------
    def test_broad_phase_uses_clusters(
        self,
        model: BayesianPreferenceModel,
        random_embeddings: dict[int, NDArray[np.float32]],
        clusters: list[ClusterInfo],
    ) -> None:
        """When clusters are provided, BROAD phase pairs representatives
        from different clusters.
        """
        engine = _make_pair_selector(model, random_embeddings)

        pairs = engine.select_pairs(
            n_pairs=5,
            phase=ComparisonPhase.BROAD,
            clusters=clusters,
            compared_pairs=set(),
            exploration_rate=0.0,  # No exploration -- pure exploitation
        )

        assert len(pairs) > 0
        assert len(pairs) <= 5

        # All pairs should be tagged as BROAD phase
        for pair in pairs:
            assert pair.phase == ComparisonPhase.BROAD

        # At least one pair should have scenes from different clusters
        cluster_membership: dict[int, int] = {}
        for cluster in clusters:
            for sid in cluster.scene_ids:
                cluster_membership[sid] = cluster.cluster_id

        cross_cluster_count = 0
        for pair in pairs:
            c_a = cluster_membership.get(pair.scene_a_id)
            c_b = cluster_membership.get(pair.scene_b_id)
            if c_a is not None and c_b is not None and c_a != c_b:
                cross_cluster_count += 1

        assert cross_cluster_count > 0, "Expected at least one inter-cluster pair in BROAD phase"

    # 13 --------------------------------------------------------------
    def test_broad_phase_without_clusters(
        self,
        model: BayesianPreferenceModel,
        random_embeddings: dict[int, NDArray[np.float32]],
    ) -> None:
        """Without clusters, BROAD phase falls back to diverse sampling
        (creates ad-hoc clusters internally).
        """
        engine = _make_pair_selector(model, random_embeddings)

        pairs = engine.select_pairs(
            n_pairs=5,
            phase=ComparisonPhase.BROAD,
            clusters=None,  # No clusters provided
            compared_pairs=set(),
            exploration_rate=0.0,
        )

        assert len(pairs) > 0
        assert len(pairs) <= 5

        # All should be BROAD
        for pair in pairs:
            assert pair.phase == ComparisonPhase.BROAD

        # Verify pairs contain valid scene IDs
        valid_ids = set(random_embeddings.keys())
        for pair in pairs:
            assert pair.scene_a_id in valid_ids
            assert pair.scene_b_id in valid_ids
            assert pair.scene_a_id != pair.scene_b_id

    # 14 --------------------------------------------------------------
    def test_refine_phase_selects_similar(
        self,
        random_embeddings: dict[int, NDArray[np.float32]],
    ) -> None:
        """REFINE phase pairs should have high cosine similarity (within the
        [0.75, 0.95] sweet-spot or backfill).

        We construct embeddings with very small perturbations around a base
        direction so that many pairs naturally fall in the [0.75, 0.95]
        similarity range that the refine phase targets.
        """
        rng = np.random.default_rng(707)
        embs: dict[int, NDArray[np.float32]] = {}

        # Create a "base" direction and very small perturbations so pairwise
        # cosine similarities land in the [0.75, 0.95] sweet-spot.
        # In 128-D, noise_scale ~0.03 per component gives ||noise||^2 ~ 0.12,
        # yielding cosine similarities around 0.80-0.95 between pairs.
        base = rng.standard_normal(128).astype(np.float32)
        base /= np.linalg.norm(base)

        for i in range(50):
            noise = rng.standard_normal(128).astype(np.float32) * 0.03
            v = base + noise
            v /= np.linalg.norm(v)
            embs[i] = v

        # Add some dissimilar scenes for diversity
        for i in range(50, 80):
            v = rng.standard_normal(128).astype(np.float32)
            v /= np.linalg.norm(v)
            embs[i] = v

        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        # Warm-start to give the model non-zero scores (similar scenes score
        # highest so the refine phase draws pairs from them)
        engagement = {i: float(50 - abs(i - 25)) for i in range(80)}
        model.warm_start_from_engagement(embs, engagement)

        engine = _make_pair_selector(model, embs)

        pairs = engine.select_pairs(
            n_pairs=10,
            phase=ComparisonPhase.REFINE,
            clusters=None,
            compared_pairs=set(),
            exploration_rate=0.0,
        )

        assert len(pairs) > 0

        # Check that at least some refine pairs have high similarity
        # (the engine targets [0.75, 0.95] and backfills otherwise)
        high_sim_pairs = [p for p in pairs if p.similarity >= 0.7]
        assert len(high_sim_pairs) > 0, (
            f"Expected some pairs with similarity >= 0.7, "
            f"got similarities: {[p.similarity for p in pairs]}"
        )

    # 15 --------------------------------------------------------------
    def test_boundary_phase_maximizes_info_gain(
        self,
        random_embeddings: dict[int, NDArray[np.float32]],
    ) -> None:
        """BOUNDARY phase pairs should have high information gain (meaning
        the model has uncertainty about these comparisons).
        """
        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)

        # Do a few updates to create non-uniform uncertainty
        rng = np.random.default_rng(808)
        ids = list(random_embeddings.keys())
        for _ in range(10):
            i, j = rng.choice(len(ids), size=2, replace=False)
            model.update(random_embeddings[ids[i]], random_embeddings[ids[j]])

        engine = _make_pair_selector(model, random_embeddings)

        pairs = engine.select_pairs(
            n_pairs=10,
            phase=ComparisonPhase.BOUNDARY,
            clusters=None,
            compared_pairs=set(),
            exploration_rate=0.0,
        )

        assert len(pairs) > 0

        # All boundary pairs should be tagged correctly
        for pair in pairs:
            assert pair.phase == ComparisonPhase.BOUNDARY

        # Information gain should be positive for boundary pairs
        for pair in pairs:
            assert pair.information_gain >= 0.0

        # Pairs should be sorted by information gain (highest first)
        ig_values = [p.information_gain for p in pairs]
        assert ig_values == sorted(ig_values, reverse=True)

    # 16 --------------------------------------------------------------
    def test_never_selects_already_compared(
        self,
        model: BayesianPreferenceModel,
        random_embeddings: dict[int, NDArray[np.float32]],
        clusters: list[ClusterInfo],
    ) -> None:
        """Previously compared pairs are excluded from selection.

        Note: The engine mutates the compared_pairs set internally (adding
        newly selected pairs to prevent duplicates within a call).  We
        snapshot the set before passing it to the second call so we can
        verify that no pair from round 1 appears in round 2.
        """
        engine = _make_pair_selector(model, random_embeddings)

        # First round: select 5 pairs with explicit clusters
        pairs_round1 = engine.select_pairs(
            n_pairs=5,
            phase=ComparisonPhase.BROAD,
            clusters=clusters,
            compared_pairs=set(),
            exploration_rate=0.0,
        )

        assert len(pairs_round1) > 0

        # Build the compared set from round 1
        round1_pairs: set[tuple[int, int]] = set()
        for p in pairs_round1:
            pair = (min(p.scene_a_id, p.scene_b_id), max(p.scene_a_id, p.scene_b_id))
            round1_pairs.add(pair)

        # Pass a *copy* because the engine mutates the set internally
        pairs_round2 = engine.select_pairs(
            n_pairs=5,
            phase=ComparisonPhase.BROAD,
            clusters=clusters,
            compared_pairs=round1_pairs.copy(),
            exploration_rate=0.0,
        )

        # Verify no pair from round 2 was already in round 1
        for p in pairs_round2:
            pair = (min(p.scene_a_id, p.scene_b_id), max(p.scene_a_id, p.scene_b_id))
            assert pair not in round1_pairs, (
                f"Pair {pair} was already compared in round 1 but selected again"
            )

    # 17 --------------------------------------------------------------
    def test_exploration_rate(
        self,
        model: BayesianPreferenceModel,
        random_embeddings: dict[int, NDArray[np.float32]],
        clusters: list[ClusterInfo],
    ) -> None:
        """With exploration_rate=1.0, all pairs should be from random
        exploration (no phase-specific exploitation).
        """
        engine = _make_pair_selector(model, random_embeddings)

        # With exploration_rate=1.0, all pairs are exploration pairs
        pairs = engine.select_pairs(
            n_pairs=10,
            phase=ComparisonPhase.BROAD,
            clusters=clusters,
            compared_pairs=set(),
            exploration_rate=1.0,
        )

        assert len(pairs) > 0
        assert len(pairs) <= 10

        # All pairs should be valid
        valid_ids = set(random_embeddings.keys())
        for p in pairs:
            assert p.scene_a_id in valid_ids
            assert p.scene_b_id in valid_ids
            assert p.scene_a_id != p.scene_b_id

    # 18 --------------------------------------------------------------
    def test_pair_normalization(
        self,
        model: BayesianPreferenceModel,
        random_embeddings: dict[int, NDArray[np.float32]],
    ) -> None:
        """(A,B) and (B,A) treated as same pair -- normalised to (min, max)."""
        engine = _make_pair_selector(model, random_embeddings)

        # The static method should normalise both orderings to the same tuple
        assert engine._normalize_pair(5, 10) == (5, 10)
        assert engine._normalize_pair(10, 5) == (5, 10)
        assert engine._normalize_pair(7, 7) == (7, 7)

        # When we mark (5, 10) as compared, selecting should exclude both
        # orderings
        compared: set[tuple[int, int]] = {(5, 10)}

        # The engine should not select (5, 10) or (10, 5) since they normalise
        # to the same pair. We test this indirectly by selecting many pairs and
        # verifying none match.
        pairs = engine.select_pairs(
            n_pairs=50,
            phase=ComparisonPhase.BROAD,
            clusters=None,
            compared_pairs=compared,
            exploration_rate=0.0,
        )

        for p in pairs:
            normalised = (
                min(p.scene_a_id, p.scene_b_id),
                max(p.scene_a_id, p.scene_b_id),
            )
            assert normalised != (5, 10), (
                "Selected the compared pair (5, 10) despite it being excluded"
            )


# ===========================================================================
# TestConvergenceMetrics
# ===========================================================================


class TestConvergenceMetrics:
    """Tests for convergence tracking."""

    # 19 --------------------------------------------------------------
    def test_confidence_pct_range(self) -> None:
        """confidence_pct should always be in [0, 100].

        Note: confidence_pct combines sigma factor with a count dampener that
        requires ~50 comparisons for full credit. Tests use n_comparisons=50
        to isolate sigma factor testing.
        """
        # Test with various avg_sigma values (using n_comparisons=50 for full count factor)
        # Formula: sigma_factor = (3.0 - avg_sigma) / 2.9, clamped to [0, 1]
        test_cases: list[tuple[float, float, float]] = [
            (3.0, 3.0, 0.0),  # Maximum uncertainty -> 0% confidence
            (0.1, 0.1, 100.0),  # Fully converged (sigma=0.1) -> 100% confidence
            (1.55, 1.55, 50.0),  # Midpoint: (3.0-1.55)/2.9 = 0.5 -> 50%
            (5.0, 5.0, 0.0),  # Beyond max uncertainty -> clamped to 0%
            (0.0, 0.0, 100.0),  # Below min uncertainty -> clamped to 100%
        ]

        for avg_sigma, max_sigma, expected_confidence in test_cases:
            metrics = ConvergenceMetrics(
                avg_sigma=avg_sigma,
                max_sigma_top50=max_sigma,
                n_comparisons=50,  # Full count factor (50/50 = 1.0)
                phase=ComparisonPhase.BROAD,
            )
            assert 0.0 <= metrics.confidence_pct <= 100.0, (
                f"confidence_pct={metrics.confidence_pct} out of range for avg_sigma={avg_sigma}"
            )
            assert metrics.confidence_pct == pytest.approx(expected_confidence, abs=1.0)

    # 20 --------------------------------------------------------------
    def test_is_converged(self) -> None:
        """Returns True when sigma thresholds are met."""
        # Converged: max_sigma_top50 < 1.5 AND avg_sigma < 2.5
        converged = ConvergenceMetrics(
            avg_sigma=1.0,
            max_sigma_top50=1.0,
            n_comparisons=50,
            phase=ComparisonPhase.BOUNDARY,
        )
        assert converged.is_converged is True

        # Not converged: max_sigma_top50 too high
        not_converged_max = ConvergenceMetrics(
            avg_sigma=1.0,
            max_sigma_top50=2.0,
            n_comparisons=50,
            phase=ComparisonPhase.BOUNDARY,
        )
        assert not_converged_max.is_converged is False

        # Not converged: avg_sigma too high
        not_converged_avg = ConvergenceMetrics(
            avg_sigma=2.8,
            max_sigma_top50=1.0,
            n_comparisons=50,
            phase=ComparisonPhase.BOUNDARY,
        )
        assert not_converged_avg.is_converged is False

        # Edge case: exactly at threshold
        edge = ConvergenceMetrics(
            avg_sigma=2.5,
            max_sigma_top50=1.5,
            n_comparisons=30,
            phase=ComparisonPhase.REFINE,
        )
        assert edge.is_converged is False  # Must be strictly less than


# ===========================================================================
# TestSignalWeight
# ===========================================================================


class TestSignalWeight:
    """Tests for response time weighting behaviour."""

    # 21 --------------------------------------------------------------
    def test_response_time_weights(self) -> None:
        """Quick decisions should produce larger uncertainty reduction.

        Quick decisions indicate more certain preferences. We simulate this
        by mapping response time to signal_weight:
          - < 1000ms  -> signal_weight 2.0 (snap decision, very confident)
          - 1000-3000ms -> signal_weight 1.0 (normal)
          - > 3000ms  -> signal_weight 0.5 (hesitant, less confident)

        The primary observable effect of signal_weight is in the precision
        (inverse variance) update: precision_update = lambda * d^2 * weight.
        Higher weight produces strictly more precision gain per comparison.
        """
        rng = np.random.default_rng(909)
        e_winner = rng.standard_normal(128).astype(np.float32)
        e_winner /= np.linalg.norm(e_winner)
        e_loser = rng.standard_normal(128).astype(np.float32)
        e_loser /= np.linalg.norm(e_loser)

        def _response_time_to_weight(ms: int) -> float:
            """Map response time to signal weight."""
            if ms < 1000:
                return 2.0
            elif ms <= 3000:
                return 1.0
            else:
                return 0.5

        # Verify weight mapping
        assert _response_time_to_weight(500) == 2.0
        assert _response_time_to_weight(1500) == 1.0
        assert _response_time_to_weight(5000) == 0.5

        # Quick response (high weight)
        model_quick = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        model_quick.update(e_winner, e_loser, signal_weight=2.0)

        # Normal response
        model_normal = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        model_normal.update(e_winner, e_loser, signal_weight=1.0)

        # Slow response (low weight)
        model_slow = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        model_slow.update(e_winner, e_loser, signal_weight=0.5)

        # Higher weight should reduce uncertainty more (guaranteed by update
        # equations: precision_update proportional to weight)
        avg_sigma_quick = float(np.mean(model_quick.sigma_sq))
        avg_sigma_normal = float(np.mean(model_normal.sigma_sq))
        avg_sigma_slow = float(np.mean(model_slow.sigma_sq))

        assert avg_sigma_quick < avg_sigma_normal
        assert avg_sigma_normal < avg_sigma_slow

        # The effect should be especially pronounced on the dimensions most
        # aligned with the difference vector d = e_winner - e_loser
        d = e_winner - e_loser
        top_dims = np.argsort(np.abs(d))[-10:]

        sigma_top_quick = float(np.mean(model_quick.sigma_sq[top_dims]))
        sigma_top_normal = float(np.mean(model_normal.sigma_sq[top_dims]))
        sigma_top_slow = float(np.mean(model_slow.sigma_sq[top_dims]))

        assert sigma_top_quick < sigma_top_normal
        assert sigma_top_normal < sigma_top_slow


# ===========================================================================
# Additional edge case tests
# ===========================================================================


class TestBayesianPreferenceModelEdgeCases:
    """Additional edge-case tests for robustness."""

    def test_warm_start_too_few_scenes(self) -> None:
        """warm_start_from_engagement should be a no-op with < 3 scenes."""
        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        rng = np.random.default_rng(1010)

        embs: dict[int, NDArray[np.float32]] = {
            0: rng.standard_normal(128).astype(np.float32),
            1: rng.standard_normal(128).astype(np.float32),
        }
        scores: dict[int, float] = {0: 1.0, 1: 2.0}

        model.warm_start_from_engagement(embs, scores)

        # mu should still be zeros since we had < 3 scenes
        np.testing.assert_array_equal(model.mu, np.zeros(128, dtype=np.float32))

    def test_warm_start_identical_scores(self) -> None:
        """warm_start should be a no-op when all engagement scores are equal."""
        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        rng = np.random.default_rng(1111)

        embs: dict[int, NDArray[np.float32]] = {}
        for i in range(10):
            v = rng.standard_normal(128).astype(np.float32)
            v /= np.linalg.norm(v)
            embs[i] = v

        scores: dict[int, float] = dict.fromkeys(range(10), 5.0)  # All identical

        model.warm_start_from_engagement(embs, scores)

        # mu should still be zeros since score_std ~ 0
        np.testing.assert_array_equal(model.mu, np.zeros(128, dtype=np.float32))

    def test_convergence_metrics_empty_embeddings(self) -> None:
        """get_convergence_metrics should handle empty embedding dict."""
        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        metrics = model.get_convergence_metrics({})

        assert metrics.avg_sigma == 3.0
        assert metrics.max_sigma_top50 == 3.0
        assert metrics.n_comparisons == 0

    def test_phase_inference(self) -> None:
        """_infer_phase transitions correctly through phases (0-9 BROAD, 10-27 REFINE, 28+ BOUNDARY)."""
        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)

        # 0 comparisons -> BROAD
        assert model._infer_phase() == ComparisonPhase.BROAD

        model.n_comparisons = 9
        assert model._infer_phase() == ComparisonPhase.BROAD

        model.n_comparisons = 10
        assert model._infer_phase() == ComparisonPhase.REFINE

        model.n_comparisons = 27
        assert model._infer_phase() == ComparisonPhase.REFINE

        model.n_comparisons = 28
        assert model._infer_phase() == ComparisonPhase.BOUNDARY

        model.n_comparisons = 100
        assert model._infer_phase() == ComparisonPhase.BOUNDARY

    def test_predict_score_returns_tuple(self) -> None:
        """predict_score returns (mean_score, uncertainty) tuple."""
        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        rng = np.random.default_rng(1212)
        emb = rng.standard_normal(128).astype(np.float32)
        emb /= np.linalg.norm(emb)

        result = model.predict_score(emb)

        assert isinstance(result, tuple)
        assert len(result) == 2
        score, uncertainty = result
        assert isinstance(score, float)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0.0

    def test_multiple_updates_n_comparisons_count(self) -> None:
        """n_comparisons accurately counts all updates."""
        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        rng = np.random.default_rng(1313)

        for _i in range(25):
            e_a = rng.standard_normal(128).astype(np.float32)
            e_b = rng.standard_normal(128).astype(np.float32)
            model.update(e_a, e_b)

        assert model.n_comparisons == 25

    def test_sigmoid_stability(self) -> None:
        """_sigmoid should not overflow for extreme inputs."""
        # Very large positive
        assert _sigmoid(1000.0) == pytest.approx(1.0, abs=1e-10)
        # Very large negative
        assert _sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-10)
        # Zero
        assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-10)
        # Moderate values
        assert 0.0 < _sigmoid(5.0) < 1.0
        assert 0.0 < _sigmoid(-5.0) < 1.0


class TestPairSelectionEngineEdgeCases:
    """Edge case tests for pair selection."""

    def test_too_few_scenes(self) -> None:
        """Engine returns empty list when fewer than 2 scenes."""
        model = BayesianPreferenceModel(dims=128, noise_variance=1.0)
        rng = np.random.default_rng(1414)
        embs: dict[int, NDArray[np.float32]] = {
            0: rng.standard_normal(128).astype(np.float32),
        }

        engine = _make_pair_selector(model, embs)
        pairs = engine.select_pairs(
            n_pairs=5,
            phase=ComparisonPhase.BROAD,
            clusters=None,
            compared_pairs=set(),
            exploration_rate=0.0,
        )

        assert pairs == []

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Cosine similarity of identical vectors should be 1.0."""
        rng = np.random.default_rng(1515)
        v = rng.standard_normal(128).astype(np.float32)
        v /= np.linalg.norm(v)

        sim = PairSelectionEngine._cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Cosine similarity of orthogonal vectors should be 0.0."""
        v1 = np.zeros(128, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(128, dtype=np.float32)
        v2[1] = 1.0

        sim = PairSelectionEngine._cosine_similarity(v1, v2)
        assert sim == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_zero_vector(self) -> None:
        """Cosine similarity with a zero vector should be 0.0."""
        rng = np.random.default_rng(1616)
        v = rng.standard_normal(128).astype(np.float32)
        zero = np.zeros(128, dtype=np.float32)

        sim = PairSelectionEngine._cosine_similarity(v, zero)
        assert sim == 0.0
