"""Intelligent pair selection and scene selection for preference learning.

Provides two selection modes:

- **Pair selection** (``select_pairs``): generates A-vs-B pairs for
  pairwise comparison UI.
- **Scene selection** (``select_swipe_scenes``): selects individual scenes
  for single-scene swipe UI (Tinder-style like/dislike/skip).

Both modes use a three-phase strategy (BROAD / REFINE / BOUNDARY) that
progressively refines the preference model.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from stash_ai.preferences.types import ComparisonPhase, PreferencePair

if TYPE_CHECKING:
    from stash_ai.preferences.model import BayesianPreferenceModel


# ---------------------------------------------------------------------------
# Cluster info (minimal struct for pair selection)
# ---------------------------------------------------------------------------


@dataclass
class ClusterInfo:
    """Minimal cluster info needed for pair selection.

    Mirrors the taste-map cluster output but only carries what the pair
    selector needs -- scene membership, centroid embedding, and engagement
    share (fraction of total engagement within this cluster).
    """

    cluster_id: int
    scene_ids: list[int]
    centroid: NDArray[np.float32]
    engagement_share: float = 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CANDIDATE_PAIRS: int = 500
"""Hard cap on candidate pair enumeration to keep selection responsive."""

_DEFAULT_BROAD_K: int = 5
"""Default cluster count when no clusters are provided for broad phase."""

_REFINE_SIMILARITY_LOW: float = 0.75
"""Lower bound of cosine similarity for refine-phase candidate pairs."""

_REFINE_SIMILARITY_HIGH: float = 0.95
"""Upper bound of cosine similarity for refine-phase candidate pairs."""


# ---------------------------------------------------------------------------
# PairSelectionEngine
# ---------------------------------------------------------------------------


class PairSelectionEngine:
    """Selects maximally informative scene pairs for preference comparison.

    Uses a three-phase strategy:

    - **Phase 1 (BROAD):** Inter-cluster representative comparisons.  Picks
      the scene closest to each cluster centroid and builds a round-robin
      schedule so every cluster is compared against every other.  If no
      clusters are supplied, ad-hoc k-means is run to partition the embedding
      space.

    - **Phase 2 (REFINE):** Similar pairs within preferred clusters.  Pairs
      are drawn from scenes that the current model scores highly, keeping
      cosine similarity in the ``[0.75, 0.95]`` sweet-spot so the user can
      make meaningful distinctions.

    - **Phase 3 (BOUNDARY):** Uncertainty sampling at the decision boundary.
      Computes expected information gain for candidate pairs based on the
      model's posterior variance and selects the most informative ones.

    Integrates with existing taste-map clusters and scene embeddings.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        model: BayesianPreferenceModel,
        embeddings: dict[int, NDArray[np.float32]],
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialise the pair selector.

        Args:
            model: The Bayesian preference model (provides ``predict_score``
                and posterior parameters for information-gain calculation).
            embeddings: Mapping of ``scene_id`` -> embedding vector.  Only
                scenes present in this dict can be selected.
            log_callback: Optional ``(message, level)`` logger.
        """
        self._model: BayesianPreferenceModel = model
        self._embeddings: dict[int, NDArray[np.float32]] = embeddings
        self._log: Callable[[str, str], None] = log_callback or (lambda msg, level: None)

        # Sorted scene id list for deterministic iteration
        self._scene_ids: list[int] = sorted(embeddings.keys())

    # -------------------------------------------------------------- logging

    def _info(self, msg: str) -> None:
        self._log(msg, "info")

    def _debug(self, msg: str) -> None:
        self._log(msg, "debug")

    # -------------------------------------------------- public entry point

    def select_pairs(
        self,
        n_pairs: int,
        phase: ComparisonPhase,
        clusters: list[ClusterInfo] | None = None,
        compared_pairs: set[tuple[int, int]] | None = None,
        exploration_rate: float = 0.2,
    ) -> list[PreferencePair]:
        """Select *n_pairs* informative pairs for the given phase.

        Args:
            n_pairs: Number of pairs to return.
            phase: Current comparison phase.
            clusters: Optional cluster info from the taste map.  Used in the
                BROAD phase for inter-cluster scheduling; ignored otherwise.
            compared_pairs: Set of ``(min_id, max_id)`` tuples that have
                already been compared.  These are excluded from selection.
            exploration_rate: Epsilon for epsilon-greedy exploration.  With
                this probability a random under-explored pair is returned
                instead of the phase-specific selection.

        Returns:
            List of :class:`PreferencePair` objects (may be shorter than
            *n_pairs* if not enough valid candidates remain).
        """
        if len(self._scene_ids) < 2:
            self._info("Not enough scenes for pair selection")
            return []

        compared: set[tuple[int, int]] = compared_pairs or set()
        result: list[PreferencePair] = []

        # Determine how many pairs are exploration vs exploitation
        n_explore = 0
        if exploration_rate > 0:
            n_explore = sum(1 for _ in range(n_pairs) if random.random() < exploration_rate)
        n_exploit = n_pairs - n_explore

        # Phase-specific exploitation pairs
        if n_exploit > 0:
            phase_pairs = self._select_phase_pairs(n_exploit, phase, clusters, compared)
            result.extend(phase_pairs)
            # Track newly selected pairs so exploration doesn't duplicate
            for p in phase_pairs:
                compared.add(self._normalize_pair(p.scene_a_id, p.scene_b_id))

        # Exploration pairs (random from underexplored scenes)
        if n_explore > 0:
            explore_pairs = self._select_exploration_pairs(n_explore, compared)
            result.extend(explore_pairs)

        self._info(
            f"[PairSelector] Selected {len(result)} pairs "
            f"(phase={phase.value}, exploit={n_exploit}, explore={n_explore})"
        )
        return result

    # -------------------------------------------- swipe scene selection

    def select_swipe_scenes(
        self,
        n_scenes: int,
        phase: ComparisonPhase,
        clusters: list[ClusterInfo] | None = None,
        exploration_rate: float = 0.2,
        negative_rate: float = 0.1,
        pure_random: bool = False,
    ) -> list[int]:
        """Select individual scenes for single-scene swipe mode.

        Unlike :meth:`select_pairs` which returns A-vs-B pairs, this method
        returns a flat list of scene IDs -- one per swipe card.  The
        three-phase strategy still applies:

        - **BROAD**: Sample one scene per cluster (round-robin across clusters)
          for maximum diversity across the embedding space.
        - **REFINE**: Pick scenes that the model already scores highly so the
          user refines preferences within the preferred region.
        - **BOUNDARY**: Pick scenes with highest posterior uncertainty -- the
          ones whose rating would teach the model the most.

        Epsilon-greedy exploration mixes in random under-explored scenes.
        Negative sampling mixes in low-scored scenes so the model also learns
        what the user dislikes (only active after BROAD phase, when the model
        has enough signal to rank scenes meaningfully).

        Args:
            n_scenes: Number of scenes to return.
            phase: Current comparison phase.
            clusters: Optional cluster info for BROAD phase.
            exploration_rate: Fraction of scenes chosen at random for
                exploration (default 0.2).
            negative_rate: Fraction of scenes chosen from predicted-disliked
                region (default 0.1).  Only active after BROAD phase.
            pure_random: If True, bypass all phase/cluster logic and select
                scenes uniformly at random. No bias from clusters or model.

        Returns:
            List of scene IDs (may be shorter than *n_scenes* if the pool
            is exhausted).
        """
        if not self._scene_ids:
            self._info("[SwipeSelector] No scenes available")
            return []

        # Pure random mode: bypass all phase logic, just sample uniformly
        if pure_random:
            return self._select_pure_random_scenes(n_scenes)

        # Negative sampling only after BROAD — model needs signal first
        effective_negative_rate = negative_rate if phase != ComparisonPhase.BROAD else 0.0

        # Allocate budget: exploitation + exploration + negative = n_scenes
        n_negative = 0
        n_explore = 0
        for _ in range(n_scenes):
            r = random.random()
            if r < effective_negative_rate:
                n_negative += 1
            elif r < effective_negative_rate + exploration_rate:
                n_explore += 1
        n_exploit = n_scenes - n_explore - n_negative

        selected: list[int] = []
        selected_set: set[int] = set()

        # Phase-specific exploitation scenes
        if n_exploit > 0:
            phase_scenes = self._select_phase_scenes(
                n_exploit,
                phase,
                clusters,
                selected_set,
            )
            for sid in phase_scenes:
                if sid not in selected_set:
                    selected.append(sid)
                    selected_set.add(sid)

        # Negative sampling: scenes the model predicts the user will dislike
        if n_negative > 0:
            negative_scenes = self._select_negative_scenes(
                n_negative,
                selected_set,
            )
            for sid in negative_scenes:
                if sid not in selected_set:
                    selected.append(sid)
                    selected_set.add(sid)

        # Exploration: random from underexplored scenes
        if n_explore > 0:
            explore_scenes = self._select_exploration_scenes(
                n_explore,
                selected_set,
            )
            for sid in explore_scenes:
                if sid not in selected_set:
                    selected.append(sid)
                    selected_set.add(sid)

        self._info(
            f"[SwipeSelector] Selected {len(selected)} scenes "
            f"(phase={phase.value}, exploit={n_exploit}, "
            f"negative={n_negative}, explore={n_explore})"
        )
        return selected

    # ------------------------------------------------ swipe phase dispatch

    def _select_phase_scenes(
        self,
        n_scenes: int,
        phase: ComparisonPhase,
        clusters: list[ClusterInfo] | None,
        exclude: set[int],
    ) -> list[int]:
        """Dispatch to the correct phase-specific scene selector."""
        if phase == ComparisonPhase.BROAD:
            return self._select_broad_scenes(n_scenes, clusters, exclude)
        elif phase == ComparisonPhase.REFINE:
            return self._select_refine_scenes(n_scenes, exclude)
        elif phase == ComparisonPhase.BOUNDARY:
            return self._select_boundary_scenes(n_scenes, exclude)
        else:  # Defensive fallback for future enum values
            return self._select_broad_scenes(n_scenes, clusters, exclude)  # type: ignore[unreachable]

    def _select_broad_scenes(
        self,
        n_scenes: int,
        clusters: list[ClusterInfo] | None,
        exclude: set[int],
    ) -> list[int]:
        """BROAD phase: sample one scene per cluster, round-robin.

        Ensures diversity across the embedding space.  When clusters are
        exhausted, fills remaining slots from the full pool randomly.
        """
        if clusters is None or len(clusters) < 2:
            clusters = self._create_adhoc_clusters()

        if not clusters:
            return self._select_random_scenes(n_scenes, exclude)

        # Shuffle clusters for variety across sessions
        shuffled_clusters = list(clusters)
        random.shuffle(shuffled_clusters)

        selected: list[int] = []
        selected_set: set[int] = set(exclude)

        # Round-robin: pick one scene per cluster per pass
        passes = 0
        max_passes = (n_scenes // len(shuffled_clusters)) + 2
        while len(selected) < n_scenes and passes < max_passes:
            passes += 1
            for cluster in shuffled_clusters:
                if len(selected) >= n_scenes:
                    break
                # Get valid members of this cluster
                valid_ids = [
                    sid
                    for sid in cluster.scene_ids
                    if sid in self._embeddings and sid not in selected_set
                ]
                if not valid_ids:
                    continue
                # Pick randomly from cluster members
                chosen = random.choice(valid_ids)
                selected.append(chosen)
                selected_set.add(chosen)

        # Backfill if clusters ran out
        if len(selected) < n_scenes:
            backfill = self._select_random_scenes(
                n_scenes - len(selected),
                selected_set,
            )
            selected.extend(backfill)

        return selected[:n_scenes]

    def _select_refine_scenes(
        self,
        n_scenes: int,
        exclude: set[int],
    ) -> list[int]:
        """REFINE phase: pick scenes the model scores highly.

        Selects from the top half of scored scenes with some randomness
        to explore the preferred region.
        """
        scored: list[tuple[int, float]] = [
            (sid, self._model.predict_score(self._embeddings[sid])[0])
            for sid in self._scene_ids
            if sid not in exclude
        ]
        scored.sort(key=lambda t: t[1], reverse=True)

        # Use top half as candidate pool, weighted sample
        pool_size = max(n_scenes * 2, len(scored) // 2)
        pool = scored[:pool_size]
        if not pool:
            return []

        # Softmax-weighted sampling (higher scores more likely)
        scores_arr = np.array([s for _, s in pool], dtype=np.float64)
        # Temperature-scaled softmax for some exploration
        temperature = 0.5
        scores_arr = scores_arr / temperature
        scores_arr -= scores_arr.max()  # numerical stability
        weights = np.exp(scores_arr)
        weights /= weights.sum()

        rng = np.random.default_rng()
        n_to_sample = min(n_scenes, len(pool))
        indices = rng.choice(
            len(pool),
            size=n_to_sample,
            replace=False,
            p=weights,
        )
        return [pool[int(i)][0] for i in indices]

    def _select_boundary_scenes(
        self,
        n_scenes: int,
        exclude: set[int],
    ) -> list[int]:
        """BOUNDARY phase: pick scenes with highest posterior uncertainty.

        These are the scenes whose rating would teach the model the most.
        """
        uncertainties: list[tuple[int, float]] = []
        for sid in self._scene_ids:
            if sid in exclude:
                continue
            _, sigma = self._model.predict_score(self._embeddings[sid])
            uncertainties.append((sid, sigma))

        # Sort by uncertainty descending
        uncertainties.sort(key=lambda t: t[1], reverse=True)

        return [sid for sid, _ in uncertainties[:n_scenes]]

    def _select_negative_scenes(
        self,
        n_scenes: int,
        exclude: set[int],
    ) -> list[int]:
        """Select scenes the model predicts the user will dislike.

        Samples from the bottom half of scored scenes with inverse-softmax
        weighting (lower scores more likely).  This teaches the model where
        the dislike boundary is, which is just as informative as showing
        liked scenes.
        """
        scored: list[tuple[int, float]] = [
            (sid, self._model.predict_score(self._embeddings[sid])[0])
            for sid in self._scene_ids
            if sid not in exclude
        ]
        if not scored:
            return []

        scored.sort(key=lambda t: t[1])  # ascending — lowest first

        # Use bottom half as candidate pool
        pool_size = max(n_scenes * 2, len(scored) // 2)
        pool = scored[:pool_size]
        if not pool:
            return []

        # Inverse-softmax: negate scores so lower scores get higher weight
        scores_arr = np.array([-s for _, s in pool], dtype=np.float64)
        temperature = 0.5
        scores_arr = scores_arr / temperature
        scores_arr -= scores_arr.max()  # numerical stability
        weights = np.exp(scores_arr)
        weights /= weights.sum()

        rng = np.random.default_rng()
        n_to_sample = min(n_scenes, len(pool))
        indices = rng.choice(
            len(pool),
            size=n_to_sample,
            replace=False,
            p=weights,
        )
        return [pool[int(i)][0] for i in indices]

    def _select_exploration_scenes(
        self,
        n_scenes: int,
        exclude: set[int],
    ) -> list[int]:
        """Select random scenes for epsilon-greedy exploration."""
        return self._select_random_scenes(n_scenes, exclude)

    def _select_random_scenes(
        self,
        n_scenes: int,
        exclude: set[int],
    ) -> list[int]:
        """Uniformly random scene selection (excluding given set)."""
        candidates = [sid for sid in self._scene_ids if sid not in exclude]
        if not candidates:
            return []
        random.shuffle(candidates)
        return candidates[:n_scenes]

    def _select_pure_random_scenes(self, n_scenes: int) -> list[int]:
        """Select scenes uniformly at random with no bias.

        Unlike phase-specific selection, this method:
        - Does NOT use clusters or cluster centroids
        - Does NOT use model predictions or uncertainty
        - Does NOT weight by engagement or any other signal
        - Gives every scene an equal probability of selection

        Use this to bootstrap training without any pre-existing bias.
        """
        self._info(f"[PureRandom] Selecting {n_scenes} scenes uniformly at random")
        result = self._select_random_scenes(n_scenes, exclude=set())
        self._info(f"[PureRandom] Selected {len(result)} scenes")
        return result

    # -------------------------------------------------------- phase dispatch

    def _select_phase_pairs(
        self,
        n_pairs: int,
        phase: ComparisonPhase,
        clusters: list[ClusterInfo] | None,
        compared: set[tuple[int, int]],
    ) -> list[PreferencePair]:
        """Dispatch to the correct phase-specific selector."""
        if phase == ComparisonPhase.BROAD:
            return self._select_broad_pairs(n_pairs, clusters, compared)
        elif phase == ComparisonPhase.REFINE:
            return self._select_refine_pairs(n_pairs, compared)
        elif phase == ComparisonPhase.BOUNDARY:
            return self._select_boundary_pairs(n_pairs, compared)
        else:  # Defensive fallback for future enum values
            self._info(f"[PairSelector] Unknown phase {phase}, falling back to broad")  # type: ignore[unreachable]
            return self._select_broad_pairs(n_pairs, clusters, compared)

    # ====================================================================
    # Phase 1 - BROAD: inter-cluster representative comparisons
    # ====================================================================

    def _select_broad_pairs(
        self,
        n_pairs: int,
        clusters: list[ClusterInfo] | None,
        compared: set[tuple[int, int]],
    ) -> list[PreferencePair]:
        """Phase 1: compare representative scenes across clusters.

        If clusters are provided, pick the scene closest to each centroid and
        generate a round-robin tournament among representatives.  If no
        clusters are given, run quick k-means on the fly to partition the
        embedding space.
        """
        if clusters is None or len(clusters) < 2:
            clusters = self._create_adhoc_clusters()

        if len(clusters) < 2:
            # Fallback: just return diverse random pairs
            return self._select_random_diverse_pairs(n_pairs, compared)

        # Find representative for each cluster (closest to centroid)
        representatives: list[int] = []
        for cluster in clusters:
            valid_ids = [sid for sid in cluster.scene_ids if sid in self._embeddings]
            if not valid_ids:
                continue
            centroid = cluster.centroid
            best_id = min(
                valid_ids,
                key=lambda sid: float(np.linalg.norm(self._embeddings[sid] - centroid)),
            )
            representatives.append(best_id)

        if len(representatives) < 2:
            return self._select_random_diverse_pairs(n_pairs, compared)

        # Round-robin pairs among representatives
        candidate_pairs: list[tuple[int, int]] = []
        for i in range(len(representatives)):
            for j in range(i + 1, len(representatives)):
                pair = self._normalize_pair(representatives[i], representatives[j])
                if pair not in compared:
                    candidate_pairs.append(pair)

        # Shuffle so we don't always start with the same cluster pair
        random.shuffle(candidate_pairs)

        result: list[PreferencePair] = []
        for a_id, b_id in candidate_pairs[:n_pairs]:
            sim = self._cosine_similarity(self._embeddings[a_id], self._embeddings[b_id])
            prob, _ = self._model.predict_comparison(self._embeddings[a_id], self._embeddings[b_id])
            result.append(
                PreferencePair(
                    scene_a_id=a_id,
                    scene_b_id=b_id,
                    phase=ComparisonPhase.BROAD,
                    predicted_probability=prob,
                    information_gain=0.0,
                    similarity=sim,
                )
            )

        # If we exhausted round-robin pairs, backfill with random members
        if len(result) < n_pairs:
            remaining = n_pairs - len(result)
            for existing_pair in result:
                compared.add(
                    self._normalize_pair(existing_pair.scene_a_id, existing_pair.scene_b_id)
                )
            backfill = self._select_random_diverse_pairs(remaining, compared)
            for p in backfill:
                # Re-tag as BROAD phase
                result.append(
                    PreferencePair(
                        scene_a_id=p.scene_a_id,
                        scene_b_id=p.scene_b_id,
                        phase=ComparisonPhase.BROAD,
                        predicted_probability=p.predicted_probability,
                        information_gain=p.information_gain,
                        similarity=p.similarity,
                    )
                )

        return result[:n_pairs]

    def _create_adhoc_clusters(self) -> list[ClusterInfo]:
        """Run quick k-means to create ad-hoc clusters when none provided."""
        n_scenes = len(self._scene_ids)
        if n_scenes < 2:
            return []

        k = min(_DEFAULT_BROAD_K, max(2, int(math.sqrt(n_scenes))))

        # Build embedding matrix
        ids = self._scene_ids
        matrix = np.array([self._embeddings[sid] for sid in ids], dtype=np.float32)

        # Simple k-means (limited iterations for speed)
        labels = self._quick_kmeans(matrix, k, max_iter=20)

        clusters: list[ClusterInfo] = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            cluster_scene_ids = [ids[i] for i in range(n_scenes) if mask[i]]
            if not cluster_scene_ids:
                continue
            centroid = matrix[mask].mean(axis=0)
            clusters.append(
                ClusterInfo(
                    cluster_id=cluster_id,
                    scene_ids=cluster_scene_ids,
                    centroid=centroid,
                )
            )

        return clusters

    @staticmethod
    def _quick_kmeans(
        data: NDArray[np.float32],
        k: int,
        max_iter: int = 20,
    ) -> NDArray[np.intp]:
        """Minimal k-means implementation for ad-hoc clustering.

        Returns cluster labels for each row of *data*.
        """
        n = data.shape[0]
        if n <= k:
            return np.arange(n, dtype=np.intp)

        # k-means++ initialisation
        rng = np.random.default_rng()
        centroids = np.empty((k, data.shape[1]), dtype=np.float32)
        first_idx = rng.integers(0, n)
        centroids[0] = data[first_idx]

        for c in range(1, k):
            # Squared distances to nearest existing centroid
            dists = np.min(
                np.sum((data[:, None, :] - centroids[None, :c, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / (dists.sum() + 1e-12)
            chosen = rng.choice(n, p=probs)
            centroids[c] = data[chosen]

        labels = np.zeros(n, dtype=np.intp)
        for _ in range(max_iter):
            # Assign
            diffs = data[:, None, :] - centroids[None, :, :]  # (n, k, d)
            sq_dists = np.sum(diffs**2, axis=2)  # (n, k)
            new_labels = np.argmin(sq_dists, axis=1).astype(np.intp)

            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update centroids
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centroids[c] = data[mask].mean(axis=0)

        return labels

    # ====================================================================
    # Phase 2 - REFINE: similar pairs within preferred clusters
    # ====================================================================

    def _select_refine_pairs(
        self,
        n_pairs: int,
        compared: set[tuple[int, int]],
    ) -> list[PreferencePair]:
        """Phase 2: compare similar scenes within the preferred region.

        Finds scenes with high model scores, then pairs them with nearby
        (cosine similarity in [0.75, 0.95]) scenes for fine-grained
        preference elicitation.
        """
        # Score all scenes via the preference model
        scored: list[tuple[int, float]] = [
            (sid, self._model.predict_score(self._embeddings[sid])[0]) for sid in self._scene_ids
        ]
        scored.sort(key=lambda t: t[1], reverse=True)

        # Use the top half of scored scenes as the candidate pool
        pool_size = max(10, len(scored) // 2)
        pool_ids: list[int] = [sid for sid, _ in scored[:pool_size]]

        # Sample candidate pairs from the pool
        candidates: list[tuple[int, int, float]] = []  # (a, b, similarity)
        sampled: int = 0

        # Iterate top-scored scenes, pair with nearby pool members
        for i, sid_a in enumerate(pool_ids):
            if sampled >= _MAX_CANDIDATE_PAIRS:
                break
            emb_a = self._embeddings[sid_a]
            # Compare against remaining pool members
            for sid_b in pool_ids[i + 1 :]:
                if sampled >= _MAX_CANDIDATE_PAIRS:
                    break
                pair = self._normalize_pair(sid_a, sid_b)
                if pair in compared:
                    continue
                sim = self._cosine_similarity(emb_a, self._embeddings[sid_b])
                if _REFINE_SIMILARITY_LOW <= sim <= _REFINE_SIMILARITY_HIGH:
                    candidates.append((sid_a, sid_b, sim))
                    sampled += 1

        # Sort by similarity descending -- closest-but-distinct pairs first
        candidates.sort(key=lambda t: t[2], reverse=True)

        result: list[PreferencePair] = []
        for a_id, b_id, sim in candidates[:n_pairs]:
            prob, _ = self._model.predict_comparison(self._embeddings[a_id], self._embeddings[b_id])
            result.append(
                PreferencePair(
                    scene_a_id=a_id,
                    scene_b_id=b_id,
                    phase=ComparisonPhase.REFINE,
                    predicted_probability=prob,
                    information_gain=0.0,
                    similarity=sim,
                )
            )

        # Backfill if not enough candidates in the similarity sweet-spot
        if len(result) < n_pairs:
            for existing_pair in result:
                compared.add(
                    self._normalize_pair(existing_pair.scene_a_id, existing_pair.scene_b_id)
                )
            backfill = self._select_random_diverse_pairs(n_pairs - len(result), compared)
            for p in backfill:
                result.append(
                    PreferencePair(
                        scene_a_id=p.scene_a_id,
                        scene_b_id=p.scene_b_id,
                        phase=ComparisonPhase.REFINE,
                        predicted_probability=p.predicted_probability,
                        information_gain=p.information_gain,
                        similarity=p.similarity,
                    )
                )

        return result[:n_pairs]

    # ====================================================================
    # Phase 3 - BOUNDARY: uncertainty / information-gain sampling
    # ====================================================================

    def _select_boundary_pairs(
        self,
        n_pairs: int,
        compared: set[tuple[int, int]],
    ) -> list[PreferencePair]:
        """Phase 3: select pairs that maximise expected information gain.

        For each candidate pair ``(i, j)`` the information gain is
        approximated as::

            IG(i, j) = 0.5 * log(1 + d^T diag(sigma^2) d / noise_var)

        where ``d = e_i - e_j`` is the embedding difference, ``sigma^2`` is
        the diagonal posterior variance, and ``noise_var`` is the observation
        noise.

        To keep computation tractable we sample at most
        ``_MAX_CANDIDATE_PAIRS`` random un-compared pairs and pick the top
        *n_pairs* by information gain.
        """
        sigma_sq: NDArray[np.float32] = self._model.sigma_sq
        noise_var: float = self._model.noise_variance

        # Build candidate pool by random sampling
        candidates = self._sample_uncomared_pairs(_MAX_CANDIDATE_PAIRS, compared)

        if not candidates:
            self._debug("[PairSelector] No uncompared pairs for boundary phase")
            return []

        # Score each candidate by information gain
        scored: list[tuple[int, int, float, float]] = []
        for a_id, b_id in candidates:
            d = self._embeddings[a_id] - self._embeddings[b_id]
            # d^T diag(sigma^2) d  == sum(d_i^2 * sigma_i^2)
            quad_form: float = float(np.sum(d * d * sigma_sq))
            ig = 0.5 * math.log1p(quad_form / (noise_var + 1e-12))
            sim = self._cosine_similarity(self._embeddings[a_id], self._embeddings[b_id])
            scored.append((a_id, b_id, ig, sim))

        # Sort by information gain descending
        scored.sort(key=lambda t: t[2], reverse=True)

        result: list[PreferencePair] = []
        for a_id, b_id, ig, sim in scored[:n_pairs]:
            prob, _ = self._model.predict_comparison(self._embeddings[a_id], self._embeddings[b_id])
            result.append(
                PreferencePair(
                    scene_a_id=a_id,
                    scene_b_id=b_id,
                    phase=ComparisonPhase.BOUNDARY,
                    predicted_probability=prob,
                    information_gain=ig,
                    similarity=sim,
                )
            )

        return result

    # ====================================================================
    # Exploration pairs (epsilon-greedy)
    # ====================================================================

    def _select_exploration_pairs(
        self,
        n_pairs: int,
        compared: set[tuple[int, int]],
    ) -> list[PreferencePair]:
        """Select random pairs prioritising underexplored scenes.

        A scene's exploration count is the number of comparisons it appears
        in.  Pairs are drawn preferentially from the least-compared scenes.
        """
        # Count per-scene comparisons
        comparison_counts: Counter[int] = Counter()
        for a_id, b_id in compared:
            comparison_counts[a_id] += 1
            comparison_counts[b_id] += 1

        # Assign each scene an inverse-frequency weight
        max_count = max(comparison_counts.values()) if comparison_counts else 0
        weights: dict[int, float] = {}
        for sid in self._scene_ids:
            count = comparison_counts.get(sid, 0)
            # Scenes with zero comparisons get highest weight
            weights[sid] = float(max_count - count + 1)

        total_weight = sum(weights.values())
        if total_weight <= 0:
            # Uniform fallback
            for sid in self._scene_ids:
                weights[sid] = 1.0
            total_weight = float(len(self._scene_ids))

        # Weighted random sampling of pairs
        scene_list = list(weights.keys())
        scene_weights = np.array([weights[sid] for sid in scene_list], dtype=np.float64)
        scene_probs = scene_weights / scene_weights.sum()

        rng = np.random.default_rng()
        result: list[PreferencePair] = []
        attempts = 0
        max_attempts = n_pairs * 20

        while len(result) < n_pairs and attempts < max_attempts:
            attempts += 1
            indices = rng.choice(len(scene_list), size=2, replace=False, p=scene_probs)
            a_id = scene_list[int(indices[0])]
            b_id = scene_list[int(indices[1])]
            pair = self._normalize_pair(a_id, b_id)

            if pair in compared:
                continue

            compared.add(pair)
            sim = self._cosine_similarity(self._embeddings[a_id], self._embeddings[b_id])
            prob, _ = self._model.predict_comparison(self._embeddings[a_id], self._embeddings[b_id])
            result.append(
                PreferencePair(
                    scene_a_id=pair[0],
                    scene_b_id=pair[1],
                    phase=ComparisonPhase.BROAD,  # tag as broad for exploration
                    predicted_probability=prob,
                    information_gain=0.0,
                    similarity=sim,
                )
            )

        return result

    # ====================================================================
    # Helpers
    # ====================================================================

    def _select_random_diverse_pairs(
        self,
        n_pairs: int,
        compared: set[tuple[int, int]],
    ) -> list[PreferencePair]:
        """Fallback: randomly sample un-compared pairs."""
        candidates = self._sample_uncomared_pairs(min(n_pairs * 5, _MAX_CANDIDATE_PAIRS), compared)
        random.shuffle(candidates)

        result: list[PreferencePair] = []
        for a_id, b_id in candidates[:n_pairs]:
            sim = self._cosine_similarity(self._embeddings[a_id], self._embeddings[b_id])
            prob, _ = self._model.predict_comparison(self._embeddings[a_id], self._embeddings[b_id])
            result.append(
                PreferencePair(
                    scene_a_id=a_id,
                    scene_b_id=b_id,
                    phase=ComparisonPhase.BROAD,
                    predicted_probability=prob,
                    information_gain=0.0,
                    similarity=sim,
                )
            )

        return result

    def _sample_uncomared_pairs(
        self,
        max_pairs: int,
        compared: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Randomly sample up to *max_pairs* un-compared pairs.

        Uses rejection sampling which is efficient when the compared set is
        small relative to the total number of possible pairs.
        """
        n = len(self._scene_ids)
        total_possible = n * (n - 1) // 2

        # If most pairs have been compared, enumerate remaining
        if len(compared) > total_possible * 0.7:
            remaining: list[tuple[int, int]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    pair = (self._scene_ids[i], self._scene_ids[j])
                    if pair not in compared:
                        remaining.append(pair)
                    if len(remaining) >= max_pairs:
                        break
                if len(remaining) >= max_pairs:
                    break
            random.shuffle(remaining)
            return remaining[:max_pairs]

        # Rejection sampling
        rng = random.Random()
        result: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = max_pairs * 10

        while len(result) < max_pairs and attempts < max_attempts:
            attempts += 1
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            pair = self._normalize_pair(self._scene_ids[i], self._scene_ids[j])
            if pair in compared or pair in seen:
                continue
            seen.add(pair)
            result.append(pair)

        return result

    @staticmethod
    def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """Cosine similarity between two vectors (dot product of L2-normed)."""
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _normalize_pair(a: int, b: int) -> tuple[int, int]:
        """Return ``(min, max)`` for consistent pair hashing."""
        return (a, b) if a < b else (b, a)
