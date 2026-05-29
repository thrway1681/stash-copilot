"""Taste clustering engine using K-Means with silhouette score optimization.

Groups engaged scenes into taste clusters by embedding similarity,
computes weighted centroids, and auto-labels via tag embedding matching.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from stash_ai.embeddings.tag_vocabulary import TagVocabulary
    from stash_ai.recommendations.types import TasteCluster, TasteProfile


MIN_K = 2
MAX_K_DEFAULT = 8
MAX_K_CEILING = 25
FALLBACK_K = 3
MIN_SCENES_FOR_CLUSTERING = 6  # Need at least 6 scenes to form 2 clusters


def _compute_max_k(n_scenes: int) -> int:
    """Scale max k based on dataset size.

    Uses log2(n) * 1.5, clamped to [MAX_K_DEFAULT, MAX_K_CEILING].
    Examples: 100 scenes → 10, 1000 → 15, 12000 → 20, 50000 → 23.
    """
    return min(max(MAX_K_DEFAULT, int(math.log2(max(n_scenes, 1)) * 1.5)), MAX_K_CEILING)


def find_optimal_k(
    embeddings: NDArray[np.float32],
    min_k: int = MIN_K,
    max_k: int | None = None,
    log: Callable[[str, str], None] | None = None,
) -> tuple[int, float]:
    """Find optimal number of clusters using silhouette score.

    Args:
        embeddings: Matrix of shape (n_scenes, n_dims).
        min_k: Minimum clusters to try.
        max_k: Maximum clusters to try (None = auto-scale by dataset size).
        log: Logging callback.

    Returns:
        Tuple of (optimal_k, best_silhouette_score).
    """
    _log = log or (lambda msg, level: None)
    n_samples = len(embeddings)

    if max_k is None:
        max_k = _compute_max_k(n_samples)
        _log(f"Auto max_k={max_k} for {n_samples} scenes", "debug")

    # Cap max_k at n_samples - 1
    max_k = min(max_k, n_samples - 1)
    if max_k < min_k:
        _log(
            f"Too few scenes ({n_samples}) for clustering range {min_k}-{max_k}, using k={min_k}",
            "warning",
        )
        return min_k, 0.0

    best_k = FALLBACK_K
    best_score = -1.0

    # Use sampling for silhouette on large datasets (O(n²) otherwise)
    sil_kwargs: dict[str, object] = {}
    if n_samples > 2000:
        sil_kwargs = {"sample_size": 2000, "random_state": 42}
        _log(f"Using silhouette sampling (n={n_samples} > 2000)", "debug")

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = float(silhouette_score(embeddings, labels, **sil_kwargs))
        _log(f"  k={k}: silhouette={score:.4f}", "debug")

        if score > best_score:
            best_score = score
            best_k = k

    _log(f"Optimal k={best_k} (silhouette={best_score:.4f})", "info")
    return best_k, best_score


def cluster_scenes(
    scene_ids: list[int],
    embeddings: NDArray[np.float32],
    engagement_scores: dict[int, float],
    optimal_k: int,
) -> tuple[list[list[int]], NDArray[np.float32], list[int]]:
    """Run K-Means and compute engagement-weighted centroids.

    Args:
        scene_ids: Scene IDs corresponding to embedding rows.
        embeddings: Matrix of shape (n_scenes, n_dims).
        engagement_scores: Mapping of scene_id -> engagement score.
        optimal_k: Number of clusters.

    Returns:
        Tuple of (cluster_scene_ids, centroids, labels) where:
        - cluster_scene_ids[i] is the list of scene_ids in cluster i
        - centroids is shape (k, n_dims) engagement-weighted centroids
        - labels[i] is the cluster assignment for scene_ids[i]
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings).tolist()

    n_dims = embeddings.shape[1]
    cluster_scene_ids: list[list[int]] = [[] for _ in range(optimal_k)]
    centroids = np.zeros((optimal_k, n_dims), dtype=np.float32)
    centroid_weights = np.zeros(optimal_k, dtype=np.float32)

    for i, scene_id in enumerate(scene_ids):
        cluster_idx = labels[i]
        cluster_scene_ids[cluster_idx].append(scene_id)

        weight = engagement_scores.get(scene_id, 1.0)
        centroids[cluster_idx] += embeddings[i] * weight
        centroid_weights[cluster_idx] += weight

    # Normalize centroids
    for k in range(optimal_k):
        if centroid_weights[k] > 0:
            centroids[k] /= centroid_weights[k]
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(centroids[k])
        if norm > 0:
            centroids[k] /= norm

    return cluster_scene_ids, centroids, labels


def compute_umap_projection(
    embeddings: NDArray[np.float32],
    labels: list[int] | None = None,
    log: Callable[[str, str], None] | None = None,
) -> NDArray[np.float32]:
    """Project embeddings to 3D using UMAP.

    When labels are provided, runs in supervised mode so that points
    sharing the same cluster label are pulled together, improving
    visual separability of clusters in the 3D projection.

    Args:
        embeddings: Matrix of shape (n_scenes, n_dims).
        labels: Per-scene cluster labels for supervised UMAP. Same length
            as embeddings. Pass None for unsupervised projection.
        log: Logging callback.

    Returns:
        3D coordinates of shape (n_scenes, 3).
    """
    import umap

    _log = log or (lambda msg, level: None)
    n_samples = len(embeddings)
    n_neighbors = min(15, max(2, n_samples - 1))
    init_method = "random" if n_samples < 10 else "spectral"

    mode = "supervised" if labels is not None else "unsupervised"
    _log(f"Running UMAP 3D ({mode}): {n_samples} scenes, n_neighbors={n_neighbors}", "info")

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        init=init_method,
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings, y=labels)
    return cast("NDArray[np.float32]", coords.astype(np.float32))


def build_taste_profile(
    scene_ids: list[int],
    embeddings: NDArray[np.float32],
    engagement_scores: dict[int, float],
    tag_vocabulary: TagVocabulary,
    model_key: str,
    log: Callable[[str, str], None] | None = None,
    num_clusters: int | None = None,
) -> TasteProfile:
    """Build a complete taste profile with clustering, labeling, and UMAP.

    Args:
        scene_ids: Scene IDs for profile scenes.
        embeddings: Visual embeddings matrix (n_scenes, n_dims).
        engagement_scores: scene_id -> engagement score mapping.
        tag_vocabulary: Vocabulary for auto-labeling clusters.
        model_key: Embedding model identifier.
        log: Logging callback.
        num_clusters: Fixed number of clusters (None = auto-detect via silhouette).

    Returns:
        Complete TasteProfile with clusters and coordinates.
    """
    from stash_ai.recommendations.types import TasteCluster, TasteProfile

    _log = log or (lambda msg, level: None)
    n_scenes = len(scene_ids)

    if n_scenes < MIN_SCENES_FOR_CLUSTERING:
        _log(
            f"Only {n_scenes} scenes - too few for clustering (need {MIN_SCENES_FOR_CLUSTERING})",
            "warning",
        )
        # Create single cluster with all scenes
        total_engagement = sum(engagement_scores.get(sid, 0) for sid in scene_ids)
        centroid = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        tag_matches = tag_vocabulary.match_cluster_centroid(centroid, top_k=8)
        auto_label = " / ".join(str(m["text"]) for m in tag_matches[:2]) if tag_matches else "Mixed"

        cluster = TasteCluster(
            cluster_id=0,
            centroid=centroid,
            scene_ids=scene_ids,
            engagement_total=total_engagement,
            engagement_share=1.0,
            auto_label=auto_label,
            user_label=None,
            weight_override=None,
            excluded=False,
            tag_matches=tag_matches,  # type: ignore[arg-type]
        )
        return TasteProfile(
            clusters=[cluster],
            optimal_k=1,
            silhouette_score=0.0,
            model_key=model_key,
        )

    # Find optimal k (or use user-specified value)
    if num_clusters is not None:
        optimal_k = num_clusters
        sil_score = 0.0
        _log(f"Using user-specified k={optimal_k}", "info")
    else:
        _log("Finding optimal cluster count...", "info")
        optimal_k, sil_score = find_optimal_k(embeddings, log=_log)

    # Cluster scenes
    _log(f"Clustering {n_scenes} scenes into {optimal_k} clusters...", "info")
    cluster_scene_ids, centroids, _labels = cluster_scenes(
        scene_ids, embeddings, engagement_scores, optimal_k
    )

    # Compute total engagement for share calculation
    total_engagement = sum(engagement_scores.get(sid, 0) for sid in scene_ids)

    # Compute mean centroid for differential labeling
    mean_centroid = np.mean(centroids, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_centroid)
    if norm > 0:
        mean_centroid /= norm

    # Build cluster objects with auto-labels
    clusters: list[TasteCluster] = []
    for k in range(optimal_k):
        cluster_eng = sum(engagement_scores.get(sid, 0) for sid in cluster_scene_ids[k])
        eng_share = cluster_eng / total_engagement if total_engagement > 0 else 1.0 / optimal_k

        # Auto-label: differential (what distinguishes this cluster from others)
        tag_matches = tag_vocabulary.match_cluster_centroid_differential(
            centroids[k], mean_centroid, top_k=8
        )
        auto_label = (
            " / ".join(str(m["text"]) for m in tag_matches[:2])
            if tag_matches
            else f"Cluster {k + 1}"
        )

        clusters.append(
            TasteCluster(
                cluster_id=k,
                centroid=centroids[k],
                scene_ids=cluster_scene_ids[k],
                engagement_total=cluster_eng,
                engagement_share=eng_share,
                auto_label=auto_label,
                user_label=None,
                weight_override=None,
                excluded=False,
                tag_matches=tag_matches,  # type: ignore[arg-type]
            )
        )

    # Sort clusters by engagement share (largest first)
    clusters.sort(key=lambda c: c.engagement_share, reverse=True)
    # Re-assign cluster IDs after sorting
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i

    _log(f"Built {len(clusters)} taste clusters", "info")
    for c in clusters:
        _log(
            f"  Cluster {c.cluster_id}: '{c.auto_label}' - {len(c.scene_ids)} scenes, {c.engagement_share:.1%} engagement",
            "info",
        )

    return TasteProfile(
        clusters=clusters,
        optimal_k=optimal_k,
        silhouette_score=sil_score,
        model_key=model_key,
    )
