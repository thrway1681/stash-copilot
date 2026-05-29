"""Frame embedding analysis for intra-scene similarity and representative frame selection."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict

from ..embeddings.base import BaseImageEmbeddingProvider
from ..embeddings.config import EmbeddingConfig
from ..embeddings.provider import get_embedding_provider
from .frame_extractor import ExtractedFrame, FrameExtractionConfig, FrameExtractor

if TYPE_CHECKING:
    from ..stash_client import StashClient


class FrameEmbedding(TypedDict):
    """Single frame embedding with metadata."""

    index: int  # 1-based frame index
    timestamp: float  # Timestamp in seconds
    embedding: list[float]  # Frame embedding vector


class FrameSimilarityMatrix(TypedDict):
    """Pairwise similarity between frames."""

    frame_indices: list[int]  # Frame indices (1-based)
    timestamps: list[float]  # Frame timestamps
    matrix: list[list[float]]  # NxN similarity matrix (cosine similarity)


class DimensionalityReductionResult(TypedDict):
    """Result of dimensionality reduction on frame embeddings."""

    method: Literal["pca", "tsne", "umap"]
    coordinates: list[list[float]]  # Nx2 coordinates
    frame_indices: list[int]  # Corresponding frame indices
    timestamps: list[float]  # Corresponding timestamps
    explained_variance: float | None  # For PCA only (cumulative 2 components)


class RepresentativeFrameResult(TypedDict):
    """Result of frame selection algorithm."""

    selected_indices: list[int]  # Indices of selected frames (1-based)
    selected_timestamps: list[float]  # Timestamps of selected frames
    cluster_assignments: list[int] | None  # Which cluster each frame belongs to
    selection_method: str  # Algorithm used
    diversity_score: float  # Average pairwise distance of selected frames


class MethodComparisonResult(TypedDict):
    """Result of comparing all selection methods."""

    method: str
    diversity_score: float  # Average pairwise embedding distance (0-2)
    temporal_spread: float  # How evenly frames cover the duration (0-1)
    coverage_score: float  # Fraction of frames within epsilon of a selected frame (0-1)
    combined_score: float  # Weighted combination of metrics (0-1)
    selected_indices: list[int]
    selected_timestamps: list[float]


class MethodComparisonSummary(TypedDict):
    """Summary of method comparison with recommendation."""

    methods: list[MethodComparisonResult]
    recommended_method: str
    recommendation_reason: str
    weights: dict[str, float]  # Weights used for combined score


@dataclass
class FrameAnalysisConfig:
    """Configuration for frame analysis."""

    n_representative: int = 8  # Number of frames to select (or base for dynamic)
    selection_method: Literal["kmeans", "maximin", "coverage"] = "kmeans"
    reduction_methods: list[Literal["pca", "tsne", "umap"]] = field(
        default_factory=lambda: ["pca", "tsne", "umap"]
    )
    # t-SNE parameters
    tsne_perplexity: int = 30
    tsne_learning_rate: float = 200.0
    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    # Coverage parameters
    coverage_epsilon: float = 0.2
    # Dynamic frame count parameters
    dynamic_frame_count: bool = False  # Enable duration-based frame count
    frames_per_minute: float = 1.0  # Frames to select per minute of video
    min_frames: int = 4  # Minimum frames regardless of duration
    max_frames: int = 50  # Maximum frames to prevent excessive processing
    # Comparison mode
    compare_methods: bool = True  # Run all methods and recommend best

    def calculate_frame_count(self, duration_seconds: float) -> int:
        """
        Calculate the number of representative frames based on video duration.

        Args:
            duration_seconds: Video duration in seconds

        Returns:
            Number of frames to select
        """
        if not self.dynamic_frame_count:
            return self.n_representative

        duration_minutes = duration_seconds / 60.0
        calculated = int(duration_minutes * self.frames_per_minute)

        # Apply bounds
        return max(self.min_frames, min(calculated, self.max_frames))


class FrameAnalysisResult(TypedDict):
    """Complete frame analysis result for a scene."""

    scene_id: int
    frame_count: int
    embedding_dimensions: int
    embedding_model: str
    embeddings: list[FrameEmbedding]
    similarity_matrix: FrameSimilarityMatrix
    reductions: list[DimensionalityReductionResult]
    representative: RepresentativeFrameResult


class FrameAnalyzer:
    """Analyze frame embeddings within a scene for similarity and diversity."""

    def __init__(
        self,
        embedder: BaseImageEmbeddingProvider,
        config: FrameAnalysisConfig | None = None,
    ) -> None:
        """
        Initialize the frame analyzer.

        Args:
            embedder: Image embedding provider (CLIP, OpenCLIP, SigLIP)
            config: Analysis configuration
        """
        self.embedder = embedder
        self.config = config or FrameAnalysisConfig()

    def embed_all_frames(
        self,
        frame_paths: list[str],
        frame_metadata: list[ExtractedFrame],
    ) -> list[FrameEmbedding]:
        """
        Generate embeddings for each frame individually.

        Args:
            frame_paths: Paths to frame image files
            frame_metadata: Metadata for each frame (index, timestamp)

        Returns:
            List of FrameEmbedding objects
        """
        # Cast to satisfy mypy - list[str] is a valid list[ImageInput]
        results = self.embedder.embed_images(cast("list[Any]", frame_paths))
        return [
            FrameEmbedding(
                index=meta.index,
                timestamp=meta.timestamp,
                embedding=result["embedding"],
            )
            for meta, result in zip(frame_metadata, results)
        ]

    def compute_similarity_matrix(
        self,
        embeddings: list[FrameEmbedding],
    ) -> FrameSimilarityMatrix:
        """
        Compute pairwise cosine similarity matrix between frames.

        Args:
            embeddings: List of frame embeddings

        Returns:
            FrameSimilarityMatrix with NxN similarity scores
        """
        _n = len(embeddings)  # Used for NxN matrix documentation
        vectors: NDArray[np.float32] = np.array(
            [e["embedding"] for e in embeddings], dtype=np.float32
        )

        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = vectors / norms

        # Compute similarity matrix (dot product of normalized vectors)
        similarity: NDArray[np.float32] = np.dot(normalized, normalized.T)

        return FrameSimilarityMatrix(
            frame_indices=[e["index"] for e in embeddings],
            timestamps=[e["timestamp"] for e in embeddings],
            matrix=similarity.tolist(),
        )

    def reduce_dimensions(
        self,
        embeddings: list[FrameEmbedding],
        method: Literal["pca", "tsne", "umap"],
    ) -> DimensionalityReductionResult:
        """
        Apply dimensionality reduction for visualization.

        Args:
            embeddings: List of frame embeddings
            method: Reduction method (pca, tsne, umap)

        Returns:
            DimensionalityReductionResult with 2D coordinates
        """
        vectors: NDArray[np.float32] = np.array(
            [e["embedding"] for e in embeddings], dtype=np.float32
        )

        explained_variance: float | None = None
        coordinates: NDArray[np.float32]

        if method == "pca":
            from sklearn.decomposition import PCA

            n_components = min(2, vectors.shape[0], vectors.shape[1])
            pca = PCA(n_components=n_components)
            coordinates = pca.fit_transform(vectors)
            explained_variance = float(np.sum(pca.explained_variance_ratio_))

            # Pad to 2D if only 1 component
            if coordinates.shape[1] < 2:
                coordinates = np.column_stack([coordinates, np.zeros(coordinates.shape[0])])

        elif method == "tsne":
            from sklearn.manifold import TSNE

            # t-SNE requires perplexity < n_samples
            perplexity = min(self.config.tsne_perplexity, max(1, len(embeddings) - 1))
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate=self.config.tsne_learning_rate,
                random_state=42,
                max_iter=1000,  # renamed from n_iter in scikit-learn 1.5+
            )
            coordinates = tsne.fit_transform(vectors)

        elif method == "umap":
            import umap

            n_samples = len(embeddings)
            # UMAP n_neighbors must be < n_samples
            n_neighbors = min(self.config.umap_n_neighbors, max(2, n_samples - 1))

            # Use random init for small datasets (spectral fails with few samples)
            # Spectral layout requires n_samples > n_neighbors + 1
            init_method = "random" if n_samples < 10 else "spectral"

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=self.config.umap_min_dist,
                init=init_method,
                random_state=42,
            )
            coordinates = reducer.fit_transform(vectors)

        else:
            raise ValueError(f"Unknown reduction method: {method}")

        return DimensionalityReductionResult(
            method=method,
            coordinates=coordinates.tolist(),
            frame_indices=[e["index"] for e in embeddings],
            timestamps=[e["timestamp"] for e in embeddings],
            explained_variance=explained_variance,
        )

    def select_representative_frames(
        self,
        embeddings: list[FrameEmbedding],
        n_select: int | None = None,
        method: Literal["kmeans", "maximin", "coverage"] | None = None,
    ) -> RepresentativeFrameResult:
        """
        Select diverse, representative frames from the embedding space.

        Args:
            embeddings: List of frame embeddings
            n_select: Number of frames to select (default from config)
            method: Selection algorithm (default from config)

        Returns:
            RepresentativeFrameResult with selected frame indices
        """
        n_select = n_select or self.config.n_representative
        method = method or self.config.selection_method
        n_frames = len(embeddings)

        # Don't select more frames than available
        n_select = min(n_select, n_frames)

        vectors: NDArray[np.float32] = np.array(
            [e["embedding"] for e in embeddings], dtype=np.float32
        )

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = vectors / norms

        cluster_assignments: list[int] | None = None

        if method == "kmeans":
            selected, cluster_assignments = self._select_kmeans(normalized, n_select)
        elif method == "maximin":
            selected = self._select_maximin(normalized, n_select)
        elif method == "coverage":
            selected = self._select_coverage(normalized, n_select)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        # Calculate diversity score (average pairwise distance of selected frames)
        selected_vectors = normalized[selected]
        diversity_score = self._calculate_diversity(selected_vectors)

        # Map back to 1-based frame indices
        selected_indices = [embeddings[i]["index"] for i in selected]
        selected_timestamps = [embeddings[i]["timestamp"] for i in selected]

        return RepresentativeFrameResult(
            selected_indices=selected_indices,
            selected_timestamps=selected_timestamps,
            cluster_assignments=cluster_assignments,
            selection_method=method,
            diversity_score=diversity_score,
        )

    def _select_kmeans(
        self,
        vectors: NDArray[np.float32],
        n_select: int,
    ) -> tuple[list[int], list[int]]:
        """
        Select frames closest to k-means cluster centroids.

        Args:
            vectors: Normalized embedding vectors
            n_select: Number of frames to select

        Returns:
            Tuple of (selected indices, cluster assignments for all frames)
        """
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)

        # For each cluster, find frame closest to centroid
        selected: list[int] = []
        for i in range(n_select):
            cluster_mask = cluster_labels == i
            if not np.any(cluster_mask):
                continue

            cluster_indices = np.where(cluster_mask)[0]
            cluster_vectors = vectors[cluster_mask]
            centroid = kmeans.cluster_centers_[i]

            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            closest_in_cluster = np.argmin(distances)
            selected.append(int(cluster_indices[closest_in_cluster]))

        # Sort by frame index for temporal order
        selected.sort()

        return selected, cluster_labels.tolist()

    def _select_maximin(
        self,
        vectors: NDArray[np.float32],
        n_select: int,
    ) -> list[int]:
        """
        Greedily select frames maximizing minimum pairwise distance.

        This algorithm ensures maximum coverage of the embedding space.

        Args:
            vectors: Normalized embedding vectors
            n_select: Number of frames to select

        Returns:
            List of selected frame indices (0-based)
        """
        n = len(vectors)
        selected: list[int] = [0]  # Start with first frame

        for _ in range(n_select - 1):
            # Find frame with maximum distance to nearest selected frame
            min_distances = np.full(n, np.inf)

            for s in selected:
                distances = np.linalg.norm(vectors - vectors[s], axis=1)
                min_distances = np.minimum(min_distances, distances)

            # Exclude already selected
            for s in selected:
                min_distances[s] = -np.inf

            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)

        selected.sort()
        return selected

    def _select_coverage(
        self,
        vectors: NDArray[np.float32],
        n_select: int,
    ) -> list[int]:
        """
        Select frames that cover the embedding space with epsilon-balls.

        Greedy set cover approximation with temporal diversity tie-breaking.
        When coverage is equal, prefers frames that are temporally spread.

        Args:
            vectors: Normalized embedding vectors
            n_select: Number of frames to select

        Returns:
            List of selected frame indices (0-based)
        """
        n = len(vectors)
        epsilon = self.config.coverage_epsilon
        uncovered = set(range(n))
        selected: list[int] = []

        while len(selected) < n_select and uncovered:
            # Find frame that covers the most uncovered points
            best_idx = -1
            best_coverage = -1
            best_temporal_distance = -1.0

            for i in range(n):
                if i in selected:
                    continue

                # Count how many uncovered points this frame covers
                distances = np.linalg.norm(vectors - vectors[i], axis=1)
                covered_by_i = {j for j in uncovered if distances[j] <= epsilon}
                coverage = len(covered_by_i)

                # Calculate temporal distance from already selected frames
                # Use frame index as temporal proxy (higher = more spread)
                if selected:
                    temporal_distance: float = float(min(abs(i - s) for s in selected))
                else:
                    # First frame: prefer middle of video for better coverage
                    temporal_distance = n / 2 - abs(i - n / 2)

                # Prefer higher coverage, then higher temporal spread
                if coverage > best_coverage or (
                    coverage == best_coverage and temporal_distance > best_temporal_distance
                ):
                    best_coverage = coverage
                    best_idx = i
                    best_temporal_distance = temporal_distance

            if best_idx == -1:
                # No more coverage possible, select frame with max temporal distance
                # This ensures temporal spread when embedding space is fully covered
                if uncovered:
                    remaining = list(uncovered)
                    if selected:
                        # Pick the uncovered frame farthest from any selected frame
                        max_dist = -1
                        for r in remaining:
                            dist = min(abs(r - s) for s in selected)
                            if dist > max_dist:
                                max_dist = dist
                                best_idx = r
                    else:
                        # No frames selected yet, pick middle frame
                        best_idx = remaining[len(remaining) // 2]

            if best_idx == -1:
                break

            # Mark points as covered
            distances = np.linalg.norm(vectors - vectors[best_idx], axis=1)
            newly_covered = {j for j in uncovered if distances[j] <= epsilon}
            uncovered -= newly_covered
            selected.append(best_idx)

        selected.sort()
        return selected

    def _calculate_diversity(self, vectors: NDArray[np.float32]) -> float:
        """
        Calculate diversity score as average pairwise distance.

        Args:
            vectors: Normalized embedding vectors for selected frames

        Returns:
            Average pairwise Euclidean distance (0-2 for normalized vectors)
        """
        n = len(vectors)
        if n < 2:
            return 0.0

        total_distance = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_distance += float(np.linalg.norm(vectors[i] - vectors[j]))
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _calculate_temporal_spread(
        self,
        selected_indices: list[int],
        total_frames: int,
    ) -> float:
        """
        Calculate how evenly selected frames cover the video duration.

        Uses coefficient of variation of gaps between selected frames.
        Perfect spread (evenly spaced) = 1.0, all clustered = 0.0

        Args:
            selected_indices: 0-based indices of selected frames
            total_frames: Total number of frames in video

        Returns:
            Temporal spread score (0-1, higher is better)
        """
        if len(selected_indices) < 2:
            return 1.0 if len(selected_indices) == 1 else 0.0

        sorted_indices = sorted(selected_indices)

        # Calculate gaps between consecutive selected frames
        # Include gaps from start and to end
        gaps = [sorted_indices[0]]  # Gap from start
        for i in range(1, len(sorted_indices)):
            gaps.append(sorted_indices[i] - sorted_indices[i - 1])
        gaps.append(total_frames - 1 - sorted_indices[-1])  # Gap to end

        # Ideal gap for perfect spread
        ideal_gap = total_frames / (len(selected_indices) + 1)

        # Calculate variance from ideal
        if ideal_gap == 0:
            return 1.0

        deviations = [(g - ideal_gap) ** 2 for g in gaps]
        variance = sum(deviations) / len(deviations)
        std_dev = variance**0.5

        # Normalize: CV of 0 = perfect, higher = worse
        # Convert to 0-1 score where 1 is perfect
        cv = std_dev / ideal_gap if ideal_gap > 0 else 0
        spread_score = max(0.0, 1.0 - cv)

        return spread_score

    def _calculate_coverage_score(
        self,
        selected_indices: list[int],
        vectors: NDArray[np.float32],
        epsilon: float,
    ) -> float:
        """
        Calculate what fraction of all frames are within epsilon of a selected frame.

        Args:
            selected_indices: 0-based indices of selected frames
            vectors: Normalized embedding vectors for all frames
            epsilon: Coverage radius in embedding space

        Returns:
            Coverage score (0-1, fraction of frames covered)
        """
        n = len(vectors)
        if n == 0 or len(selected_indices) == 0:
            return 0.0

        covered = set()
        for idx in selected_indices:
            distances = np.linalg.norm(vectors - vectors[idx], axis=1)
            for i, d in enumerate(distances):
                if d <= epsilon:
                    covered.add(i)

        return len(covered) / n

    def compare_all_methods(
        self,
        embeddings: list[FrameEmbedding],
        n_select: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> MethodComparisonSummary:
        """
        Run all selection methods and compare their results.

        Args:
            embeddings: List of frame embeddings
            n_select: Number of frames to select (default from config)
            weights: Optional weights for combined score. Keys: diversity, temporal, coverage

        Returns:
            MethodComparisonSummary with results and recommendation
        """
        n_select = n_select or self.config.n_representative
        n_frames = len(embeddings)
        n_select = min(n_select, n_frames)

        # Default weights: balance all three metrics
        if weights is None:
            weights = {
                "diversity": 0.4,  # Visual distinctiveness
                "temporal": 0.35,  # Duration coverage
                "coverage": 0.25,  # Embedding space coverage
            }

        vectors: NDArray[np.float32] = np.array(
            [e["embedding"] for e in embeddings], dtype=np.float32
        )
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = vectors / norms

        methods: list[Literal["kmeans", "maximin", "coverage"]] = ["kmeans", "maximin", "coverage"]
        results: list[MethodComparisonResult] = []

        for method in methods:
            # Run selection
            rep_result = self.select_representative_frames(
                embeddings, n_select=n_select, method=method
            )

            # Get 0-based indices for metric calculation
            indices_0based = [
                i for i, e in enumerate(embeddings) if e["index"] in rep_result["selected_indices"]
            ]

            # Calculate metrics
            selected_vectors = normalized[indices_0based]
            diversity = self._calculate_diversity(selected_vectors)
            temporal = self._calculate_temporal_spread(indices_0based, n_frames)
            coverage = self._calculate_coverage_score(
                indices_0based, normalized, self.config.coverage_epsilon
            )

            # Normalize diversity to 0-1 (max possible is 2.0 for normalized vectors)
            diversity_normalized = diversity / 2.0

            # Combined score
            combined = (
                weights["diversity"] * diversity_normalized
                + weights["temporal"] * temporal
                + weights["coverage"] * coverage
            )

            results.append(
                MethodComparisonResult(
                    method=method,
                    diversity_score=diversity,
                    temporal_spread=temporal,
                    coverage_score=coverage,
                    combined_score=combined,
                    selected_indices=rep_result["selected_indices"],
                    selected_timestamps=rep_result["selected_timestamps"],
                )
            )

        # Find best method
        best = max(results, key=lambda r: r["combined_score"])
        best_method = best["method"]

        # Generate recommendation reason
        reasons = []
        if best["diversity_score"] == max(r["diversity_score"] for r in results):
            reasons.append("highest visual diversity")
        if best["temporal_spread"] == max(r["temporal_spread"] for r in results):
            reasons.append("best temporal coverage")
        if best["coverage_score"] == max(r["coverage_score"] for r in results):
            reasons.append("best embedding space coverage")

        if not reasons:
            reasons.append("best overall balance of metrics")

        recommendation_reason = f"{best_method.upper()} selected for {', '.join(reasons)}"

        return MethodComparisonSummary(
            methods=results,
            recommended_method=best_method,
            recommendation_reason=recommendation_reason,
            weights=weights,
        )


class FrameAnalysisTask:
    """Stash task for analyzing frame embeddings within a scene."""

    def __init__(
        self,
        stash: "StashClient",
        image_embedding_config: EmbeddingConfig,
        analysis_config: FrameAnalysisConfig | None = None,
        frame_config: FrameExtractionConfig | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Initialize the frame analysis task.

        Args:
            stash: StashClient instance
            image_embedding_config: Config for image embedding model
            analysis_config: Optional analysis configuration
            frame_config: Optional frame extraction configuration
            log_callback: Optional logging callback
            progress_callback: Optional progress callback
        """
        self.stash = stash
        self.image_embedding_config = image_embedding_config
        self.analysis_config = analysis_config or FrameAnalysisConfig()
        self.frame_config = frame_config or FrameExtractionConfig()
        self.log = log_callback or (lambda msg, level: None)
        self.progress = progress_callback or (lambda current, total: None)

        # Get assets directory
        plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.assets_dir = os.path.join(plugin_dir, "assets")
        self.cache_dir = os.path.join(self.assets_dir, "embedded_frames")

        # Initialize embedding provider
        embedder = get_embedding_provider(image_embedding_config)
        if not embedder.supports_images:
            raise ValueError(
                f"Embedding provider {image_embedding_config.provider} does not support images"
            )
        self.embedder = cast("BaseImageEmbeddingProvider", embedder)

        # Initialize frame extractor
        self.frame_extractor = FrameExtractor(
            config=self.frame_config,
            cache_dir=self.cache_dir,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        # Initialize analyzer
        self.analyzer = FrameAnalyzer(
            embedder=self.embedder,
            config=self.analysis_config,
        )

    def run(self, scene_id: int) -> FrameAnalysisResult | None:
        """
        Run frame analysis for a scene.

        Args:
            scene_id: Scene ID to analyze

        Returns:
            FrameAnalysisResult or None if failed
        """
        self.log(f"Starting frame analysis for scene {scene_id}", "info")
        self.progress(0, 100)

        # Create output directory and write initial status for UI polling
        output_dir = self._get_output_dir(scene_id)
        self._write_status(output_dir, scene_id, "running")

        # Get scene info
        scene = self._get_scene_info(scene_id)
        if not scene:
            self.log(f"Scene {scene_id} not found", "error")
            self._write_status(output_dir, scene_id, "error", "Scene not found")
            return None

        video_path = scene.get("path")
        duration = scene.get("duration", 0)

        if not video_path or not os.path.exists(video_path):
            self.log(f"Video file not found: {video_path}", "error")
            self._write_status(output_dir, scene_id, "error", "Video file not found")
            return None

        # Step 1: Extract frames
        self.log("Extracting frames...", "info")
        self.progress(10, 100)
        frames = self.frame_extractor.get_or_extract_frames(str(scene_id), video_path, duration)

        if not frames:
            self.log("No frames extracted", "error")
            self._write_status(output_dir, scene_id, "error", "No frames extracted")
            return None

        self.log(f"Extracted {len(frames)} frames", "info")

        # Get frame paths
        frame_paths = self.frame_extractor.get_frame_paths(str(scene_id))
        if not frame_paths:
            self.log("Failed to get frame paths", "error")
            self._write_status(output_dir, scene_id, "error", "Failed to get frame paths")
            return None

        # Step 2: Generate embeddings for all frames
        self.log("Generating frame embeddings...", "info")
        self.progress(20, 100)
        embeddings = self.analyzer.embed_all_frames(frame_paths, frames)
        self.log(
            f"Generated {len(embeddings)} embeddings ({self.embedder.dimensions}D)",
            "info",
        )

        # Step 3: Compute similarity matrix
        self.log("Computing similarity matrix...", "info")
        self.progress(40, 100)
        similarity_matrix = self.analyzer.compute_similarity_matrix(embeddings)

        # Step 4: Apply dimensionality reduction
        self.log("Applying dimensionality reduction...", "info")
        self.progress(50, 100)
        reductions: list[DimensionalityReductionResult] = []
        for method in self.analysis_config.reduction_methods:
            self.log(f"  Running {method.upper()}...", "debug")
            reduction_result = self.analyzer.reduce_dimensions(embeddings, method)
            reductions.append(reduction_result)
            if reduction_result["explained_variance"]:
                self.log(
                    f"  {method.upper()} explained variance: {reduction_result['explained_variance']:.2%}",
                    "debug",
                )

        # Step 5: Select representative frames
        self.log("Selecting representative frames...", "info")
        self.progress(70, 100)

        # Calculate frame count (dynamic based on duration, or static from config)
        n_frames = self.analysis_config.calculate_frame_count(duration)
        # Don't select more frames than available
        n_frames = min(n_frames, len(embeddings))

        if self.analysis_config.dynamic_frame_count:
            self.log(
                f"Dynamic frame count: {n_frames} frames "
                f"({self.analysis_config.frames_per_minute}/min, "
                f"duration: {duration / 60:.1f}min)",
                "info",
            )

        # Run comparison of all methods if enabled
        method_comparison: MethodComparisonSummary | None = None
        if self.analysis_config.compare_methods:
            self.log("Comparing all selection methods...", "info")
            method_comparison = self.analyzer.compare_all_methods(embeddings, n_select=n_frames)

            # Log comparison results
            self.log("=" * 50, "info")
            self.log("METHOD COMPARISON RESULTS", "info")
            self.log("=" * 50, "info")
            self.log(
                f"{'Method':<10} {'Diversity':>10} {'Temporal':>10} "
                f"{'Coverage':>10} {'Combined':>10}",
                "info",
            )
            self.log("-" * 50, "info")
            for m in method_comparison["methods"]:
                self.log(
                    f"{m['method']:<10} {m['diversity_score']:>10.3f} "
                    f"{m['temporal_spread']:>10.3f} {m['coverage_score']:>10.3f} "
                    f"{m['combined_score']:>10.3f}",
                    "info",
                )
            self.log("-" * 50, "info")
            self.log(f"Recommendation: {method_comparison['recommendation_reason']}", "info")
            self.log("=" * 50, "info")

            # Use the recommended method's results
            best_method = method_comparison["recommended_method"]
            best_result = next(
                m for m in method_comparison["methods"] if m["method"] == best_method
            )
            representative = RepresentativeFrameResult(
                selected_indices=best_result["selected_indices"],
                selected_timestamps=best_result["selected_timestamps"],
                cluster_assignments=None,  # Not available from comparison
                selection_method=best_method,
                diversity_score=best_result["diversity_score"],
            )
        else:
            representative = self.analyzer.select_representative_frames(
                embeddings, n_select=n_frames
            )

        self.log(
            f"Selected {len(representative['selected_indices'])} frames "
            f"(method: {representative['selection_method']}, "
            f"diversity: {representative['diversity_score']:.3f})",
            "info",
        )

        # Step 6: Generate visualizations
        self.log("Generating visualizations...", "info")
        self.progress(80, 100)
        _plot_paths = self._generate_visualizations(
            scene_id,
            embeddings,
            similarity_matrix,
            reductions,
            representative,
            output_dir,
        )

        # Step 7: Save results
        self.log("Saving results...", "info")
        self.progress(90, 100)

        result = FrameAnalysisResult(
            scene_id=scene_id,
            frame_count=len(frames),
            embedding_dimensions=self.embedder.dimensions,
            embedding_model=self.embedder.model,
            embeddings=embeddings,
            similarity_matrix=similarity_matrix,
            reductions=reductions,
            representative=representative,
        )

        # Save JSON results (without full embeddings for readability)
        self._save_results(result, output_dir, method_comparison)

        self.progress(100, 100)
        self.log(f"Frame analysis complete. Output: {output_dir}", "info")
        self.log(f"Selected frames: {representative['selected_indices']}", "info")

        return result

    def _get_scene_info(self, scene_id: int) -> dict[str, Any] | None:
        """Get scene info from Stash."""
        from ..tools.database import get_readonly_connection, get_stash_db_path

        try:
            db_path = get_stash_db_path()
            conn = get_readonly_connection(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT fo.path || '/' || f.basename as full_path, vf.duration
                FROM scenes s
                JOIN scenes_files sf ON s.id = sf.scene_id
                JOIN files f ON sf.file_id = f.id
                JOIN folders fo ON f.parent_folder_id = fo.id
                LEFT JOIN video_files vf ON f.id = vf.file_id
                WHERE s.id = ?
                LIMIT 1
                """,
                (scene_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return {"path": row[0], "duration": row[1] or 0}
            return None
        except Exception as e:
            self.log(f"Database error: {e}", "error")
            return None

    def _get_output_dir(self, scene_id: int) -> str:
        """Get or create output directory for analysis results."""
        output_dir = os.path.join(self.assets_dir, f"frame_analysis_{scene_id}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _generate_visualizations(
        self,
        scene_id: int,
        embeddings: list[FrameEmbedding],
        similarity_matrix: FrameSimilarityMatrix,
        reductions: list[DimensionalityReductionResult],
        representative: RepresentativeFrameResult,
        output_dir: str,
    ) -> list[str]:
        """Generate and save visualization plots."""
        from .frame_visualization import FrameVisualizer

        visualizer = FrameVisualizer(output_dir)
        plot_paths: list[str] = []

        # Individual reduction plots
        for reduction in reductions:
            path = visualizer.plot_reduction(
                reduction,
                representative["selected_indices"],
                scene_id,
            )
            plot_paths.append(path)
            self.log(f"  Saved {reduction['method']}_plot.png", "debug")

        # Similarity heatmap
        path = visualizer.plot_similarity_matrix(similarity_matrix, scene_id)
        plot_paths.append(path)
        self.log("  Saved similarity_heatmap.png", "debug")

        # Comparison plot
        path = visualizer.plot_comparison(
            reductions,
            representative["selected_indices"],
            scene_id,
        )
        plot_paths.append(path)
        self.log("  Saved comparison_plot.png", "debug")

        return plot_paths

    def _write_status(
        self,
        output_dir: str,
        scene_id: int,
        status: str,
        error: str | None = None,
    ) -> None:
        """Write status file for UI polling."""
        from datetime import datetime

        status_data: dict[str, Any] = {
            "status": status,
            "scene_id": scene_id,
            "updated_at": datetime.now().isoformat(),
        }
        if error:
            status_data["error"] = error

        status_path = os.path.join(output_dir, "analysis_status.json")
        with open(status_path, "w") as f:
            json.dump(status_data, f, indent=2)

    def _save_results(
        self,
        result: FrameAnalysisResult,
        output_dir: str,
        method_comparison: MethodComparisonSummary | None = None,
    ) -> None:
        """Save analysis results to JSON with UI-friendly format."""
        scene_id = result["scene_id"]

        # Create UI-friendly summary
        summary: dict[str, Any] = {
            "status": "complete",
            "scene_id": scene_id,
            "frame_count": result["frame_count"],
            "embedding_dimensions": result["embedding_dimensions"],
            "embedding_model": result["embedding_model"],
            "similarity_stats": {
                "min": float(np.min(result["similarity_matrix"]["matrix"])),
                "max": float(np.max(result["similarity_matrix"]["matrix"])),
                "mean": float(np.mean(result["similarity_matrix"]["matrix"])),
                "median": float(np.median(result["similarity_matrix"]["matrix"])),
            },
            "representative": result["representative"],
            "reduction_methods": [r["method"] for r in result["reductions"]],
            # UI-specific fields
            "plots": {
                "similarity_heatmap": "similarity_heatmap.png",
                "pca": "pca_plot.png",
                "tsne": "tsne_plot.png",
                "umap": "umap_plot.png",
                "comparison": "comparison_plot.png",
            },
            "frame_cache_dir": f"embedded_frames/scene_{scene_id}",
            "frame_timestamps": result["similarity_matrix"]["timestamps"],
            "frame_indices": result["similarity_matrix"]["frame_indices"],
        }

        # Add PCA explained variance if available
        for r in result["reductions"]:
            if r["method"] == "pca" and r["explained_variance"]:
                summary["pca_explained_variance"] = r["explained_variance"]

        # Add method comparison results if available
        if method_comparison:
            summary["method_comparison"] = {
                "methods": [
                    {
                        "method": m["method"],
                        "diversity_score": round(m["diversity_score"], 4),
                        "temporal_spread": round(m["temporal_spread"], 4),
                        "coverage_score": round(m["coverage_score"], 4),
                        "combined_score": round(m["combined_score"], 4),
                    }
                    for m in method_comparison["methods"]
                ],
                "recommended_method": method_comparison["recommended_method"],
                "recommendation_reason": method_comparison["recommendation_reason"],
                "weights": method_comparison["weights"],
            }

        # Save summary (this is what the UI polls)
        summary_path = os.path.join(output_dir, "analysis_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Also update the status file to complete
        self._write_status(output_dir, scene_id, "complete")

        # Save full results (including embeddings) separately for debugging
        full_path = os.path.join(output_dir, "analysis_results.json")
        with open(full_path, "w") as f:
            json.dump(dict(result), f, indent=2)
