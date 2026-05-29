"""Smart frame selection using multi-scale novelty detection.

This module provides intelligent frame selection for VLM analysis by:
1. Selecting temporally uniform baseline frames for coverage
2. Boosting with high-novelty frames to capture action moments
3. Deduplicating to avoid sending near-identical frames

The multi-scale novelty approach detects changes at different time scales:
- Short-term (±2s): Quick cuts, rapid movements
- Medium-term (±15s): Scene transitions, position changes
- Long-term (±60s): Major scene changes, sustained climaxes

Based on POC validation with scenes 10020 and 4858.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FrameSelection:
    """A selected frame with metadata."""

    index: int
    timestamp: float
    path: str
    novelty_score: float
    selection_reason: str  # "temporal" | "novelty"


class SmartFrameSelector:
    """
    Intelligent frame selection using multi-scale novelty detection.

    Algorithm:
    1. Compute novelty at multiple time scales (windows)
    2. Combine with configurable weights
    3. Select temporal baseline (evenly spaced)
    4. Boost with highest-novelty frames
    5. Deduplicate near-identical frames

    Default settings based on POC validation:
    - windows: [2, 15, 60] - Short/Medium/Long scales
    - weights: [0, 1, 1] - Medium+Long (no short-term)
    - dedup_threshold: 0.90 - Fairly strict dedup
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        weights: list[float] | None = None,
        dedup_threshold: float = 0.90,
    ) -> None:
        """
        Initialize the smart frame selector.

        Args:
            windows: Time windows for novelty computation in seconds.
                Default: [2, 15, 60] for short/medium/long scales.
            weights: Weights for each window scale.
                Default: [0, 1, 1] to emphasize medium and long-term novelty.
            dedup_threshold: Cosine similarity above which frames are
                considered duplicates. Default: 0.90.
        """
        self.windows = windows or [2, 15, 60]
        self.weights = weights or [0.0, 1.0, 1.0]
        self.dedup_threshold = dedup_threshold

        if len(self.windows) != len(self.weights):
            raise ValueError(
                f"Windows and weights must have same length: "
                f"{len(self.windows)} vs {len(self.weights)}"
            )

    def compute_novelty_single_scale(
        self,
        embeddings: NDArray[np.float32],
        window_size: int,
    ) -> NDArray[np.float64]:
        """
        Compute novelty scores at a single time scale.

        Novelty = 1 - mean(similarity to neighbors within window).
        High novelty indicates the frame is visually different from
        its temporal neighbors.

        Edge frames are normalized to compensate for having fewer neighbors
        than middle frames (asymmetric windows at boundaries).

        Args:
            embeddings: Frame embeddings (N, D), assumed normalized.
            window_size: Number of frames before/after to compare.

        Returns:
            Novelty scores (N,) in range [0, 1].
        """
        n_frames = len(embeddings)
        novelty = np.zeros(n_frames, dtype=np.float64)

        # Expected neighbors for a frame in the middle of the video
        expected_neighbors = 2 * window_size

        for i in range(n_frames):
            # Define neighbor range
            start = max(0, i - window_size)
            end = min(n_frames, i + window_size + 1)
            neighbors = [j for j in range(start, end) if j != i]

            if not neighbors:
                novelty[i] = 0.0
                continue

            # Compute mean similarity to neighbors
            neighbor_embs = embeddings[neighbors]
            similarities: NDArray[np.float32] = np.dot(neighbor_embs, embeddings[i])
            raw_novelty = 1.0 - float(np.mean(similarities))

            # Normalize for edge frames with fewer neighbors than expected
            # This dampens artificially inflated novelty at video boundaries
            actual_neighbors = len(neighbors)
            normalization_factor = min(1.0, actual_neighbors / expected_neighbors)
            novelty[i] = raw_novelty * normalization_factor

        return novelty

    def compute_multiscale_novelty(
        self,
        embeddings: NDArray[np.float32],
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        """
        Compute novelty at multiple time scales and combine.

        Args:
            embeddings: Frame embeddings (N, D), assumed normalized.

        Returns:
            Tuple of:
            - composite: Weighted combination of all scales (N,)
            - scale_novelties: List of per-scale novelty arrays
        """
        n_frames = len(embeddings)
        scale_novelties: list[NDArray[np.float64]] = []

        for window in self.windows:
            novelty = self.compute_novelty_single_scale(embeddings, window)
            scale_novelties.append(novelty)

        # Normalize weights
        weights_arr = np.array(self.weights, dtype=np.float64)
        weight_sum = weights_arr.sum()
        if weight_sum > 0:
            weights_arr = weights_arr / weight_sum
        else:
            # If all weights are 0, use uniform weights
            weights_arr = np.ones_like(weights_arr) / len(weights_arr)

        # Weighted combination
        composite = np.zeros(n_frames, dtype=np.float64)
        for w, novelty in zip(weights_arr, scale_novelties):
            composite += w * novelty

        return composite, scale_novelties

    def select_frames(
        self,
        frame_paths: list[str],
        embeddings: NDArray[np.float32],
        timestamps: list[float],
        max_frames: int = 64,
        temporal_ratio: float = 0.5,
    ) -> list[FrameSelection]:
        """
        Select frames using temporal baseline + novelty boost.

        Algorithm:
        1. Compute multi-scale novelty for all frames
        2. Select temporal_ratio * max_frames uniformly spaced (baseline)
        3. Fill remaining slots with highest-novelty non-selected frames
        4. Deduplicate by skipping frames too similar to selected ones
        5. Sort by timestamp for chronological order

        Args:
            frame_paths: List of frame file paths.
            embeddings: Frame embeddings (N, D), assumed normalized.
            timestamps: Frame timestamps in seconds.
            max_frames: Target number of frames to select.
            temporal_ratio: Fraction of budget for temporal baseline (0-1).

        Returns:
            List of FrameSelection sorted by timestamp.
        """
        if len(frame_paths) != len(embeddings):
            raise ValueError(
                f"frame_paths and embeddings must have same length: "
                f"{len(frame_paths)} vs {len(embeddings)}"
            )

        if len(frame_paths) != len(timestamps):
            raise ValueError(
                f"frame_paths and timestamps must have same length: "
                f"{len(frame_paths)} vs {len(timestamps)}"
            )

        n_frames = len(embeddings)

        # If we have fewer frames than requested, return all
        if n_frames <= max_frames:
            return [
                FrameSelection(
                    index=i,
                    timestamp=timestamps[i],
                    path=frame_paths[i],
                    novelty_score=0.0,
                    selection_reason="temporal",
                )
                for i in range(n_frames)
            ]

        # Compute multi-scale novelty
        novelty_scores, _ = self.compute_multiscale_novelty(embeddings)

        selected_indices: set[int] = set()
        selections: list[FrameSelection] = []

        # Phase 1: Temporal baseline (evenly spaced frames)
        temporal_count = max(1, int(max_frames * temporal_ratio))
        temporal_indices = np.linspace(0, n_frames - 1, temporal_count, dtype=int)

        for idx in temporal_indices:
            idx = int(idx)
            selected_indices.add(idx)
            selections.append(
                FrameSelection(
                    index=idx,
                    timestamp=timestamps[idx],
                    path=frame_paths[idx],
                    novelty_score=float(novelty_scores[idx]),
                    selection_reason="temporal",
                )
            )

        # Phase 2: Stratified novelty boost
        # Divide video into temporal segments and select top novelty from each
        # This ensures even temporal distribution regardless of content patterns
        novelty_budget = max_frames - len(selections)
        if novelty_budget > 0:
            n_segments = 4  # Quartiles
            segment_size = n_frames // n_segments
            frames_per_segment = novelty_budget // n_segments
            remainder = novelty_budget % n_segments

            for seg in range(n_segments):
                # Define segment boundaries
                seg_start = seg * segment_size
                seg_end = (seg + 1) * segment_size if seg < n_segments - 1 else n_frames

                # Frames to select from this segment (distribute remainder to later segments)
                seg_budget = frames_per_segment + (1 if seg >= n_segments - remainder else 0)

                # Get frame indices in this segment, sorted by novelty descending
                seg_indices = list(range(seg_start, seg_end))
                seg_indices.sort(key=lambda i: novelty_scores[i], reverse=True)

                seg_selected = 0
                for idx in seg_indices:
                    if seg_selected >= seg_budget:
                        break

                    # Stop if we've reached max frames overall
                    if len(selections) >= max_frames:
                        break

                    # Skip if already selected
                    if idx in selected_indices:
                        continue

                    # Deduplication check
                    too_similar = False
                    for sel in selections:
                        sim = float(np.dot(embeddings[idx], embeddings[sel.index]))
                        if sim > self.dedup_threshold:
                            too_similar = True
                            break

                    if too_similar:
                        continue

                    # Add this frame
                    selected_indices.add(idx)
                    selections.append(
                        FrameSelection(
                            index=idx,
                            timestamp=timestamps[idx],
                            path=frame_paths[idx],
                            novelty_score=float(novelty_scores[idx]),
                            selection_reason="novelty",
                        )
                    )
                    seg_selected += 1

        # Sort by timestamp for chronological order
        selections.sort(key=lambda x: x.timestamp)

        return selections

    def get_selection_stats(
        self,
        selections: list[FrameSelection],
    ) -> dict[str, int | float]:
        """
        Get statistics about a frame selection.

        Args:
            selections: List of selected frames.

        Returns:
            Dict with statistics.
        """
        if not selections:
            return {
                "total": 0,
                "temporal_count": 0,
                "novelty_count": 0,
                "temporal_ratio": 0.0,
                "novelty_ratio": 0.0,
                "min_novelty": 0.0,
                "max_novelty": 0.0,
                "mean_novelty": 0.0,
            }

        temporal = [s for s in selections if s.selection_reason == "temporal"]
        novelty = [s for s in selections if s.selection_reason == "novelty"]
        novelty_scores = [s.novelty_score for s in selections]

        return {
            "total": len(selections),
            "temporal_count": len(temporal),
            "novelty_count": len(novelty),
            "temporal_ratio": len(temporal) / len(selections),
            "novelty_ratio": len(novelty) / len(selections),
            "min_novelty": min(novelty_scores),
            "max_novelty": max(novelty_scores),
            "mean_novelty": sum(novelty_scores) / len(novelty_scores),
        }
