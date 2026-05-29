"""Visualization utilities for frame embedding analysis."""

import os

import numpy as np
from numpy.typing import NDArray

from .frame_analysis import DimensionalityReductionResult, FrameSimilarityMatrix


class FrameVisualizer:
    """Generate visualizations for frame analysis results."""

    def __init__(self, output_dir: str) -> None:
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save plot files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Use non-interactive backend for headless operation
        import matplotlib

        matplotlib.use("Agg")

    def plot_reduction(
        self,
        result: DimensionalityReductionResult,
        selected_indices: list[int] | None = None,
        scene_id: int = 0,
    ) -> str:
        """
        Generate scatter plot of reduced embeddings.

        Points are colored by timestamp (early=blue, late=red).
        Selected frames are highlighted with black circles.

        Args:
            result: Dimensionality reduction result
            selected_indices: Indices of selected representative frames (1-based)
            scene_id: Scene ID for title

        Returns:
            Path to saved plot file
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        coords: NDArray[np.float32] = np.array(result["coordinates"])
        timestamps = result["timestamps"]
        frame_indices = result["frame_indices"]

        # Normalize timestamps for color mapping
        t_min, t_max = min(timestamps), max(timestamps)
        if t_max > t_min:
            t_norm = [(t - t_min) / (t_max - t_min) for t in timestamps]
        else:
            t_norm = [0.5] * len(timestamps)

        # Plot all points
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=t_norm,
            cmap="coolwarm",
            s=100,
            alpha=0.7,
            edgecolors="white",
            linewidth=1,
        )

        # Add frame index labels
        for i, (x, y) in enumerate(coords):
            ax.annotate(
                str(frame_indices[i]),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        # Highlight selected frames
        if selected_indices:
            selected_mask = [idx in selected_indices for idx in frame_indices]
            selected_coords = coords[selected_mask]
            ax.scatter(
                selected_coords[:, 0],
                selected_coords[:, 1],
                s=200,
                facecolors="none",
                edgecolors="black",
                linewidth=2,
                label="Selected",
            )
            ax.legend(loc="upper right")

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Time (normalized)")

        # Labels and title
        method = result["method"].upper()
        title = f"Frame Embeddings - {method}"
        if result["explained_variance"]:
            title += f" (Explained variance: {result['explained_variance']:.1%})"
        title += f"\nScene {scene_id}"

        ax.set_title(title)
        ax.set_xlabel(f"{method} Component 1")
        ax.set_ylabel(f"{method} Component 2")

        # Save
        filename = f"{result['method']}_plot.png"
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return filepath

    def plot_similarity_matrix(
        self,
        matrix: FrameSimilarityMatrix,
        scene_id: int,
    ) -> str:
        """
        Generate heatmap of frame-to-frame similarity.

        Args:
            matrix: Pairwise similarity matrix
            scene_id: Scene ID for title

        Returns:
            Path to saved plot file
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))

        sim_matrix: NDArray[np.float32] = np.array(matrix["matrix"])
        frame_indices = matrix["frame_indices"]
        n = len(frame_indices)

        # Create heatmap
        im = ax.imshow(sim_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Cosine Similarity")

        # Set tick labels
        tick_positions = list(range(n))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)

        # Use frame indices as labels (or timestamps for clarity)
        if n <= 20:
            labels = [str(idx) for idx in frame_indices]
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
        else:
            # Show every Nth label for readability
            step = max(1, n // 10)
            ax.set_xticks(tick_positions[::step])
            ax.set_yticks(tick_positions[::step])
            ax.set_xticklabels(
                [str(frame_indices[i]) for i in tick_positions[::step]],
                rotation=45,
                ha="right",
            )
            ax.set_yticklabels([str(frame_indices[i]) for i in tick_positions[::step]])

        # Labels and title
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Frame Index")
        ax.set_title(f"Frame Similarity Matrix\nScene {scene_id}")

        # Add statistics annotation
        stats_text = (
            f"Mean: {np.mean(sim_matrix):.3f}\n"
            f"Median: {np.median(sim_matrix):.3f}\n"
            f"Min: {np.min(sim_matrix):.3f}\n"
            f"Max: {np.max(sim_matrix):.3f}"
        )
        ax.text(
            1.02,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="center",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        # Save
        filepath = os.path.join(self.output_dir, "similarity_heatmap.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return filepath

    def plot_comparison(
        self,
        reductions: list[DimensionalityReductionResult],
        selected_indices: list[int],
        scene_id: int,
    ) -> str:
        """
        Generate side-by-side comparison of all reduction methods.

        Args:
            reductions: List of reduction results
            selected_indices: Indices of selected frames (1-based)
            scene_id: Scene ID for title

        Returns:
            Path to saved plot file
        """
        import matplotlib.pyplot as plt

        n_methods = len(reductions)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))

        if n_methods == 1:
            axes = [axes]

        for ax, result in zip(axes, reductions):
            coords: NDArray[np.float32] = np.array(result["coordinates"])
            timestamps = result["timestamps"]
            frame_indices = result["frame_indices"]

            # Normalize timestamps
            t_min, t_max = min(timestamps), max(timestamps)
            if t_max > t_min:
                t_norm = [(t - t_min) / (t_max - t_min) for t in timestamps]
            else:
                t_norm = [0.5] * len(timestamps)

            # Plot all points
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=t_norm,
                cmap="coolwarm",
                s=60,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

            # Highlight selected frames
            selected_mask = [idx in selected_indices for idx in frame_indices]
            selected_coords = coords[selected_mask]
            ax.scatter(
                selected_coords[:, 0],
                selected_coords[:, 1],
                s=120,
                facecolors="none",
                edgecolors="black",
                linewidth=2,
            )

            # Title
            method = result["method"].upper()
            title = method
            if result["explained_variance"]:
                title += f"\n(Var: {result['explained_variance']:.1%})"
            ax.set_title(title)
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")

        fig.suptitle(
            f"Frame Embedding Comparison - Scene {scene_id}\n"
            f"(Black circles = {len(selected_indices)} selected representative frames)",
            fontsize=12,
        )
        plt.tight_layout()

        # Save
        filepath = os.path.join(self.output_dir, "comparison_plot.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return filepath

    def plot_frame_timeline(
        self,
        selected_indices: list[int],
        selected_timestamps: list[float],
        total_frames: int,
        duration: float,
        scene_id: int,
    ) -> str:
        """
        Generate timeline showing where selected frames fall in the video.

        Args:
            selected_indices: Indices of selected frames (1-based)
            selected_timestamps: Timestamps of selected frames
            total_frames: Total number of frames analyzed
            duration: Video duration in seconds
            scene_id: Scene ID for title

        Returns:
            Path to saved plot file
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3))

        # Draw timeline
        ax.axhline(y=0, color="gray", linewidth=2, alpha=0.5)

        # Mark all frame positions lightly
        all_positions = np.linspace(0, duration, total_frames)
        ax.scatter(
            all_positions,
            np.zeros_like(all_positions),
            s=20,
            color="lightgray",
            alpha=0.5,
            label="All frames",
        )

        # Mark selected frames prominently
        ax.scatter(
            selected_timestamps,
            np.zeros_like(selected_timestamps),
            s=100,
            color="red",
            marker="^",
            label="Selected",
            zorder=5,
        )

        # Add frame index labels
        for idx, ts in zip(selected_indices, selected_timestamps):
            ax.annotate(
                str(idx),
                (ts, 0),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        # Format x-axis as time
        ax.set_xlim(-duration * 0.02, duration * 1.02)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Time (seconds)")
        ax.set_yticks([])
        ax.set_title(
            f"Selected Frame Timeline - Scene {scene_id}\n"
            f"({len(selected_indices)} of {total_frames} frames selected)"
        )
        ax.legend(loc="upper right")

        # Save
        filepath = os.path.join(self.output_dir, "frame_timeline.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return filepath
