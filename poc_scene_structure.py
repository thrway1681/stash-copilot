#!/usr/bin/env python3
"""
POC: Embedding-Based Scene Structure Detection (Interactive Plotly Version)

This proof of concept analyzes frame embeddings to detect scene structure:
- Transition points where visual similarity drops
- Segments of visually cohesive content
- Novelty spikes (frames that differ significantly from neighbors)

Usage:
    uv run python poc_scene_structure.py <scene_id> [scene_id2] [scene_id3] ...
    uv run python poc_scene_structure.py 10020 4858 10123  # Test specific scenes

Output:
    - Console: Detected segments with timestamps
    - Interactive HTML: assets/structure_analysis/scene_{id}_structure.html
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stash_ai.embeddings.storage import EmbeddingStorage


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def compute_consecutive_similarity(
    frames: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """
    Compute cosine similarity between consecutive frames.

    Args:
        frames: List of frame dicts with 'timestamp' and 'embedding'

    Returns:
        Tuple of (timestamps, similarities)
        - timestamps: Timestamp of each frame (starting from index 1)
        - similarities: Similarity to previous frame
    """
    timestamps = []
    similarities = []

    for i in range(1, len(frames)):
        prev_emb = np.array(frames[i - 1]["embedding"], dtype=np.float32)
        curr_emb = np.array(frames[i]["embedding"], dtype=np.float32)

        # Cosine similarity (embeddings should be normalized)
        similarity = float(np.dot(prev_emb, curr_emb))
        timestamps.append(frames[i]["timestamp"])
        similarities.append(similarity)

    return timestamps, similarities


def detect_transitions(
    timestamps: list[float],
    similarities: list[float],
    threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """
    Detect transition points where similarity drops below threshold.

    Args:
        timestamps: Frame timestamps
        similarities: Consecutive similarity values
        threshold: Similarity threshold for transition detection

    Returns:
        List of transition dicts with timestamp, similarity, and drop magnitude
    """
    transitions = []

    for i, (ts, sim) in enumerate(zip(timestamps, similarities)):
        if sim < threshold:
            # Calculate drop from local context
            prev_sims = similarities[max(0, i - 3) : i]
            avg_prev = np.mean(prev_sims) if prev_sims else 1.0
            drop = avg_prev - sim

            transitions.append(
                {
                    "timestamp": ts,
                    "timestamp_str": format_timestamp(ts),
                    "similarity": sim,
                    "drop_magnitude": drop,
                    "frame_index": i + 1,  # +1 because similarities start at frame 1
                }
            )

    return transitions


def compute_novelty_scores(
    frames: list[dict[str, Any]], window: int = 5
) -> tuple[list[float], list[float]]:
    """
    Compute novelty score for each frame (how different from neighbors).

    Novelty = 1 - average similarity to surrounding frames

    Args:
        frames: List of frame dicts
        window: Number of frames on each side to compare

    Returns:
        Tuple of (timestamps, novelty_scores)
    """
    timestamps = []
    novelty_scores = []

    for i in range(len(frames)):
        curr_emb = np.array(frames[i]["embedding"], dtype=np.float32)
        similarities = []

        # Compare to surrounding frames
        for j in range(max(0, i - window), min(len(frames), i + window + 1)):
            if j != i:
                other_emb = np.array(frames[j]["embedding"], dtype=np.float32)
                similarities.append(float(np.dot(curr_emb, other_emb)))

        avg_similarity = np.mean(similarities) if similarities else 1.0
        novelty = 1.0 - avg_similarity

        timestamps.append(frames[i]["timestamp"])
        novelty_scores.append(novelty)

    return timestamps, novelty_scores


def identify_segments(
    transitions: list[dict[str, Any]],
    total_duration: float,
    min_segment_duration: float = 10.0,
) -> list[dict[str, Any]]:
    """
    Group frames into segments based on transition points.

    Args:
        transitions: List of detected transitions
        total_duration: Total scene duration
        min_segment_duration: Minimum segment length to report

    Returns:
        List of segment dicts with start/end times and duration
    """
    # Build segment boundaries
    boundaries = [0.0]
    boundaries.extend([t["timestamp"] for t in transitions])
    boundaries.append(total_duration)

    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        duration = end - start

        if duration >= min_segment_duration:
            segments.append(
                {
                    "segment_index": len(segments),
                    "start_time": start,
                    "start_str": format_timestamp(start),
                    "end_time": end,
                    "end_str": format_timestamp(end),
                    "duration": duration,
                    "duration_str": format_timestamp(duration),
                }
            )

    return segments


def create_interactive_plot(
    scene_id: int,
    timestamps: list[float],
    similarities: list[float],
    novelty_timestamps: list[float],
    novelty_scores: list[float],
    transitions: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    threshold: float,
    stats: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create interactive Plotly visualization of scene structure analysis.
    """
    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Consecutive Frame Similarity (lower = more change)",
            "Frame Novelty (higher = more different from neighbors)",
            f"Detected Segments ({len(segments)} segments)",
        ),
        row_heights=[0.4, 0.35, 0.25],
    )

    # Color scheme
    similarity_color = "#10b981"  # Emerald
    novelty_color = "#8b5cf6"  # Purple
    transition_color = "#ef4444"  # Red
    threshold_color = "#f59e0b"  # Amber

    # =====================================================================
    # Plot 1: Consecutive Similarity
    # =====================================================================

    # Create hover text with formatted timestamps
    sim_hover = [
        f"Time: {format_timestamp(ts)}<br>Similarity: {sim:.4f}<br>Frame: {i + 1}"
        for i, (ts, sim) in enumerate(zip(timestamps, similarities))
    ]

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=similarities,
            mode="lines",
            name="Similarity",
            line=dict(color=similarity_color, width=1.5),
            hovertext=sim_hover,
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=threshold_color,
        annotation_text=f"Threshold: {threshold}",
        annotation_position="top right",
        row=1,
        col=1,
    )

    # Mark transitions
    if transitions:
        trans_x = [t["timestamp"] for t in transitions]
        trans_y = [t["similarity"] for t in transitions]
        trans_hover = [
            f"<b>TRANSITION</b><br>"
            f"Time: {t['timestamp_str']}<br>"
            f"Similarity: {t['similarity']:.4f}<br>"
            f"Drop: {t['drop_magnitude']:.4f}"
            for t in transitions
        ]

        fig.add_trace(
            go.Scatter(
                x=trans_x,
                y=trans_y,
                mode="markers",
                name=f"Transitions ({len(transitions)})",
                marker=dict(
                    color=transition_color,
                    size=8,
                    symbol="triangle-down",
                ),
                hovertext=trans_hover,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # =====================================================================
    # Plot 2: Novelty Scores
    # =====================================================================

    nov_hover = [
        f"Time: {format_timestamp(ts)}<br>Novelty: {nov:.4f}<br>Frame: {i}"
        for i, (ts, nov) in enumerate(zip(novelty_timestamps, novelty_scores))
    ]

    # Fill area under curve
    fig.add_trace(
        go.Scatter(
            x=novelty_timestamps,
            y=novelty_scores,
            mode="lines",
            name="Novelty",
            line=dict(color=novelty_color, width=1),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.2)",
            hovertext=nov_hover,
            hoverinfo="text",
        ),
        row=2,
        col=1,
    )

    # 95th percentile threshold
    novelty_p95 = float(np.percentile(novelty_scores, 95))
    fig.add_hline(
        y=novelty_p95,
        line_dash="dot",
        line_color=novelty_color,
        annotation_text=f"95th percentile: {novelty_p95:.4f}",
        annotation_position="top right",
        row=2,
        col=1,
    )

    # Mark high novelty spikes
    high_novelty_indices = [i for i, nov in enumerate(novelty_scores) if nov > novelty_p95]
    if high_novelty_indices:
        spike_x = [novelty_timestamps[i] for i in high_novelty_indices]
        spike_y = [novelty_scores[i] for i in high_novelty_indices]
        spike_hover = [
            f"<b>HIGH NOVELTY</b><br>"
            f"Time: {format_timestamp(novelty_timestamps[i])}<br>"
            f"Novelty: {novelty_scores[i]:.4f}"
            for i in high_novelty_indices
        ]

        fig.add_trace(
            go.Scatter(
                x=spike_x,
                y=spike_y,
                mode="markers",
                name=f"High Novelty ({len(spike_x)})",
                marker=dict(
                    color=novelty_color,
                    size=6,
                    symbol="circle",
                ),
                hovertext=spike_hover,
                hoverinfo="text",
            ),
            row=2,
            col=1,
        )

    # =====================================================================
    # Plot 3: Segment Timeline
    # =====================================================================

    # Segment colors from a nice palette
    segment_colors = [
        "#3b82f6",
        "#10b981",
        "#f59e0b",
        "#ef4444",
        "#8b5cf6",
        "#ec4899",
        "#06b6d4",
        "#84cc16",
        "#f97316",
        "#6366f1",
    ]

    # Add segments as filled rectangles
    for i, seg in enumerate(segments):
        color = segment_colors[i % len(segment_colors)]

        # Add rectangle for segment
        fig.add_shape(
            type="rect",
            x0=seg["start_time"],
            x1=seg["end_time"],
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=0.4,
            line=dict(width=0),
            row=3,
            col=1,
        )

        # Add invisible scatter for hover info
        mid_time = (seg["start_time"] + seg["end_time"]) / 2
        fig.add_trace(
            go.Scatter(
                x=[mid_time],
                y=[0.5],
                mode="markers+text",
                name=f"Seg {i}",
                marker=dict(size=1, opacity=0),
                text=[f"Seg {i}"],
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                hovertext=[
                    f"<b>Segment {i}</b><br>"
                    f"Start: {seg['start_str']}<br>"
                    f"End: {seg['end_str']}<br>"
                    f"Duration: {seg['duration_str']}"
                ],
                hoverinfo="text",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Add transition markers on segment timeline
    for t in transitions:
        fig.add_vline(
            x=t["timestamp"],
            line_color=transition_color,
            line_width=2,
            line_dash="solid",
            row=3,
            col=1,
        )

    # =====================================================================
    # Layout and styling
    # =====================================================================

    duration_str = format_timestamp(stats["duration"])
    title_text = (
        f"<b>Scene {scene_id} Structure Analysis</b><br>"
        f"<sup>Duration: {duration_str} | Frames: {stats['frame_count']} | "
        f"Transitions: {stats['transition_count']} | Segments: {stats['segment_count']}</sup>"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=900,
        hovermode="x unified",
        template="plotly_dark",
    )

    # Y-axis labels
    fig.update_yaxes(title_text="Similarity", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Novelty", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=3, col=1, range=[0, 1])

    # X-axis with range slider on bottom
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        row=3,
        col=1,
    )

    # Save to HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    print(f"   Saved interactive plot: {output_path}")


def analyze_scene_structure(
    scene_id: int,
    storage: EmbeddingStorage,
    threshold: float = 0.85,
    min_segment_duration: float = 10.0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Analyze scene structure using frame embeddings.

    Args:
        scene_id: Scene ID to analyze
        storage: EmbeddingStorage instance
        threshold: Similarity threshold for transition detection
        min_segment_duration: Minimum segment length to report
        output_dir: Directory to save plot (None = don't save)

    Returns:
        Analysis results dict
    """
    # Load frames
    frames = storage._load_all_frames_for_scene(scene_id)

    if not frames:
        return {"scene_id": scene_id, "error": f"No frame embeddings found for scene {scene_id}"}

    if len(frames) < 2:
        return {
            "scene_id": scene_id,
            "error": f"Scene {scene_id} has only {len(frames)} frame(s), need at least 2",
        }

    # Get duration from frames
    duration = frames[-1]["timestamp"] + 1.0

    # Compute consecutive similarity
    timestamps, similarities = compute_consecutive_similarity(frames)

    # Detect transitions
    transitions = detect_transitions(timestamps, similarities, threshold)

    # Compute novelty scores
    novelty_timestamps, novelty_scores = compute_novelty_scores(frames)

    # Identify segments
    segments = identify_segments(transitions, duration, min_segment_duration)

    # Statistics
    stats = {
        "frame_count": len(frames),
        "duration": duration,
        "duration_str": format_timestamp(duration),
        "similarity_mean": float(np.mean(similarities)),
        "similarity_std": float(np.std(similarities)),
        "similarity_min": float(np.min(similarities)),
        "similarity_max": float(np.max(similarities)),
        "novelty_mean": float(np.mean(novelty_scores)),
        "novelty_max": float(np.max(novelty_scores)),
        "transition_count": len(transitions),
        "segment_count": len(segments),
    }

    result = {
        "scene_id": scene_id,
        "stats": stats,
        "transitions": transitions,
        "segments": segments,
        "threshold": threshold,
    }

    # Generate interactive plot if output directory specified
    if output_dir:
        output_path = output_dir / f"scene_{scene_id}_structure.html"
        create_interactive_plot(
            scene_id=scene_id,
            timestamps=timestamps,
            similarities=similarities,
            novelty_timestamps=novelty_timestamps,
            novelty_scores=novelty_scores,
            transitions=transitions,
            segments=segments,
            threshold=threshold,
            stats=stats,
            output_path=output_path,
        )
        result["plot_path"] = str(output_path)

    return result


def print_analysis_report(result: dict[str, Any]) -> None:
    """Print formatted analysis report to console."""
    if "error" in result:
        print(f"\n  ERROR: {result['error']}")
        return

    scene_id = result["scene_id"]
    stats = result["stats"]
    transitions = result["transitions"]
    segments = result["segments"]

    print(f"\n{'=' * 60}")
    print(f"  Scene {scene_id} - Structure Analysis Report")
    print(f"{'=' * 60}")

    print("\n  Overview:")
    print(f"   Duration:      {stats['duration_str']} ({stats['duration']:.1f}s)")
    print(f"   Frames:        {stats['frame_count']}")
    print(f"   Threshold:     {result['threshold']}")

    print("\n  Similarity Statistics:")
    print(f"   Mean:          {stats['similarity_mean']:.4f}")
    print(f"   Std Dev:       {stats['similarity_std']:.4f}")
    print(f"   Range:         [{stats['similarity_min']:.4f}, {stats['similarity_max']:.4f}]")

    print("\n  Novelty Statistics:")
    print(f"   Mean:          {stats['novelty_mean']:.4f}")
    print(f"   Max:           {stats['novelty_max']:.4f}")

    print(f"\n  Transitions Detected: {stats['transition_count']}")
    if transitions:
        print("   " + "-" * 45)
        print("   Timestamp       Similarity    Drop")
        print("   " + "-" * 45)
        # Sort by drop magnitude to show most significant first
        sorted_trans = sorted(transitions, key=lambda x: x["drop_magnitude"], reverse=True)
        for t in sorted_trans[:10]:  # Show top 10 by magnitude
            print(
                f"   {t['timestamp_str']:>12}      {t['similarity']:.4f}       {t['drop_magnitude']:.4f}"
            )
        if len(transitions) > 10:
            print(f"   ... and {len(transitions) - 10} more transitions")

    print(f"\n  Segments Identified: {stats['segment_count']}")
    if segments:
        print("   " + "-" * 50)
        print("   #     Start        End          Duration")
        print("   " + "-" * 50)
        for seg in segments[:15]:  # Show first 15
            print(
                f"   {seg['segment_index']:<5} {seg['start_str']:>8}    ->  {seg['end_str']:>8}    {seg['duration_str']:>8}"
            )
        if len(segments) > 15:
            print(f"   ... and {len(segments) - 15} more segments")

    if "plot_path" in result:
        print(f"\n  Interactive plot: {result['plot_path']}")

    print()


def detect_model_key() -> str:
    """Auto-detect the model key used for frame embeddings."""
    storage = EmbeddingStorage(model_key="siglip")  # Temporary to connect
    conn = storage._get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT model_key, COUNT(*) as cnt
        FROM frame_embeddings
        GROUP BY model_key
        ORDER BY cnt DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()

    if row:
        return row["model_key"]
    return "siglip"  # Default fallback


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide at least one scene ID")
        print("Example: uv run python poc_scene_structure.py 10020 4858 10123")
        sys.exit(1)

    # Parse scene IDs
    scene_ids = []
    for arg in sys.argv[1:]:
        try:
            scene_ids.append(int(arg))
        except ValueError:
            print(f"Warning: Skipping invalid scene ID: {arg}")

    if not scene_ids:
        print("Error: No valid scene IDs provided")
        sys.exit(1)

    # Auto-detect model key from existing embeddings
    model_key = detect_model_key()
    storage = EmbeddingStorage(model_key=model_key)

    # Output directory for plots
    output_dir = project_root / "assets" / "structure_analysis"

    print("\n  Scene Structure Detection POC (Interactive Plotly)")
    print(f"   Analyzing {len(scene_ids)} scene(s)")
    print(f"   Model key: {storage.model_key}")
    print(f"   Output dir: {output_dir}")

    # Analyze each scene
    results = []
    for scene_id in scene_ids:
        print(f"\n  Processing scene {scene_id}...")
        result = analyze_scene_structure(
            scene_id=scene_id,
            storage=storage,
            threshold=0.85,
            min_segment_duration=10.0,
            output_dir=output_dir,
        )
        results.append(result)
        print_analysis_report(result)

    # Summary
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")
        for r in results:
            if "error" in r:
                print(f"   Scene {r.get('scene_id', '?')}: Error - {r['error']}")
            else:
                s = r["stats"]
                print(
                    f"   Scene {r['scene_id']}: {s['segment_count']} segments, "
                    f"{s['transition_count']} transitions, "
                    f"sim={s['similarity_mean']:.3f}+/-{s['similarity_std']:.3f}"
                )

    print(f"\n  Analysis complete! HTML files saved to: {output_dir}")


if __name__ == "__main__":
    main()
