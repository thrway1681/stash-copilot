#!/usr/bin/env python3
"""
POC: Smart Frame Selection Visualization

Demonstrates novelty-based frame selection on a scene with existing frame cache.
Outputs a montage showing which frames would be sent to a VLM.

Usage:
    uv run python poc_smart_selection.py --scene-id 123 --max-frames 50 --output montage.png
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class FrameSelection:
    """A selected frame with metadata."""

    index: int
    timestamp: float
    novelty_score: float
    selection_reason: str  # "temporal" | "novelty"
    frame_path: str


def compute_novelty_scores(
    embeddings: NDArray[np.float32], window_size: int = 2
) -> NDArray[np.float64]:
    """
    Compute novelty score for each frame.

    Novelty = 1 - similarity to temporal neighbors.
    High novelty indicates visual change (action, position change, climax, etc.)

    Args:
        embeddings: Frame embeddings as numpy array (N, D)
        window_size: Number of frames before/after to compare against

    Returns:
        Novelty scores for each frame (0-1 range, higher = more novel)
    """
    n_frames = len(embeddings)
    novelty: NDArray[np.float64] = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        start = max(0, i - window_size)
        end = min(n_frames, i + window_size + 1)
        neighbors = [j for j in range(start, end) if j != i]

        if not neighbors:
            novelty[i] = 0.0
            continue

        neighbor_embs = embeddings[neighbors]
        similarities: NDArray[np.float32] = np.dot(neighbor_embs, embeddings[i])
        novelty[i] = 1.0 - float(np.mean(similarities))

    return novelty


def select_frames(
    frame_paths: list[str],
    embeddings: NDArray[np.float32],
    timestamps: list[float],
    max_frames: int = 50,
    temporal_ratio: float = 0.5,
    dedup_threshold: float = 0.85,
) -> list[FrameSelection]:
    """
    Select frames using temporal + novelty approach.

    Algorithm:
    1. Select temporal_ratio * max_frames uniformly spaced (baseline coverage)
    2. Fill remaining slots with highest-novelty frames not already selected
    3. Deduplicate by skipping frames too similar to already-selected ones

    Args:
        frame_paths: List of frame file paths
        embeddings: Frame embeddings (N, D)
        timestamps: Frame timestamps in seconds
        max_frames: Target number of frames
        temporal_ratio: Fraction of budget for uniform temporal sampling
        dedup_threshold: Skip frames with similarity > this to selected frames

    Returns:
        List of selected frames with metadata
    """
    n_frames = len(embeddings)
    novelty_scores = compute_novelty_scores(embeddings)

    selected_indices: set[int] = set()
    selections: list[FrameSelection] = []

    # Phase 1: Temporal baseline (evenly spaced frames)
    temporal_count = int(max_frames * temporal_ratio)
    temporal_indices = np.linspace(0, n_frames - 1, temporal_count, dtype=int)

    for idx in temporal_indices:
        idx = int(idx)
        selected_indices.add(idx)
        selections.append(
            FrameSelection(
                index=idx,
                timestamp=timestamps[idx],
                novelty_score=float(novelty_scores[idx]),
                selection_reason="temporal",
                frame_path=frame_paths[idx],
            )
        )

    # Phase 2: Novelty boost (highest novelty frames)
    novelty_ranked = np.argsort(novelty_scores)[::-1]

    for idx in novelty_ranked:
        idx = int(idx)
        if len(selections) >= max_frames:
            break
        if idx in selected_indices:
            continue

        # Deduplication check
        too_similar = False
        for sel in selections:
            sim = float(np.dot(embeddings[idx], embeddings[sel.index]))
            if sim > dedup_threshold:
                too_similar = True
                break

        if too_similar:
            continue

        selected_indices.add(idx)
        selections.append(
            FrameSelection(
                index=idx,
                timestamp=timestamps[idx],
                novelty_score=float(novelty_scores[idx]),
                selection_reason="novelty",
                frame_path=frame_paths[idx],
            )
        )

    # Sort by timestamp for chronological order
    selections.sort(key=lambda x: x.timestamp)
    return selections


def create_montage(
    selections: list[FrameSelection],
    output_path: str,
    cols: int = 10,
    thumb_size: tuple[int, int] = (160, 90),
) -> None:
    """
    Create a visual montage of selected frames.

    Green border = temporal baseline
    Orange border = novelty-selected (action moments)

    Args:
        selections: List of selected frames
        output_path: Output image path
        cols: Number of columns in the grid
        thumb_size: Size of each thumbnail (width, height)
    """
    rows = (len(selections) + cols - 1) // cols
    montage_width = cols * thumb_size[0]
    montage_height = rows * (thumb_size[1] + 30)  # Extra space for labels

    montage = Image.new("RGB", (montage_width, montage_height), (30, 30, 30))
    draw = ImageDraw.Draw(montage)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()

    for i, sel in enumerate(selections):
        row = i // cols
        col = i % cols
        x = col * thumb_size[0]
        y = row * (thumb_size[1] + 30)

        # Load frame image
        if Path(sel.frame_path).exists():
            img = Image.open(sel.frame_path)
            # Resize preserving aspect ratio
            img.thumbnail(thumb_size)
            # Center in the cell
            paste_x = x + (thumb_size[0] - img.width) // 2
            paste_y = y + (thumb_size[1] - img.height) // 2
            montage.paste(img, (paste_x, paste_y))

        # Draw border based on selection reason
        border_color = (0, 200, 100) if sel.selection_reason == "temporal" else (255, 150, 0)
        draw.rectangle(
            [x, y, x + thumb_size[0] - 1, y + thumb_size[1] - 1],
            outline=border_color,
            width=2,
        )

        # Draw label
        mins = int(sel.timestamp // 60)
        secs = int(sel.timestamp % 60)
        label = f"{mins}:{secs:02d} | N:{sel.novelty_score:.2f}"
        draw.text((x + 2, y + thumb_size[1] + 2), label, fill=(200, 200, 200), font=font)

        # Draw selection type indicator
        type_label = "T" if sel.selection_reason == "temporal" else "N"
        draw.text((x + thumb_size[0] - 12, y + 2), type_label, fill=border_color, font=font)

    montage.save(output_path)
    print(f"Montage saved to: {output_path}")


def create_timeline(
    all_novelty_scores: NDArray[np.float64],
    all_timestamps: list[float],
    selections: list[FrameSelection],
    output_path: str,
    width: int = 1600,
    height: int = 400,
) -> None:
    """
    Create a timeline chart showing novelty scores and selected frames.

    Args:
        all_novelty_scores: Novelty scores for all frames
        all_timestamps: Timestamps for all frames
        selections: Selected frames
        output_path: Output image path
        width: Chart width
        height: Chart height
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping timeline chart")
        return

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Plot all novelty scores as a line
    ax.plot(all_timestamps, all_novelty_scores, color="gray", alpha=0.5, linewidth=0.5)

    # Separate temporal and novelty selections
    temporal = [s for s in selections if s.selection_reason == "temporal"]
    novelty = [s for s in selections if s.selection_reason == "novelty"]

    # Plot temporal selections
    if temporal:
        ax.scatter(
            [s.timestamp for s in temporal],
            [s.novelty_score for s in temporal],
            c="green",
            s=50,
            label=f"Temporal ({len(temporal)})",
            zorder=5,
        )

    # Plot novelty selections
    if novelty:
        ax.scatter(
            [s.timestamp for s in novelty],
            [s.novelty_score for s in novelty],
            c="orange",
            s=50,
            label=f"Novelty ({len(novelty)})",
            zorder=5,
        )

    ax.set_xlabel("Timestamp (seconds)")
    ax.set_ylabel("Novelty Score")
    ax.set_title("Smart Frame Selection - Novelty Timeline")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Timeline saved to: {output_path}")


def load_frame_paths(cache_dir: Path) -> list[tuple[str, int]]:
    """
    Load frame paths from cache directory.

    Returns:
        List of (path, frame_number) tuples sorted by frame number
    """
    frames: list[tuple[str, int]] = []
    for f in cache_dir.glob("frame_*.jpg"):
        # Extract frame number from filename
        try:
            frame_num = int(f.stem.split("_")[1])
            frames.append((str(f), frame_num))
        except (IndexError, ValueError):
            continue

    frames.sort(key=lambda x: x[1])
    return frames


def compute_embeddings_from_frames(
    frame_paths: list[str],
    embedding_config_model: str = "openclip:ViT-bigG-14",
) -> NDArray[np.float32] | None:
    """
    Compute embeddings for frames using the embedding provider.

    Args:
        frame_paths: List of frame file paths
        embedding_config_model: Model key for embedding provider

    Returns:
        Embeddings as numpy array (N, D)
    """
    from stash_ai.embeddings.config import EmbeddingConfig
    from stash_ai.embeddings.provider import get_embedding_provider

    print(f"Computing embeddings for {len(frame_paths)} frames...")

    # Parse model key
    config = EmbeddingConfig.from_model_key(embedding_config_model)
    embedder = get_embedding_provider(config)

    if not embedder.supports_images:
        print(f"Error: Model {embedding_config_model} does not support images")
        return None

    try:
        # Embed all frames
        results = embedder.embed_images(frame_paths)
        if not results:
            print("Error: No embeddings returned")
            return None

        embeddings = np.array([r["embedding"] for r in results], dtype=np.float32)
        print(f"Computed {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
        return embeddings

    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return None
    finally:
        # Cleanup GPU resources
        if hasattr(embedder, "cleanup"):
            embedder.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="POC: Smart Frame Selection for VLM Analysis")
    parser.add_argument("--scene-id", type=int, required=True, help="Scene ID with cached frames")
    parser.add_argument("--max-frames", type=int, default=50, help="Target number of frames")
    parser.add_argument(
        "--output",
        type=str,
        default="smart_selection_montage.png",
        help="Output montage path",
    )
    parser.add_argument(
        "--temporal-ratio",
        type=float,
        default=0.5,
        help="Fraction of budget for temporal sampling (0-1)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for deduplication",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openclip:ViT-bigG-14",
        help="Embedding model key",
    )
    args = parser.parse_args()

    # Setup paths
    plugin_dir = Path(__file__).parent
    cache_dir = plugin_dir / "assets" / "embedded_frames" / f"scene_{args.scene_id}"

    if not cache_dir.exists():
        print(f"Error: Frame cache not found at {cache_dir}")
        print("Run 'Embed Scene' task first to extract frames")
        sys.exit(1)

    # Load frame paths
    frame_data = load_frame_paths(cache_dir)
    if not frame_data:
        print(f"Error: No frames found in {cache_dir}")
        sys.exit(1)

    frame_paths = [f[0] for f in frame_data]
    frame_indices = [f[1] for f in frame_data]

    # Compute timestamps (assuming 1fps extraction)
    # Frame 1 = timestamp 0, Frame 2 = timestamp 1, etc.
    timestamps = [float(idx - 1) for idx in frame_indices]

    print(f"Found {len(frame_paths)} frames for scene {args.scene_id}")
    print(f"Duration: ~{max(timestamps):.0f} seconds ({max(timestamps) / 60:.1f} minutes)")

    # Compute embeddings from frames
    embeddings = compute_embeddings_from_frames(frame_paths, args.model)
    if embeddings is None:
        print("Error: Failed to compute embeddings")
        sys.exit(1)

    # Select frames
    selections = select_frames(
        frame_paths=frame_paths,
        embeddings=embeddings,
        timestamps=timestamps,
        max_frames=args.max_frames,
        temporal_ratio=args.temporal_ratio,
        dedup_threshold=args.dedup_threshold,
    )

    # Compute novelty for all frames (for timeline)
    all_novelty = compute_novelty_scores(embeddings)

    # Stats
    temporal_count = sum(1 for s in selections if s.selection_reason == "temporal")
    novelty_count = sum(1 for s in selections if s.selection_reason == "novelty")

    print(f"\n{'=' * 50}")
    print("SELECTION RESULTS")
    print(f"{'=' * 50}")
    print(f"Total frames selected: {len(selections)}")
    print(f"  Temporal baseline: {temporal_count}")
    print(f"  Novelty boost: {novelty_count}")
    print(
        f"\nNovelty score range: "
        f"{min(s.novelty_score for s in selections):.3f} - "
        f"{max(s.novelty_score for s in selections):.3f}"
    )
    print(f"Mean novelty (selected): {np.mean([s.novelty_score for s in selections]):.3f}")
    print(f"Mean novelty (all): {np.mean(all_novelty):.3f}")

    # Temporal coverage check
    durations = [
        selections[i + 1].timestamp - selections[i].timestamp for i in range(len(selections) - 1)
    ]
    if durations:
        print("\nTemporal gaps between selected frames:")
        print(f"  Min: {min(durations):.1f}s")
        print(f"  Max: {max(durations):.1f}s")
        print(f"  Mean: {np.mean(durations):.1f}s")

    # Create output directory
    output_dir = Path(args.output).parent
    if output_dir != Path("."):
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create montage
    create_montage(selections, args.output)

    # Create timeline
    timeline_path = args.output.rsplit(".", 1)[0] + "_timeline.png"
    create_timeline(all_novelty, timestamps, selections, timeline_path)

    # Save JSON data
    json_path = args.output.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "scene_id": args.scene_id,
                "total_frames": len(frame_paths),
                "selected_frames": len(selections),
                "temporal_count": temporal_count,
                "novelty_count": novelty_count,
                "max_frames": args.max_frames,
                "temporal_ratio": args.temporal_ratio,
                "dedup_threshold": args.dedup_threshold,
                "model": args.model,
                "selections": [
                    {
                        "index": s.index,
                        "timestamp": s.timestamp,
                        "novelty_score": s.novelty_score,
                        "selection_reason": s.selection_reason,
                    }
                    for s in selections
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSelection data saved to: {json_path}")


if __name__ == "__main__":
    main()
