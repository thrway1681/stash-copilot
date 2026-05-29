"""Process a batch of scenes: copy frames, generate captions, update progress.

This script is designed to be run interactively each session.
For each scene in the batch, it prints the analysis frame paths so Claude
can read the images and provide augmented captions.

Usage:
    uv run python tools/dataset/process_batch.py [--batch-size N] [--dry-run]
"""
import argparse
from pathlib import Path

from tools.dataset.caption_generator import generate_caption
from tools.dataset.constants import DATASET_DIR
from tools.dataset.io_utils import (
    append_metadata_record,
    copy_frames_to_dataset,
    write_caption_files,
)
from tools.dataset.progress import (
    get_next_batch,
    load_progress,
    mark_scene_complete,
    save_progress,
    session_summary,
)


def process_scene(
    entry: dict,
    caption: str,
    visual_notes: str,
    missing_tags: list[str],
    dataset_dir: Path = DATASET_DIR,
    dry_run: bool = False,
) -> list[str]:
    """Copy frames and write caption files for one scene.

    Args:
        entry: Work queue entry from progress.json.
        caption: Final caption (tag-derived + visual observations).
        visual_notes: Free-text notes from visual frame analysis.
        missing_tags: Tag names observed but not in Stash.
        dataset_dir: Root dataset directory.
        dry_run: If True, don't write files.

    Returns:
        List of destination image filenames written.
    """
    images_dir = dataset_dir / "images"
    frame_paths = [Path(p) for p in entry["frame_paths"]]
    scene_id = entry["scene_id"]

    if dry_run:
        print(f"  [dry-run] Would copy {len(frame_paths)} frames for scene {scene_id}")
        return []

    image_names = copy_frames_to_dataset(scene_id, frame_paths, images_dir)
    write_caption_files(image_names, caption, images_dir)
    append_metadata_record(
        jsonl_path=dataset_dir / "metadata.jsonl",
        scene_id=scene_id,
        tags=entry["tags"],
        caption=caption,
        visual_notes=visual_notes,
        missing_tags=missing_tags,
        image_names=image_names,
    )
    return image_names


def run_batch(batch_size: int = 150, dry_run: bool = False) -> None:
    """Print the next batch of scenes for interactive visual analysis."""
    progress_path = DATASET_DIR / "progress.json"
    if not progress_path.exists():
        print("ERROR: progress.json not found. Run init_dataset.py first.")
        return

    progress = load_progress(progress_path)
    progress["sessions"] = progress.get("sessions", 0) + 1

    print(session_summary(progress))
    print()

    batch = get_next_batch(progress, batch_size)
    if not batch:
        print("All scenes processed! Run finalize_dataset.py to generate CSVs.")
        return

    print(f"Processing batch of {len(batch)} scenes.\n")
    print("=" * 70)

    for i, entry in enumerate(batch, 1):
        scene_id = entry["scene_id"]
        tags = entry["tags"]
        performers = entry.get("performers", [])
        studio = entry.get("studio")
        analysis_frames = entry.get("analysis_frames", [])

        baseline_caption = generate_caption(tags, performers=performers, studio=studio)

        print(f"\n[{i}/{len(batch)}] Scene {scene_id}")
        print(f"  Tags: {', '.join(tags)}")
        if performers:
            print(f"  Performers: {', '.join(performers)}")
        print(f"  Baseline caption: {baseline_caption}")
        print(f"  Analysis frames:")
        for f in analysis_frames:
            print(f"    {f}")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a batch of scenes for the LoRA dataset.")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_batch(batch_size=args.batch_size, dry_run=args.dry_run)
