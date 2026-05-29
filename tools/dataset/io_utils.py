"""File I/O utilities for dataset construction."""
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def dataset_image_name(scene_id: str, frame_path: Path) -> str:
    """Convert a source frame path to a flat dataset image filename.

    Example: scene_19/frame_0001.jpg -> s19_f0001.jpg
    """
    stem = frame_path.stem
    num = stem.replace("frame_", "")
    return f"s{scene_id}_f{num}.jpg"


def copy_frames_to_dataset(
    scene_id: str,
    frame_paths: list[Path],
    dest_dir: Path,
) -> list[str]:
    """Copy frame files to the flat dataset images directory.

    Does NOT delete the source. Returns list of destination filenames.
    """
    copied: list[str] = []
    for src in frame_paths:
        name = dataset_image_name(scene_id, src)
        dest = dest_dir / name
        if not dest.exists():
            shutil.copy2(src, dest)
        copied.append(name)
    return copied


def write_caption_files(
    image_names: list[str],
    caption: str,
    dest_dir: Path,
) -> None:
    """Write a .txt caption file alongside each image file."""
    for name in image_names:
        txt_path = dest_dir / name.replace(".jpg", ".txt")
        txt_path.write_text(caption, encoding="utf-8")


def append_metadata_record(
    jsonl_path: Path,
    scene_id: str,
    tags: list[str],
    caption: str,
    visual_notes: str,
    missing_tags: list[str],
    image_names: list[str],
) -> None:
    """Append one scene record to the metadata JSONL file."""
    record = {
        "scene_id": scene_id,
        "tags": tags,
        "caption": caption,
        "visual_notes": visual_notes,
        "missing_tags": missing_tags,
        "image_names": image_names,
        "analyzed_at": datetime.now(UTC).isoformat(),
    }
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
