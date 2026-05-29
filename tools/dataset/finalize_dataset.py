"""Finalize the dataset: generate train/val CSVs, missing tags report, README.

Usage:
    uv run python tools/dataset/finalize_dataset.py
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

from tools.dataset.constants import DATASET_DIR


def load_metadata(jsonl_path: Path) -> list[dict]:
    """Load all metadata records from the JSONL file."""
    records = []
    for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines():
        if line:
            records.append(json.loads(line))
    return records


def split_train_val(
    records: list[dict],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split at the scene level (not frame level) to prevent data leakage."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def write_csv(records: list[dict], csv_path: Path, images_dir: Path) -> None:
    """Write a train or val CSV with filepath,caption columns."""
    rows = []
    for r in records:
        for name in r["image_names"]:
            rows.append({
                "filepath": str(images_dir / name),
                "caption": r["caption"],
            })
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def generate_missing_tags_report(records: list[dict]) -> str:
    """Generate a markdown report of systematically missing tags."""
    counter: Counter[str] = Counter()
    for r in records:
        for tag in r.get("missing_tags", []):
            counter[tag] += 1

    lines = [
        "# Missing Tags Report",
        "",
        "Tags observed visually but not present in Stash, by frequency:",
        "",
        "| Tag | Scenes missing it |",
        "|---|---|",
    ]
    for tag, count in counter.most_common():
        lines.append(f"| {tag} | {count} |")

    lines += [
        "",
        f"**Total records analyzed:** {len(records)}",
        f"**Records with missing tags:** {sum(1 for r in records if r.get('missing_tags'))}",
    ]
    return "\n".join(lines)


def write_readme(
    dataset_dir: Path,
    n_train: int,
    n_val: int,
    n_scenes: int,
) -> None:
    content = f"""# LoRA Training Dataset

Generated from Stash library via visual analysis.

## Statistics

| Metric | Value |
|---|---|
| Total scenes analyzed | {n_scenes:,} |
| Training pairs | {n_train:,} |
| Validation pairs | {n_val:,} |
| Total pairs | {n_train + n_val:,} |
| Split | 90% train / 10% val (scene-level) |

## Format

Flat `images/` directory with paired `.jpg` / `.txt` files.
`train.csv` and `val.csv` contain `filepath,caption` columns compatible
with `clipora` and `open_clip_train.main --dataset-type csv`.
"""
    (dataset_dir / "README.md").write_text(content, encoding="utf-8")


def finalize(dataset_dir: Path = DATASET_DIR) -> None:
    """Generate train/val CSVs, missing tags report, and README."""
    meta_path = dataset_dir / "metadata.jsonl"
    if not meta_path.exists():
        print("ERROR: metadata.jsonl not found. Process scenes first.")
        return

    records = load_metadata(meta_path)
    if not records:
        print("No records found in metadata.jsonl.")
        return

    print(f"Loaded {len(records)} scene records.")

    train_records, val_records = split_train_val(records)
    images_dir = dataset_dir / "images"

    train_csv = dataset_dir / "train.csv"
    val_csv = dataset_dir / "val.csv"
    write_csv(train_records, train_csv, images_dir)
    write_csv(val_records, val_csv, images_dir)

    n_train = sum(len(r["image_names"]) for r in train_records)
    n_val   = sum(len(r["image_names"]) for r in val_records)

    report = generate_missing_tags_report(records)
    (dataset_dir / "missing_tags_report.md").write_text(report, encoding="utf-8")

    write_readme(dataset_dir, n_train, n_val, len(records))

    print(f"\nDataset finalized:")
    print(f"  Train pairs: {n_train:,}")
    print(f"  Val pairs:   {n_val:,}")
    print(f"  Total pairs: {n_train + n_val:,}")
    print(f"  Missing tags report: {dataset_dir / 'missing_tags_report.md'}")


if __name__ == "__main__":
    finalize()
