#!/usr/bin/env python3
"""Build a clean training CSV from metadata.jsonl, filtering out junk frames.

Reads the existing dataset WITHOUT modifying any files.  Outputs a single CSV
with (filepath, caption) rows suitable for open_clip or clipora training.

Filters applied:
  1. Error captions       — starts with "[ERROR"
  2. Black / blank frames — solid color screens, empty frames
  3. Logo-only frames     — frames showing only a logo or icon
  4. Title pages          — title cards, intro/outro screens, credits, promo text

Usage:
    uv run python tools/dataset/build_clean_csv.py
    uv run python tools/dataset/build_clean_csv.py --output custom_name.csv
    uv run python tools/dataset/build_clean_csv.py --stats  # print stats only, no CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[2] / "assets" / "lora_dataset"
IMAGES_DIR = DATASET_DIR / "images"
METADATA_PATH = DATASET_DIR / "metadata.jsonl"

# ── Filter patterns ──────────────────────────────────────────────────────

# Exact-ish matches for very short black/blank captions (case-insensitive)
_BLACK_EXACT: set[str] = {
    "black frame", "black frame.", "a black frame", "a black frame.",
    "black screen", "black screen.", "a black screen", "a black screen.",
    "black image", "black image.", "a black image", "a black image.",
    "solid black frame", "a solid black frame", "a solid black frame.",
    "solid black screen", "a solid black screen", "a solid black screen.",
    "completely black frame", "a completely black frame",
    "entirely black frame", "an entirely black frame",
    "solid black", "completely black", "entirely black",
}

# Keywords that indicate the frame is a title/logo/non-content frame.
# Matched against the FULL caption (case-insensitive).
_JUNK_KEYWORDS: list[str] = [
    "title card",
    "title screen",
    "title frame",
    "title page",
    "intro screen",
    "intro card",
    "intro frame",
    "outro screen",
    "outro card",
    "outro frame",
    "end screen",
    "end card",
    "end frame",
    "splash screen",
    "credits",
    "copyright notice",
]

# Regex patterns for captions whose primary content is logos, icons,
# watermarks, or promotional text on a plain background.
_JUNK_RE: list[re.Pattern[str]] = [
    # "X logo on a black/white/solid background"
    re.compile(
        r"\b(logo|icon|watermark)\b.{0,40}\b(black|white|solid|dark|plain)\s+(background|screen|frame)\b",
        re.IGNORECASE,
    ),
    # "on a solid black/white background" as the primary frame description
    re.compile(
        r"^.{0,60}\bon a (solid )?(black|white|dark) (background|screen)\b",
        re.IGNORECASE,
    ),
    # "black frame with text/logo/watermark"
    re.compile(
        r"^a?\s*(solid |plain |dark )?(black|white)\s*(frame|screen|image|background)\s*(with|featuring|displaying|containing)\s*(the\s+)?(text|logo|watermark|icon|word|url|link)",
        re.IGNORECASE,
    ),
    # "text ... on a black background" for short captions (promo/title text)
    re.compile(
        r"^.{0,40}(text|font|letter).{0,60}(black|dark|solid)\s*(background|screen|frame)",
        re.IGNORECASE,
    ),
]


def is_error(caption: str) -> bool:
    """Caption is an error placeholder."""
    return caption.startswith("[ERROR")


def is_black_or_blank(caption: str) -> bool:
    """Caption describes a black, blank, or solid-color frame."""
    stripped = caption.strip().rstrip(".")
    if stripped.lower() in _BLACK_EXACT:
        return True

    low = caption.lower()

    # Short captions that are just "a [color] frame/screen"
    if len(caption) < 60 and re.match(
        r"^(a |the |an )?(solid |plain |completely |entirely )?"
        r"(black|dark|white|blank|empty)\s*(frame|screen|image)\.?$",
        low,
    ):
        return True

    # "solid black frame with ..." — these are title/watermark frames
    # captured by _JUNK_RE, but short solid-color descriptions caught here
    if len(caption) < 50 and any(
        kw in low for kw in ("solid black", "solid white", "completely black", "entirely black")
    ):
        return True

    return False


def is_title_or_logo(caption: str) -> bool:
    """Caption describes a title card, logo, icon, or promo frame."""
    low = caption.lower()

    # Keyword matches — the caption mentions these as the primary subject
    for kw in _JUNK_KEYWORDS:
        if kw in low:
            return True

    # Short captions about logos/watermarks/text on backgrounds
    for pattern in _JUNK_RE:
        if pattern.search(caption):
            # For longer captions (>200 chars), only filter if the FIRST
            # sentence is the junk match — the rest might describe real content
            if len(caption) > 200:
                first_sentence = caption.split(".")[0]
                if not pattern.search(first_sentence):
                    return False
            return True

    # Catch remaining promo/text-only frames: short captions dominated by
    # references to text, URLs, handles, or brand names on plain backgrounds
    if len(caption) < 150:
        text_signals = sum(1 for kw in (
            "text that reads", "text reading", "featuring the text",
            "with the text", "displays the text", "displaying",
            "written in", "onlyfans", "twitter", "instagram",
            "subscribe", "follow me", ".com/", "@",
        ) if kw in low)
        bg_signals = sum(1 for kw in (
            "black background", "white background", "dark background",
            "solid background", "plain background", "black frame",
            "black screen",
        ) if kw in low)
        if text_signals >= 1 and bg_signals >= 1:
            return True

    return False


FilterReason = str  # "error" | "black_blank" | "title_logo" | "missing_image"


def classify_caption(name: str, caption: str) -> FilterReason | None:
    """Return the filter reason if the caption should be excluded, else None."""
    if is_error(caption):
        return "error"
    if is_black_or_blank(caption):
        return "black_blank"
    if is_title_or_logo(caption):
        return "title_logo"
    # Verify the image file exists on disk
    if not (IMAGES_DIR / name).exists():
        return "missing_image"
    return None


# ── Main ─────────────────────────────────────────────────────────────────


def load_metadata() -> list[dict]:
    """Load all records from metadata.jsonl."""
    records: list[dict] = []
    for line in METADATA_PATH.read_text(encoding="utf-8").strip().splitlines():
        if line:
            records.append(json.loads(line))
    return records


def build_clean_csv(
    output_path: Path,
    stats_only: bool = False,
) -> dict[str, int]:
    """Build filtered CSV and return statistics."""
    records = load_metadata()

    # Counters
    total = 0
    kept = 0
    filtered: dict[str, int] = {
        "error": 0,
        "black_blank": 0,
        "title_logo": 0,
        "missing_image": 0,
    }
    filtered_examples: dict[str, list[str]] = {k: [] for k in filtered}

    rows: list[tuple[str, str]] = []

    for record in records:
        captions = record.get("captions", {})
        if not isinstance(captions, dict):
            continue

        for name, caption in captions.items():
            total += 1
            reason = classify_caption(name, caption)
            if reason is not None:
                filtered[reason] += 1
                if len(filtered_examples[reason]) < 5:
                    filtered_examples[reason].append(f"{name}: {caption[:120]}")
                continue

            filepath = str(IMAGES_DIR / name)
            rows.append((filepath, caption))
            kept += 1

    stats = {
        "total_captions": total,
        "kept": kept,
        "filtered_total": total - kept,
        **filtered,
    }

    # Print report
    print(f"{'='*60}")
    print(f"  Clean Dataset Builder")
    print(f"{'='*60}")
    print(f"  Source:          {METADATA_PATH}")
    print(f"  Records:         {len(records):,}")
    print(f"  Total captions:  {total:,}")
    print(f"{'─'*60}")
    print(f"  KEPT:            {kept:,} ({kept/total*100:.1f}%)")
    print(f"  Filtered out:    {total - kept:,} ({(total-kept)/total*100:.1f}%)")
    print(f"{'─'*60}")
    print(f"  Errors:          {filtered['error']:,}")
    print(f"  Black/blank:     {filtered['black_blank']:,}")
    print(f"  Title/logo:      {filtered['title_logo']:,}")
    print(f"  Missing image:   {filtered['missing_image']:,}")
    print(f"{'─'*60}")

    for reason, examples in filtered_examples.items():
        if examples:
            print(f"\n  Sample {reason} captions:")
            for ex in examples:
                print(f"    - {ex}")

    if not stats_only:
        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "caption"])
            writer.writerows(rows)

        print(f"\n{'='*60}")
        print(f"  Output: {output_path}")
        print(f"  Rows:   {kept:,}")
        print(f"{'='*60}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean training CSV from metadata.jsonl, filtering junk frames.",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DATASET_DIR / "train_clean.csv"),
        help="Output CSV path (default: assets/lora_dataset/train_clean.csv)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print filter statistics only, don't write CSV",
    )
    args = parser.parse_args()

    build_clean_csv(Path(args.output), stats_only=args.stats)


if __name__ == "__main__":
    main()
