# LoRA Dataset Construction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-contained ~78,000 image-caption dataset from the Stash library for LoRA fine-tuning of OpenCLIP ViT-H-14, with every caption visually verified by Claude reading frames directly.

**Architecture:** A set of standalone CLI tools in `tools/dataset/` initializes the dataset structure and work queue from the Stash GraphQL API, then a multi-session interactive loop processes scenes in batches — Claude reads 2 frames per scene, writes an augmented caption combining tag metadata with visual observations, copies 20 uniformly-sampled frames to a flat `images/` directory with paired `.txt` caption files, and saves progress to a JSON checkpoint for resume.

**Tech Stack:** Python 3.12+, `uv`, `requests` (Stash GraphQL), `shutil` (frame copy), `json`/`jsonlines`, `pytest`

---

## Constants and Configuration

Throughout this plan, use these values:

```python
STASH_GRAPHQL = "http://localhost:9999/graphql"
DATASET_DIR   = "~/.stash/plugins/stash-copilot/assets/lora_dataset"
FRAMES_DIR    = "~/.stash/plugins/stash-copilot/assets/embedded_frames"
MIN_CONTENT_TAGS = 5
FRAMES_PER_SCENE = 20

ADMIN_TAGS = {
    "Embedded", "To Embed", "To Script", "Funscript",
    "Missing Performer (Male)", "HD Available", "FS: Action", "FS: Beat",
    "Start", "Free stroke", "OG beat comes back", "Funk Beat",
    "Funk Beat comes back", "Jiggle Fuck", "Hip Sway", "Mixed Audio",
    "Music Only", "Event 2024", "Event 2025",
    "Custom Marker A", "Custom Marker B", "Remix", "Cumpilation",
    "[AVN Award Winner]", "[Award Winner]", "[MiscTags: Skip]",
    "[SIT: Multi-Script]", "[Set Profile Image]",
    "[Stashbox Performer Gallery]", "[TPDB: Skip Marker]",
    "[Timestamp: Skip Sync]",
}
```

---

## Task 1: Scaffold tools/dataset/ package

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/dataset/__init__.py`
- Create: `tools/dataset/constants.py`
- Test: `tests/tools/test_constants.py`

**Step 1: Create directories**

```bash
mkdir -p tools/dataset
touch tools/__init__.py tools/dataset/__init__.py
```

**Step 2: Write `tools/dataset/constants.py`**

```python
"""Shared constants for dataset construction tools."""
from pathlib import Path

STASH_GRAPHQL = "http://localhost:9999/graphql"
DATASET_DIR   = Path("~/.stash/plugins/stash-copilot/assets/lora_dataset")
FRAMES_DIR    = Path("~/.stash/plugins/stash-copilot/assets/embedded_frames")
MIN_CONTENT_TAGS = 5
FRAMES_PER_SCENE = 20

ADMIN_TAGS: frozenset[str] = frozenset({
    "Embedded", "To Embed", "To Script", "Funscript",
    "Missing Performer (Male)", "HD Available", "FS: Action", "FS: Beat",
    "Start", "Free stroke", "OG beat comes back", "Funk Beat",
    "Funk Beat comes back", "Jiggle Fuck", "Hip Sway", "Mixed Audio",
    "Music Only", "Event 2024", "Event 2025",
    "Custom Marker A", "Custom Marker B", "Remix", "Cumpilation",
    "[AVN Award Winner]", "[Award Winner]", "[MiscTags: Skip]",
    "[SIT: Multi-Script]", "[Set Profile Image]",
    "[Stashbox Performer Gallery]", "[TPDB: Skip Marker]",
    "[Timestamp: Skip Sync]",
})

# Tag-to-category classification for caption generation
BODY_TYPE_TAGS: frozenset[str] = frozenset({
    "PAWG", "PAAG", "Curvy", "Petite", "Skinny", "Fit",
    "Big Ass", "Medium Ass", "Round Ass", "Wide Hips",
    "Flat Stomach", "slim waist",
})

PHYSICAL_ATTRIBUTE_TAGS: frozenset[str] = frozenset({
    "Big Tits", "Medium Tits", "Small Tits", "Natural Tits", "Fake Tits",
    "Perfect Tits", "Saggy Tits", "Bouncing Tits",
    "Blonde Hair", "Colored Hair", "Pigtails",
    "Blue Eyes", "brown eyes",
    "Tattoos", "Piercing", "Braces", "Tan", "Tan Lines",
    "Small Nose", "slim waist", "Flat Stomach",
    "Innie", "Fat Pussy", "Hairy Pussy", "Shaved Pussy",
    "Brown Pussy", "Pink Pussy", "Pierced Pussy",
    "Big Dick", "BBC",
})

ACT_TAGS: frozenset[str] = frozenset({
    "Blowjob", "Deepthroat", "Face Fuck", "Gag", "Ball Sucking",
    "Penis Licking", "Hands Free Blowjob", "Facesitting",
    "Pussy Eating", "Pussy Licking", "Cunnilingus",
    "Rimming", "Ass Eating", "Ass to Mouth", "69",
    "Vaginal Sex", "Anal Sex", "Anal Play", "Anal Penetration",
    "Double Penetration", "Double Anal Penetration (DAP)",
    "Double Vaginal Penetration (DVP)",
    "Handjob", "Footjob", "titfuck", "Buttjob", "Grinding",
    "Masturbation", "Pussy Fingering", "Pussy Rubbing",
    "Tribbing/Scissoring", "Pegging",
    "Oral Sex", "Outercourse", "Couple Sex",
    "Gloryhole", "Gangbang", "Orgy", "Threesome", "Threesome (FFM)",
    "Lesbian",
})

OUTCOME_TAGS: frozenset[str] = frozenset({
    "Creampie", "Vaginal Creampie", "Anal Creampie", "Surprise Creampie",
    "Facial", "Facial - POV", "Open Mouth Facial",
    "Cum on Face", "Cum on Tits", "Cum on Ass", "Cum on Pussy",
    "Cum in Mouth", "Cum Swallowing", "Spit",
    "Squirting", "Ahegao",
    "Cum", "Cumshot", "Multiple Cumshots",
})

POSITION_TAGS: frozenset[str] = frozenset({
    "Missionary", "Folded Missionary",
    "Doggy Style", "Standing Doggy Style", "Prone Bone",
    "Cowgirl", "Riding", "Reverse Cowgirl", "Reverse Riding",
    "Stand And Carry", "Standing Cradle",
})

CONTENT_STYLE_TAGS: frozenset[str] = frozenset({
    "Amateur", "Homemade", "Hardcore", "Softcore", "Erotica", "Rough",
    "POV", "Male POV", "VR", "Vertical Video", "Webcam",
    "OnlyFans", "JAV", "AI Generated", "Censored", "TikTok",
    "PMV", "Animated", "3D Animated", "Furry", "Futanari", "Rule 34",
    "Compilation", "Non-Nude", "Solo",
})

SETTING_TAGS: frozenset[str] = frozenset({
    "Outdoors", "Beach", "Pool", "Gym", "Classroom",
    "Massage", "Massage Table", "Public Sex",
})
```

**Step 3: Write test**

```python
# tests/tools/test_constants.py
from tools.dataset.constants import ADMIN_TAGS, BODY_TYPE_TAGS, ACT_TAGS

def test_admin_tags_are_frozenset() -> None:
    assert isinstance(ADMIN_TAGS, frozenset)
    assert "Embedded" in ADMIN_TAGS
    assert "Funscript" in ADMIN_TAGS

def test_no_overlap_between_act_and_admin() -> None:
    # Admin tags should not appear in content tag sets
    assert not (ADMIN_TAGS & ACT_TAGS)
    assert not (ADMIN_TAGS & BODY_TYPE_TAGS)

def test_embedded_is_admin_not_content() -> None:
    assert "Embedded" in ADMIN_TAGS
    assert "Embedded" not in ACT_TAGS
    assert "Embedded" not in BODY_TYPE_TAGS
```

**Step 4: Run tests**

```bash
uv run pytest tests/tools/test_constants.py -v
```
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add tools/ tests/tools/
git commit -m "feat(dataset): scaffold tools/dataset package and tag constants"
```

---

## Task 2: Caption generator

**Files:**
- Create: `tools/dataset/caption_generator.py`
- Test: `tests/tools/test_caption_generator.py`

**Step 1: Write test first**

```python
# tests/tools/test_caption_generator.py
from tools.dataset.caption_generator import generate_caption

def test_non_nude_solo_body_tags() -> None:
    tags = ["Non-Nude", "Solo", "PAWG", "Big Ass", "Small Tits", "Tan"]
    caption = generate_caption(tags, performers=["Mikaela Lafuente"])
    assert "non-nude" in caption.lower() or "solo" in caption.lower()
    assert "PAWG" in caption or "big ass" in caption.lower()
    assert "small tits" in caption.lower()

def test_blowjob_scene() -> None:
    tags = ["Amateur", "Blowjob", "Deepthroat", "POV", "Big Tits", "Blonde Hair"]
    caption = generate_caption(tags)
    assert "blowjob" in caption.lower() or "oral" in caption.lower()
    assert "pov" in caption.lower() or "POV" in caption
    assert "big tits" in caption.lower()

def test_admin_tags_excluded() -> None:
    tags = ["Embedded", "Blowjob", "Big Tits", "To Script", "Funscript"]
    caption = generate_caption(tags)
    assert "embedded" not in caption.lower()
    assert "funscript" not in caption.lower()
    assert "blowjob" in caption.lower()

def test_empty_meaningful_tags_returns_generic() -> None:
    tags = ["Embedded", "To Embed"]
    caption = generate_caption(tags)
    assert len(caption) > 10  # Should still return something

def test_performer_included_when_provided() -> None:
    tags = ["Amateur", "Big Tits"]
    caption = generate_caption(tags, performers=["Jane Doe"])
    assert "Jane Doe" in caption or "performer" in caption.lower()
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/tools/test_caption_generator.py -v
```
Expected: ImportError (module doesn't exist yet)

**Step 3: Write `tools/dataset/caption_generator.py`**

```python
"""Generate natural language captions from Stash scene tags."""
from tools.dataset.constants import (
    ADMIN_TAGS, BODY_TYPE_TAGS, PHYSICAL_ATTRIBUTE_TAGS,
    ACT_TAGS, OUTCOME_TAGS, POSITION_TAGS, CONTENT_STYLE_TAGS, SETTING_TAGS,
)


def generate_caption(
    tags: list[str],
    performers: list[str] | None = None,
    studio: str | None = None,
    visual_notes: str | None = None,
) -> str:
    """Generate a natural language caption for a scene from its tags.

    Combines tag categories into a descriptive sentence suitable for
    CLIP contrastive training. Admin/workflow tags are excluded.
    Visual notes from frame analysis can override/augment the tag-derived text.

    Args:
        tags: List of Stash tag names for the scene.
        performers: Optional list of performer names.
        studio: Optional studio name.
        visual_notes: Optional free-text observations from visual frame analysis.

    Returns:
        Natural language caption string.
    """
    content_tags = {t for t in tags if t not in ADMIN_TAGS}

    body_types    = content_tags & BODY_TYPE_TAGS
    physical      = content_tags & PHYSICAL_ATTRIBUTE_TAGS
    acts          = content_tags & ACT_TAGS
    outcomes      = content_tags & OUTCOME_TAGS
    positions     = content_tags & POSITION_TAGS
    styles        = content_tags & CONTENT_STYLE_TAGS
    settings      = content_tags & SETTING_TAGS
    # Remaining unclassified tags
    other         = content_tags - body_types - physical - acts - outcomes - positions - styles - settings

    parts: list[str] = []

    # --- Lead with production style ---
    style_words: list[str] = []
    if "Non-Nude" in styles:
        style_words.append("non-nude")
    elif "Softcore" in styles:
        style_words.append("softcore")
    elif "Hardcore" in styles:
        style_words.append("hardcore")
    elif "Amateur" in styles or "Homemade" in styles:
        style_words.append("amateur")
    if "POV" in styles or "Male POV" in styles:
        style_words.append("POV")
    if "VR" in styles:
        style_words.append("VR")
    style_lead = " ".join(style_words) if style_words else "adult"

    # --- Performer description ---
    perf_parts: list[str] = []
    if body_types:
        perf_parts.append(_join(body_types))
    if "Solo" in styles:
        perf_parts.append("solo")

    # Physical attributes: tits first, then hair, then other
    phys_ordered: list[str] = []
    tit_tags = physical & {"Big Tits","Medium Tits","Small Tits","Natural Tits",
                           "Fake Tits","Perfect Tits","Saggy Tits"}
    if tit_tags:
        phys_ordered.append(_join(tit_tags).lower())
    hair_tags = physical & {"Blonde Hair", "Colored Hair", "Pigtails"}
    if hair_tags:
        phys_ordered.append(_join(hair_tags).lower())
    remainder_phys = physical - tit_tags - hair_tags - {"BBC"}
    if remainder_phys:
        phys_ordered.extend(t.lower() for t in sorted(remainder_phys))

    # Ethnicity: build from remaining other tags
    ethnicity_tags = other & {"Asian","Asian Woman","Black","White","Latina",
                              "Filipino","Japanese","Interracial","PAAG","PAWG"}
    if ethnicity_tags and not body_types:
        perf_parts.insert(0, _join(ethnicity_tags).lower())

    perf_desc = "a performer"
    if perf_parts:
        perf_desc = f"a {', '.join(perf_parts)} performer"
    if phys_ordered:
        perf_desc += f" with {', '.join(phys_ordered)}"

    # Override with named performer if known
    if performers:
        # Keep body description, just add name context
        first = performers[0]
        perf_desc = f"{first} ({', '.join(perf_parts) if perf_parts else 'performer'})"
        if phys_ordered:
            perf_desc += f" with {', '.join(phys_ordered)}"

    # --- Opening sentence ---
    parts.append(f"A {style_lead} scene featuring {perf_desc}.")

    # --- Acts + positions ---
    act_strs: list[str] = []
    if acts:
        primary_acts = acts - {"Oral Sex", "Couple Sex", "Outercourse"}
        if primary_acts:
            act_strs.append(_join(primary_acts).lower())
        elif acts:
            act_strs.append(_join(acts).lower())
    if positions:
        act_strs.append(f"{_join(positions).lower()} position")
    if outcomes:
        act_strs.append(f"ending with {_join(outcomes).lower()}")
    if act_strs:
        parts.append(f"The scene features {', '.join(act_strs)}.")

    # --- Setting ---
    setting_strs: list[str] = []
    if settings:
        setting_strs.append(_join(settings).lower())
    if studio:
        setting_strs.append(f"from {studio}")
    if setting_strs:
        parts.append(f"Shot {', '.join(setting_strs)}.")

    # --- Visual notes override/augment ---
    if visual_notes:
        parts.append(visual_notes.strip())

    return " ".join(parts)


def _join(tags: set[str], separator: str = ", ") -> str:
    """Join a set of tags into a readable string."""
    return separator.join(sorted(tags))
```

**Step 4: Run tests**

```bash
uv run pytest tests/tools/test_caption_generator.py -v
```
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add tools/dataset/caption_generator.py tests/tools/test_caption_generator.py
git commit -m "feat(dataset): add caption generator with tag-to-sentence mapping"
```

---

## Task 3: Dataset initializer

**Files:**
- Create: `tools/dataset/init_dataset.py`
- Test: `tests/tools/test_init_dataset.py`

**Step 1: Write test**

```python
# tests/tools/test_init_dataset.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from tools.dataset.init_dataset import (
    fetch_scenes, filter_scenes, compute_frame_paths, build_work_queue,
)

MOCK_SCENES = [
    {"id": "19", "tags": [
        {"name": "Embedded"}, {"name": "PAWG"}, {"name": "Big Ass"},
        {"name": "Small Tits"}, {"name": "Tan"}, {"name": "Adorable"},
    ], "performers": [{"name": "Mikaela Lafuente"}], "studio": None},
    {"id": "23", "tags": [{"name": "Embedded"}],
     "performers": [], "studio": None},
    {"id": "25", "tags": [
        {"name": "Embedded"}, {"name": "Medium Tits"}, {"name": "PAWG"},
        {"name": "Natural Tits"}, {"name": "Perfect Tits"}, {"name": "Adorable"},
    ], "performers": [], "studio": None},
]

def test_filter_scenes_removes_low_tag_scenes() -> None:
    filtered = filter_scenes(MOCK_SCENES)
    ids = [s["id"] for s in filtered]
    assert "19" in ids   # 5 content tags (excluding Embedded)
    assert "23" not in ids  # only Embedded = 0 content tags
    assert "25" in ids   # 5 content tags

def test_filter_scenes_counts_content_tags_only() -> None:
    # Scene 19 has 6 tags but Embedded is admin → 5 content tags, should pass
    filtered = filter_scenes(MOCK_SCENES)
    assert any(s["id"] == "19" for s in filtered)

def test_compute_frame_paths_returns_20_paths(tmp_path: Path) -> None:
    # Create mock frame files
    scene_dir = tmp_path / "scene_19"
    scene_dir.mkdir()
    for i in range(1, 101):
        (scene_dir / f"frame_{i:04d}.jpg").touch()

    paths = compute_frame_paths("19", frames_dir=tmp_path, n=20)
    assert len(paths) == 20
    # Should be evenly spaced, not random
    assert paths[0] != paths[-1]

def test_compute_frame_paths_fewer_than_n(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_99"
    scene_dir.mkdir()
    for i in range(1, 8):
        (scene_dir / f"frame_{i:04d}.jpg").touch()

    paths = compute_frame_paths("99", frames_dir=tmp_path, n=20)
    # Should return all available if fewer than n
    assert len(paths) == 7

def test_build_work_queue_structure() -> None:
    scenes = [s for s in MOCK_SCENES if s["id"] in {"19", "25"}]
    queue = build_work_queue(scenes)
    assert len(queue) == 2
    entry = queue[0]
    assert "scene_id" in entry
    assert "tags" in entry
    assert "performers" in entry
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/tools/test_init_dataset.py -v
```
Expected: ImportError

**Step 3: Write `tools/dataset/init_dataset.py`**

```python
"""Initialize the LoRA training dataset from the Stash library.

Usage:
    uv run python tools/dataset/init_dataset.py

Creates:
    assets/lora_dataset/
    assets/lora_dataset/images/         (empty, filled by process_batch)
    assets/lora_dataset/progress.json   (work queue + checkpoint)
    assets/lora_dataset/metadata.jsonl  (empty, filled by process_batch)
"""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

from tools.dataset.constants import (
    ADMIN_TAGS,
    DATASET_DIR,
    FRAMES_DIR,
    FRAMES_PER_SCENE,
    MIN_CONTENT_TAGS,
    STASH_GRAPHQL,
)


_QUERY = """
{
  allScenes {
    id
    tags { name }
    performers { name }
    studio { name }
  }
}
"""


def fetch_scenes(graphql_url: str = STASH_GRAPHQL) -> list[dict]:
    """Fetch all scenes from Stash GraphQL API."""
    resp = requests.post(graphql_url, json={"query": _QUERY}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]["allScenes"]


def filter_scenes(
    scenes: list[dict],
    min_content_tags: int = MIN_CONTENT_TAGS,
) -> list[dict]:
    """Return only scenes with at least min_content_tags non-admin tags."""
    result = []
    for scene in scenes:
        content_tags = [t["name"] for t in scene["tags"] if t["name"] not in ADMIN_TAGS]
        if len(content_tags) >= min_content_tags:
            result.append(scene)
    return result


def compute_frame_paths(
    scene_id: str,
    frames_dir: Path = FRAMES_DIR,
    n: int = FRAMES_PER_SCENE,
) -> list[Path]:
    """Return n evenly-spaced frame paths from a scene's embedded_frames dir.

    If fewer than n frames exist, returns all available frames.
    """
    scene_dir = frames_dir / f"scene_{scene_id}"
    if not scene_dir.exists():
        return []

    all_frames = sorted(scene_dir.glob("frame_*.jpg"))
    if not all_frames:
        return []

    if len(all_frames) <= n:
        return all_frames

    # Uniform sampling: pick n evenly-spaced indices
    step = (len(all_frames) - 1) / (n - 1)
    indices = [round(i * step) for i in range(n)]
    return [all_frames[i] for i in indices]


def build_work_queue(scenes: list[dict]) -> list[dict]:
    """Build the work queue entries for each scene."""
    return [
        {
            "scene_id": s["id"],
            "tags": [t["name"] for t in s["tags"] if t["name"] not in ADMIN_TAGS],
            "performers": [p["name"] for p in (s.get("performers") or [])],
            "studio": (s.get("studio") or {}).get("name") or None,
        }
        for s in scenes
    ]


def init_dataset(
    dataset_dir: Path = DATASET_DIR,
    frames_dir: Path = FRAMES_DIR,
    graphql_url: str = STASH_GRAPHQL,
) -> None:
    """Initialize the dataset directory and progress checkpoint."""
    print("Fetching scenes from Stash...")
    all_scenes = fetch_scenes(graphql_url)
    print(f"  Total scenes: {len(all_scenes)}")

    selected = filter_scenes(all_scenes)
    print(f"  Scenes with {MIN_CONTENT_TAGS}+ content tags: {len(selected)}")

    # Compute frame paths per scene
    print("Computing frame paths...")
    work_queue = build_work_queue(selected)
    missing_frames = []
    valid_queue = []
    for entry in work_queue:
        paths = compute_frame_paths(entry["scene_id"], frames_dir)
        if not paths:
            missing_frames.append(entry["scene_id"])
            continue
        entry["frame_paths"] = [str(p) for p in paths]
        # Analysis frames: 1/3 and 1/2 of timeline for visual inspection
        n = len(paths)
        entry["analysis_frames"] = [
            str(paths[n // 3]),
            str(paths[n // 2]),
        ]
        valid_queue.append(entry)

    if missing_frames:
        print(f"  WARNING: {len(missing_frames)} scenes have no embedded frames — skipped")

    # Create directory structure
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images").mkdir(exist_ok=True)

    # Write progress checkpoint
    progress = {
        "total_scenes": len(valid_queue),
        "completed": [],
        "pending": [e["scene_id"] for e in valid_queue],
        "last_updated": datetime.now(UTC).isoformat(),
        "pairs_written": 0,
        "sessions": 0,
        "work_queue": {e["scene_id"]: e for e in valid_queue},
    }
    progress_path = dataset_dir / "progress.json"
    progress_path.write_text(json.dumps(progress, indent=2))
    print(f"  Progress checkpoint written: {progress_path}")

    # Create empty metadata file
    (dataset_dir / "metadata.jsonl").touch()

    print(f"\nDataset initialized:")
    print(f"  Directory: {dataset_dir}")
    print(f"  Scenes queued: {len(valid_queue)}")
    print(f"  Expected pairs: {len(valid_queue) * FRAMES_PER_SCENE:,}")
    print(f"\nNext step: run process_batch.py to start processing scenes.")


if __name__ == "__main__":
    init_dataset()
```

**Step 4: Run tests**

```bash
uv run pytest tests/tools/test_init_dataset.py -v
```
Expected: 5 tests PASS

**Step 5: Run the initializer against real Stash**

```bash
uv run python tools/dataset/init_dataset.py
```
Expected output:
```
Fetching scenes from Stash...
  Total scenes: 12006
  Scenes with 5+ content tags: 3903
Computing frame paths...
Dataset initialized:
  Directory: .../assets/lora_dataset
  Scenes queued: ~3900
  Expected pairs: ~78,000
```

**Step 6: Commit**

```bash
git add tools/dataset/init_dataset.py tests/tools/test_init_dataset.py
git commit -m "feat(dataset): add init_dataset script — selects 3903 scenes, writes progress.json"
```

---

## Task 4: Frame copy and caption write utilities

**Files:**
- Create: `tools/dataset/io_utils.py`
- Test: `tests/tools/test_io_utils.py`

**Step 1: Write test**

```python
# tests/tools/test_io_utils.py
import json
from pathlib import Path
import pytest
from tools.dataset.io_utils import (
    copy_frames_to_dataset, write_caption_files, append_metadata_record,
    dataset_image_name,
)

def test_dataset_image_name() -> None:
    p = Path("/some/dir/scene_19/frame_0001.jpg")
    assert dataset_image_name("19", p) == "s19_f0001.jpg"

def test_copy_frames_creates_files(tmp_path: Path) -> None:
    # Source frames
    src = tmp_path / "frames" / "scene_42"
    src.mkdir(parents=True)
    frame = src / "frame_0010.jpg"
    frame.write_bytes(b"fake jpeg data")

    dest = tmp_path / "images"
    dest.mkdir()

    copied = copy_frames_to_dataset("42", [frame], dest)
    assert len(copied) == 1
    dest_file = dest / "s42_f0010.jpg"
    assert dest_file.exists()
    assert dest_file.read_bytes() == b"fake jpeg data"
    # Source must NOT be deleted
    assert frame.exists()

def test_write_caption_files(tmp_path: Path) -> None:
    dest = tmp_path / "images"
    dest.mkdir()
    image_names = ["s19_f0001.jpg", "s19_f0050.jpg"]
    for name in image_names:
        (dest / name).write_bytes(b"img")

    caption = "An amateur scene featuring a PAWG performer with big ass."
    write_caption_files(image_names, caption, dest)

    for name in image_names:
        txt = dest / name.replace(".jpg", ".txt")
        assert txt.exists()
        assert txt.read_text() == caption

def test_append_metadata_record(tmp_path: Path) -> None:
    jsonl = tmp_path / "metadata.jsonl"
    jsonl.touch()
    append_metadata_record(
        jsonl_path=jsonl,
        scene_id="19",
        tags=["PAWG", "Big Ass"],
        caption="A non-nude scene.",
        visual_notes="Performer is dancing in lingerie.",
        missing_tags=["Striptease"],
        image_names=["s19_f0001.jpg"],
    )
    lines = jsonl.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["scene_id"] == "19"
    assert record["caption"] == "A non-nude scene."
    assert "Striptease" in record["missing_tags"]
    assert record["visual_notes"] == "Performer is dancing in lingerie."
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/tools/test_io_utils.py -v
```
Expected: ImportError

**Step 3: Write `tools/dataset/io_utils.py`**

```python
"""File I/O utilities for dataset construction."""
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def dataset_image_name(scene_id: str, frame_path: Path) -> str:
    """Convert a source frame path to a flat dataset image filename.

    Example: scene_19/frame_0001.jpg → s19_f0001.jpg
    """
    stem = frame_path.stem  # "frame_0001"
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
    """Write a .txt caption file alongside each image file.

    The caption is the same for all frames from the same scene.
    """
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
```

**Step 4: Run tests**

```bash
uv run pytest tests/tools/test_io_utils.py -v
```
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add tools/dataset/io_utils.py tests/tools/test_io_utils.py
git commit -m "feat(dataset): add frame copy and caption write utilities"
```

---

## Task 5: Progress tracker

**Files:**
- Create: `tools/dataset/progress.py`
- Test: `tests/tools/test_progress.py`

**Step 1: Write test**

```python
# tests/tools/test_progress.py
import json
from pathlib import Path
from tools.dataset.progress import (
    load_progress, save_progress, mark_scene_complete, get_next_batch,
)

def _make_progress(tmp_path: Path, completed: list[str], pending: list[str]) -> Path:
    p = tmp_path / "progress.json"
    data = {
        "total_scenes": len(completed) + len(pending),
        "completed": completed,
        "pending": pending,
        "last_updated": "2026-02-18T00:00:00+00:00",
        "pairs_written": len(completed) * 20,
        "sessions": 1,
        "work_queue": {sid: {"scene_id": sid, "tags": [], "frame_paths": []} for sid in completed + pending},
    }
    p.write_text(json.dumps(data))
    return p

def test_load_progress(tmp_path: Path) -> None:
    path = _make_progress(tmp_path, completed=["1"], pending=["2", "3"])
    prog = load_progress(path)
    assert prog["total_scenes"] == 3
    assert "1" in prog["completed"]
    assert "2" in prog["pending"]

def test_get_next_batch(tmp_path: Path) -> None:
    path = _make_progress(tmp_path, completed=[], pending=["1","2","3","4","5"])
    prog = load_progress(path)
    batch = get_next_batch(prog, batch_size=3)
    assert len(batch) == 3
    assert batch[0]["scene_id"] == "1"

def test_mark_scene_complete(tmp_path: Path) -> None:
    path = _make_progress(tmp_path, completed=[], pending=["10","20"])
    prog = load_progress(path)
    mark_scene_complete(prog, scene_id="10", pairs_added=20)
    assert "10" in prog["completed"]
    assert "10" not in prog["pending"]
    assert prog["pairs_written"] == 20

def test_save_and_reload(tmp_path: Path) -> None:
    path = _make_progress(tmp_path, completed=[], pending=["1"])
    prog = load_progress(path)
    mark_scene_complete(prog, "1", 20)
    save_progress(prog, path)
    reloaded = load_progress(path)
    assert "1" in reloaded["completed"]
    assert reloaded["pairs_written"] == 20
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/tools/test_progress.py -v
```
Expected: ImportError

**Step 3: Write `tools/dataset/progress.py`**

```python
"""Progress checkpoint management for multi-session dataset construction."""
import json
from datetime import UTC, datetime
from pathlib import Path


def load_progress(progress_path: Path) -> dict:
    """Load the progress checkpoint from disk."""
    return json.loads(progress_path.read_text(encoding="utf-8"))


def save_progress(progress: dict, progress_path: Path) -> None:
    """Persist the progress checkpoint to disk."""
    progress["last_updated"] = datetime.now(UTC).isoformat()
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def get_next_batch(progress: dict, batch_size: int = 150) -> list[dict]:
    """Return the next batch of unprocessed scene work queue entries."""
    pending_ids = progress["pending"][:batch_size]
    return [progress["work_queue"][sid] for sid in pending_ids]


def mark_scene_complete(progress: dict, scene_id: str, pairs_added: int) -> None:
    """Mark a scene as done in the in-memory progress dict."""
    if scene_id in progress["pending"]:
        progress["pending"].remove(scene_id)
    if scene_id not in progress["completed"]:
        progress["completed"].append(scene_id)
    progress["pairs_written"] += pairs_added


def session_summary(progress: dict) -> str:
    """Return a human-readable summary of current progress."""
    total = progress["total_scenes"]
    done = len(progress["completed"])
    pending = len(progress["pending"])
    pairs = progress["pairs_written"]
    pct = (done / total * 100) if total else 0
    sessions_est = max(1, pending // 150)
    return (
        f"Progress: {done}/{total} scenes ({pct:.1f}%) | "
        f"{pairs:,} pairs written | "
        f"~{sessions_est} sessions remaining"
    )
```

**Step 4: Run tests**

```bash
uv run pytest tests/tools/test_progress.py -v
```
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add tools/dataset/progress.py tests/tools/test_progress.py
git commit -m "feat(dataset): add progress checkpoint manager for multi-session resume"
```

---

## Task 6: Process batch script (interactive analysis loop)

This is the core session script. It prints each scene's metadata and analysis frame paths so Claude can read the images and provide enhanced captions.

**Files:**
- Create: `tools/dataset/process_batch.py`

**Step 1: Write `tools/dataset/process_batch.py`**

```python
"""Process a batch of scenes: copy frames, generate captions, update progress.

This script is designed to be run interactively each session.
For each scene in the batch, it prints the analysis frame paths so Claude
can read the images and provide augmented captions.

Usage:
    uv run python tools/dataset/process_batch.py [--batch-size N] [--dry-run]

After running this script, Claude reads the images listed for each scene
and calls write_scene() with the enhanced caption and any missing tags observed.
"""
import argparse
import json
from pathlib import Path

from tools.dataset.caption_generator import generate_caption
from tools.dataset.constants import DATASET_DIR, FRAMES_PER_SCENE
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

        # Generate baseline tag-derived caption
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
        # Claude reads the images above and calls process_scene() with
        # the enhanced caption for this scene.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a batch of scenes for the LoRA dataset.")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_batch(batch_size=args.batch_size, dry_run=args.dry_run)
```

**Step 2: Test-run with dry-run on first 5 scenes**

```bash
uv run python tools/dataset/process_batch.py --batch-size 5 --dry-run
```

Expected: prints 5 scenes with their tags, baseline captions, and analysis frame paths.

**Step 3: Commit**

```bash
git add tools/dataset/process_batch.py
git commit -m "feat(dataset): add process_batch script — interactive session loop for visual analysis"
```

---

## Task 7: Dataset finalizer

**Files:**
- Create: `tools/dataset/finalize_dataset.py`
- Test: `tests/tools/test_finalize_dataset.py`

**Step 1: Write test**

```python
# tests/tools/test_finalize_dataset.py
import json
import csv
from pathlib import Path
from tools.dataset.finalize_dataset import (
    load_metadata, split_train_val, write_csv, generate_missing_tags_report,
)

def _write_metadata(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def test_load_metadata(tmp_path: Path) -> None:
    meta = tmp_path / "metadata.jsonl"
    _write_metadata(meta, [
        {"scene_id": "1", "image_names": ["s1_f0001.jpg", "s1_f0050.jpg"],
         "caption": "A scene.", "tags": ["PAWG"], "missing_tags": []},
    ])
    records = load_metadata(meta)
    assert len(records) == 1
    assert records[0]["scene_id"] == "1"

def test_split_train_val_is_scene_level() -> None:
    records = [
        {"scene_id": str(i), "image_names": [f"s{i}_f001.jpg", f"s{i}_f002.jpg"],
         "caption": f"Caption {i}.", "tags": [], "missing_tags": []}
        for i in range(10)
    ]
    train, val = split_train_val(records, val_fraction=0.1)
    # Scene-level split: no scene_id appears in both
    train_ids = {r["scene_id"] for r in train}
    val_ids = {r["scene_id"] for r in val}
    assert not (train_ids & val_ids)
    # Roughly 90/10 split at scene level
    assert len(val_ids) == 1
    assert len(train_ids) == 9

def test_write_csv(tmp_path: Path) -> None:
    records = [
        {"scene_id": "1", "image_names": ["s1_f0001.jpg"],
         "caption": "A scene.", "tags": [], "missing_tags": []},
    ]
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    csv_path = tmp_path / "train.csv"
    write_csv(records, csv_path, images_dir)
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["caption"] == "A scene."
    assert "s1_f0001.jpg" in rows[0]["filepath"]

def test_missing_tags_report(tmp_path: Path) -> None:
    records = [
        {"scene_id": "1", "missing_tags": ["Blowjob", "POV"], "tags": []},
        {"scene_id": "2", "missing_tags": ["Blowjob"], "tags": []},
        {"scene_id": "3", "missing_tags": [], "tags": []},
    ]
    report = generate_missing_tags_report(records)
    assert "Blowjob" in report
    assert "2" in report  # count of scenes
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/tools/test_finalize_dataset.py -v
```
Expected: ImportError

**Step 3: Write `tools/dataset/finalize_dataset.py`**

```python
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

## Caption Format

Natural language sentences derived from Stash tags and visual frame analysis
by Claude (Anthropic). Each caption describes scene-level content:
production style, performer attributes, acts/positions observed, and setting.

## Training Command (LoRA on RTX 5090)

```bash
cd /path/to/clipora
python train.py --config train_config.yml
```

Where `train_config.yml` points `train_dataset` to `train.csv`.
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
```

**Step 4: Run tests**

```bash
uv run pytest tests/tools/test_finalize_dataset.py -v
```
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add tools/dataset/finalize_dataset.py tests/tools/test_finalize_dataset.py
git commit -m "feat(dataset): add finalize script — train/val CSVs, missing tags report, README"
```

---

## Task 8: Run the full pipeline — Phase 1

Run the automated infrastructure steps end-to-end.

**Step 1: Initialize the dataset**

```bash
uv run python tools/dataset/init_dataset.py
```

Expected:
```
Fetching scenes from Stash...
  Total scenes: 12006
  Scenes with 5+ content tags: ~3903
Computing frame paths...
Dataset initialized:
  Directory: .../assets/lora_dataset
  Scenes queued: ~3900
  Expected pairs: ~78,000
```

**Step 2: Verify structure**

```bash
ls ~/.stash/plugins/stash-copilot/assets/lora_dataset/
# Expected: images/  metadata.jsonl  progress.json
python3 -c "
import json
p = json.load(open('~/.stash/plugins/stash-copilot/assets/lora_dataset/progress.json'))
print('Pending:', len(p['pending']))
print('Completed:', len(p['completed']))
print('Sample entry:', list(p['work_queue'].values())[0])
"
```

**Step 3: Dry-run first batch**

```bash
uv run python tools/dataset/process_batch.py --batch-size 5 --dry-run
```

Expected: prints 5 scenes with their tags, baseline captions, and analysis frame paths.

**Step 4: Commit infrastructure output**

```bash
# Don't commit the dataset itself (too large) — only the progress file
echo "assets/lora_dataset/images/" >> .gitignore
echo "assets/lora_dataset/metadata.jsonl" >> .gitignore
echo "assets/lora_dataset/train.csv" >> .gitignore
echo "assets/lora_dataset/val.csv" >> .gitignore
# DO commit progress.json so sessions can be resumed
git add .gitignore assets/lora_dataset/progress.json assets/lora_dataset/metadata.jsonl
git commit -m "feat(dataset): initialize dataset — 3903 scenes queued for visual analysis"
```

---

## Task 9: Interactive visual analysis sessions (multi-session)

**This task repeats across ~26 sessions. Each session processes ~150 scenes.**

**Each session workflow:**

**Step 1: Check progress**

```bash
python3 -c "
import json
p = json.load(open('~/.stash/plugins/stash-copilot/assets/lora_dataset/progress.json'))
done = len(p['completed'])
total = p['total_scenes']
print(f'Progress: {done}/{total} ({done/total*100:.1f}%)')
print(f'Pairs written: {p[\"pairs_written\"]:,}')
"
```

**Step 2: Print next batch**

```bash
uv run python tools/dataset/process_batch.py --batch-size 150
```

**Step 3: For each scene printed, Claude:**
1. Reads the two analysis frames listed using the Read tool
2. Observes: act type, position, clothing state, setting, camera angle, visible body attributes
3. Compares to the baseline caption and existing tags
4. Calls `process_scene()` with the enhanced caption, visual notes, and missing tags
5. Calls `mark_scene_complete()` and `save_progress()`

**Step 4: After processing each scene, update progress**

```python
# (Executed by Claude inline, not as a CLI script)
from tools.dataset.process_batch import process_scene
from tools.dataset.progress import load_progress, mark_scene_complete, save_progress
from pathlib import Path

PROGRESS_PATH = Path("~/.stash/plugins/stash-copilot/assets/lora_dataset/progress.json")
progress = load_progress(PROGRESS_PATH)
entry = progress["work_queue"]["<SCENE_ID>"]

image_names = process_scene(
    entry=entry,
    caption="<ENHANCED CAPTION FROM VISUAL ANALYSIS>",
    visual_notes="<RAW OBSERVATIONS>",
    missing_tags=["<TAG1>", "<TAG2>"],  # observed but not in Stash
)

mark_scene_complete(progress, "<SCENE_ID>", pairs_added=len(image_names))
save_progress(progress, PROGRESS_PATH)
```

**Step 5: End-of-session commit**

```bash
git add assets/lora_dataset/progress.json assets/lora_dataset/metadata.jsonl
git commit -m "feat(dataset): process scenes <FIRST_ID>–<LAST_ID> (session N)"
```

---

## Task 10: Finalize dataset (after all sessions complete)

**Step 1: Run finalization**

```bash
uv run python tools/dataset/finalize_dataset.py
```

Expected:
```
Loaded 3903 scene records.
Dataset finalized:
  Train pairs: ~70,200
  Val pairs:   ~7,800
  Total pairs: ~78,000
  Missing tags report: .../missing_tags_report.md
```

**Step 2: Review missing tags report**

```bash
cat ~/.stash/plugins/stash-copilot/assets/lora_dataset/missing_tags_report.md | head -30
```

This report drives taxonomy updates in Stash — add any frequently observed missing tags.

**Step 3: Verify CSV format for training**

```bash
head -3 ~/.stash/plugins/stash-copilot/assets/lora_dataset/train.csv
# Expected: filepath,caption header + rows
python3 -c "
import csv
rows = list(csv.DictReader(open('~/.stash/plugins/stash-copilot/assets/lora_dataset/train.csv')))
print(f'Train rows: {len(rows):,}')
print(f'Sample: {rows[0]}')
"
```

**Step 4: Final commit**

```bash
git add assets/lora_dataset/progress.json assets/lora_dataset/missing_tags_report.md assets/lora_dataset/README.md
git commit -m "feat(dataset): finalize LoRA training dataset — ~78k pairs, missing tags report"
```

---

## Run Order Summary

```bash
# One-time setup
uv run python tools/dataset/init_dataset.py

# Per session (repeat ~26 times)
uv run python tools/dataset/process_batch.py --batch-size 150
# → Claude reads frames, calls process_scene() per scene, saves progress

# After all sessions
uv run python tools/dataset/finalize_dataset.py
```

## All Tests

```bash
uv run pytest tests/tools/ -v
```
Expected: 16 tests PASS across 4 test files.
