"""Tests for clip_dataset.py."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal CSV with known scene IDs."""
    csv_path = tmp_path / "train.csv"
    rows = [
        # Scene 1: 3 frames
        (str(tmp_path / "s1_f001.jpg"), "caption for scene 1 frame 1"),
        (str(tmp_path / "s1_f002.jpg"), "caption for scene 1 frame 2"),
        (str(tmp_path / "s1_f003.jpg"), "caption for scene 1 frame 3"),
        # Scene 2: 2 frames
        (str(tmp_path / "s2_f001.jpg"), "caption for scene 2 frame 1"),
        (str(tmp_path / "s2_f002.jpg"), "caption for scene 2 frame 2"),
        # Scene 3: 1 frame
        (str(tmp_path / "s3_f001.jpg"), "caption for scene 3 frame 1"),
    ]
    # Create dummy image files (1x1 white pixel JPEG)
    from PIL import Image
    for filepath, _ in rows:
        img = Image.new("RGB", (1, 1), "white")
        img.save(filepath, "JPEG")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "caption"])
        writer.writerows(rows)
    return csv_path


def test_scene_level_split_no_leakage(sample_csv: Path) -> None:
    """Train and val sets must not share scene IDs."""
    from tools.training.clip_dataset import scene_level_split

    train_rows, val_rows = scene_level_split(sample_csv, val_fraction=0.34, seed=42)

    train_scenes = {r[0].split("/")[-1].split("_")[0] for r in train_rows}
    val_scenes = {r[0].split("/")[-1].split("_")[0] for r in val_rows}

    assert len(train_scenes & val_scenes) == 0, "Scene leakage between train/val"
    assert len(train_rows) + len(val_rows) == 6
    assert len(val_rows) >= 1  # At least one scene in val


def test_scene_level_split_deterministic(sample_csv: Path) -> None:
    """Same seed produces same split."""
    from tools.training.clip_dataset import scene_level_split

    split_a = scene_level_split(sample_csv, val_fraction=0.34, seed=42)
    split_b = scene_level_split(sample_csv, val_fraction=0.34, seed=42)

    assert [r[0] for r in split_a[0]] == [r[0] for r in split_b[0]]
    assert [r[0] for r in split_a[1]] == [r[0] for r in split_b[1]]


def test_dataloader_yields_tensors(sample_csv: Path) -> None:
    """DataLoader should yield (image_batch, text_batch) tensors."""
    from tools.training.clip_dataset import CLIPDataset, scene_level_split
    import open_clip
    import torch

    _, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32")  # Small model for test
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    train_rows, _ = scene_level_split(sample_csv, val_fraction=0.34, seed=42)
    dataset = CLIPDataset(train_rows, preprocess, tokenizer)

    assert len(dataset) == len(train_rows)
    img, txt = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(txt, torch.Tensor)
