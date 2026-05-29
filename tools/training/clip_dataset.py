"""CLIP dataset with scene-level train/val splitting.

Provides a PyTorch Dataset for CLIP fine-tuning that loads image-caption pairs
from a CSV file, and a scene-level splitting function that ensures no scene
appears in both train and validation sets (preventing data leakage).
"""
from __future__ import annotations

import csv
import logging
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def scene_level_split(
    csv_path: Path | str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split a CSV dataset by scene ID so no scene leaks between train/val.

    Args:
        csv_path: Path to CSV with columns ``filepath,caption``.
        val_fraction: Fraction of *scenes* (not rows) to hold out for validation.
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of ``(train_rows, val_rows)`` where each row is
        ``(filepath, caption)``.
    """
    csv_path = Path(csv_path)

    # Read all rows from the CSV
    rows: list[tuple[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            rows.append((row[0], row[1]))

    # Group rows by scene ID extracted from filename
    scene_to_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for filepath, caption in rows:
        filename = Path(filepath).name
        match = re.match(r"s(\d+)_f", filename)
        if match:
            scene_id = match.group(1)
        else:
            # Fallback: use the full filename as its own scene ID
            scene_id = filename
        scene_to_rows[scene_id].append((filepath, caption))

    # Shuffle scene IDs deterministically
    scene_ids = sorted(scene_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(scene_ids)

    # Split: first N scenes go to val, rest to train
    n_val_scenes = max(1, math.floor(len(scene_ids) * val_fraction))
    val_scene_ids = set(scene_ids[:n_val_scenes])

    train_rows: list[tuple[str, str]] = []
    val_rows: list[tuple[str, str]] = []
    for scene_id in scene_ids:
        if scene_id in val_scene_ids:
            val_rows.extend(scene_to_rows[scene_id])
        else:
            train_rows.extend(scene_to_rows[scene_id])

    return train_rows, val_rows


class CLIPDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for CLIP image-caption pairs.

    Args:
        rows: List of ``(filepath, caption)`` tuples.
        transform: Image preprocessing callable (from ``open_clip.create_model_and_transforms``).
        tokenizer: Text tokenizer callable (from ``open_clip.get_tokenizer``).
    """

    def __init__(
        self,
        rows: list[tuple[str, str]],
        transform: Callable[..., torch.Tensor],
        tokenizer: Callable[..., torch.Tensor],
    ) -> None:
        self._rows = rows
        self._transform = transform
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filepath, caption = self._rows[index]

        try:
            image = Image.open(filepath).convert("RGB")
            image_tensor: torch.Tensor = self._transform(image)
        except Exception:
            logger.warning(
                "Corrupt or unreadable image at index %d (%s), "
                "returning a random other sample",
                index,
                filepath,
            )
            # Pick a random other index to substitute
            alt_index = random.randint(0, len(self._rows) - 1)
            if alt_index == index and len(self._rows) > 1:
                alt_index = (index + 1) % len(self._rows)
            return self[alt_index]

        # Tokenize caption — squeeze batch dim if present
        text_tensor: torch.Tensor = self._tokenizer(caption)
        if text_tensor.dim() > 1:
            text_tensor = text_tensor.squeeze(0)

        return image_tensor, text_tensor


def create_dataloaders(
    csv_path: Path | str,
    model_name: str = "ViT-B-32",
    pretrained: str = "",
    val_fraction: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader[Any], DataLoader[Any], list[tuple[str, str]], list[tuple[str, str]]]:
    """Convenience function to create train/val DataLoaders from a CSV.

    Args:
        csv_path: Path to CSV with columns ``filepath,caption``.
        model_name: OpenCLIP model architecture name.
        pretrained: Pretrained weights tag. Use ``""`` to skip downloading weights.
        val_fraction: Fraction of scenes held out for validation.
        batch_size: Batch size for both loaders.
        num_workers: Number of DataLoader worker processes.
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple of ``(train_loader, val_loader, train_rows, val_rows)``.
    """
    # Load only the transforms (pretrained="" skips weight download)
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # Split dataset by scene
    train_rows, val_rows = scene_level_split(csv_path, val_fraction=val_fraction, seed=seed)

    # Build datasets
    train_dataset = CLIPDataset(train_rows, preprocess_train, tokenizer)
    val_dataset = CLIPDataset(val_rows, preprocess_val, tokenizer)

    # Build data loaders
    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_rows, val_rows
