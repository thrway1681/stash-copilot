#!/usr/bin/env python3
"""Evaluate LoRA checkpoints: R@K metrics and nearest-neighbor galleries.

Computes retrieval R@K (R@1, R@5, R@10) for both image-to-text and text-to-image
directions, contrastive loss on the validation set, and generates nearest-neighbor
gallery images comparing base model vs LoRA-adapted model.

Usage:
    # Evaluate a specific checkpoint
    uv run python tools/training/eval_lora.py --checkpoint path/to/epoch_001.pt --run my-run

    # Watch a run directory for new checkpoints and evaluate automatically
    uv run python tools/training/eval_lora.py --watch --run my-run

    # Evaluate with custom settings
    uv run python tools/training/eval_lora.py --checkpoint best.pt --run my-run \\
        --max-pairs 1000 --n-probes 100
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageDraw, ImageFont

from tools.training.clip_dataset import scene_level_split

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path("assets/lora_training/runs")


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute retrieval Recall@K for image-to-text and text-to-image.

    Both inputs must be L2-normalised tensors of shape ``(N, D)`` where
    each row *i* in ``image_features`` is the ground-truth match for row *i*
    in ``text_features``.

    Args:
        image_features: L2-normalised image embeddings ``(N, D)``.
        text_features: L2-normalised text embeddings ``(N, D)``.
        ks: Tuple of K values to compute recall at.

    Returns:
        Dictionary with keys like ``"i2t_r@1"``, ``"t2i_r@5"`` etc.
        Values are fractions in ``[0, 1]``, not percentages.
    """
    n = image_features.size(0)
    # (N, N) similarity matrix — each row i contains sim(image_i, text_j)
    similarity: torch.Tensor = image_features @ text_features.T

    metrics: dict[str, float] = {}
    ground_truth = torch.arange(n, device=similarity.device)

    for k in ks:
        if k > n:
            continue  # skip R@K when fewer than K samples
        # Image-to-text: for each image, do the top-k retrieved texts include
        # the correct one?
        _, i2t_topk = similarity.topk(k, dim=1)  # (N, K)
        i2t_correct = (i2t_topk == ground_truth.unsqueeze(1)).any(dim=1)
        metrics[f"i2t_r@{k}"] = float(i2t_correct.float().mean().item())

        # Text-to-image: for each text (column), do the top-k retrieved
        # images include the correct one?
        _, t2i_topk = similarity.T.topk(k, dim=1)  # (N, K)
        t2i_correct = (t2i_topk == ground_truth.unsqueeze(1)).any(dim=1)
        metrics[f"t2i_r@{k}"] = float(t2i_correct.float().mean().item())

    return metrics


# ---------------------------------------------------------------------------
# Dataset encoding
# ---------------------------------------------------------------------------


class _EvalDataset(Dataset[tuple[torch.Tensor, str]]):
    """Lightweight dataset for eval — returns preprocessed image + raw caption."""

    def __init__(
        self,
        rows: list[tuple[str, str]],
        preprocess: Any,
    ) -> None:
        self._rows = rows
        self._preprocess = preprocess

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        filepath, caption = self._rows[index]
        img = Image.open(filepath).convert("RGB")
        return self._preprocess(img), caption


def _collate_eval(
    batch: list[tuple[torch.Tensor, str]],
) -> tuple[torch.Tensor, list[str]]:
    """Collate preprocessed images into a stacked tensor + caption list."""
    images, captions = zip(*batch)
    return torch.stack(images), list(captions)


def encode_dataset(
    model: torch.nn.Module,
    rows: list[tuple[str, str]],
    preprocess: Any,
    tokenizer: Any,
    device: torch.device | str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all images and texts from *rows*, returning L2-normalised features.

    Uses a DataLoader with multiple workers and pinned memory to overlap
    I/O with GPU compute, and fp16 autocast for faster inference.

    Args:
        model: An OpenCLIP model (possibly wrapped with PEFT).
        rows: List of ``(filepath, caption)`` tuples.
        preprocess: Image preprocessing transform.
        tokenizer: Text tokeniser callable.
        device: Target device (``"cuda"`` or ``"cpu"``).
        batch_size: Encoding batch size.
        num_workers: DataLoader worker processes for parallel I/O.

    Returns:
        Tuple ``(image_features, text_features)`` each of shape ``(N, D)``,
        L2-normalised along the feature dimension.
    """
    model.eval()
    use_amp = str(device) != "cpu" and torch.cuda.is_available()

    dataset = _EvalDataset(rows, preprocess)
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_amp,
        collate_fn=_collate_eval,
    )

    all_image_feats: list[torch.Tensor] = []
    all_text_feats: list[torch.Tensor] = []

    with torch.no_grad():
        for image_batch, captions in loader:
            image_batch = image_batch.to(device, non_blocking=True)
            text_tokens = tokenizer(captions).to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                img_feats: torch.Tensor = model.encode_image(image_batch)
                txt_feats: torch.Tensor = model.encode_text(text_tokens)

            all_image_feats.append(F.normalize(img_feats, dim=-1).cpu().float())
            all_text_feats.append(F.normalize(txt_feats, dim=-1).cpu().float())
            del img_feats, txt_feats

    image_features = torch.cat(all_image_feats, dim=0)
    text_features = torch.cat(all_text_feats, dim=0)
    return image_features, text_features


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------


def _contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float = 100.0,
) -> float:
    """Compute symmetric CLIP contrastive (InfoNCE) loss.

    Args:
        image_features: L2-normalised ``(N, D)`` tensor.
        text_features: L2-normalised ``(N, D)`` tensor.
        logit_scale: Scalar to multiply the cosine similarity matrix.

    Returns:
        Average of image-to-text and text-to-image cross-entropy losses.
    """
    n = image_features.size(0)
    logits = image_features @ text_features.T * logit_scale  # (N, N)
    labels = torch.arange(n, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return float(((loss_i2t + loss_t2i) / 2.0).item())


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------


def compute_baseline(
    val_rows: list[tuple[str, str]],
    model_name: str,
    pretrained: str,
    max_eval_pairs: int = 500,
    device: str = "cuda",
) -> dict[str, Any]:
    """Compute retrieval metrics and loss for the unmodified base model.

    The result is cached to ``eval/baseline.json`` by the caller so this
    only needs to run once per training run.

    Args:
        val_rows: Full validation set rows.
        model_name: OpenCLIP model architecture name.
        pretrained: Pretrained weights tag.
        max_eval_pairs: Maximum number of pairs to evaluate.
        device: Target device.

    Returns:
        Dict with keys ``"metrics"`` (R@K values), ``"loss"``, and
        ``"n_pairs"`` (number of pairs evaluated).
    """
    logger.info("Computing baseline metrics for %s / %s ...", model_name, pretrained)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Deterministic subsample
    subset = _subsample_rows(val_rows, max_eval_pairs, seed=42)

    image_feats, text_feats = encode_dataset(model, subset, preprocess, tokenizer, device)
    metrics = compute_retrieval_metrics(image_feats, text_feats)

    # Extract logit_scale from model if available
    logit_scale: float = 100.0
    if hasattr(model, "logit_scale"):
        logit_scale = float(model.logit_scale.exp().item())

    loss = _contrastive_loss(image_feats, text_feats, logit_scale=logit_scale)

    result: dict[str, Any] = {
        "metrics": metrics,
        "loss": loss,
        "n_pairs": len(subset),
        "model_name": model_name,
        "pretrained": pretrained,
    }

    logger.info(
        "Baseline — loss=%.4f  i2t_r@1=%.3f  i2t_r@5=%.3f  t2i_r@1=%.3f  t2i_r@5=%.3f",
        loss,
        metrics["i2t_r@1"],
        metrics["i2t_r@5"],
        metrics["t2i_r@1"],
        metrics["t2i_r@5"],
    )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: Path,
    val_rows: list[tuple[str, str]],
    model_name: str,
    pretrained: str,
    base_metrics: dict[str, Any] | None = None,
    max_eval_pairs: int = 500,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate a LoRA checkpoint: R@K metrics, loss, and deltas vs baseline.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        val_rows: Full validation set rows.
        model_name: OpenCLIP model architecture name.
        pretrained: Pretrained weights tag.
        base_metrics: Baseline metrics dict (from :func:`compute_baseline`).
            If provided, deltas are computed.
        max_eval_pairs: Maximum number of val pairs to evaluate.
        device: Target device.

    Returns:
        Dict with ``"metrics"``, ``"loss"``, ``"deltas"`` (if baseline
        provided), ``"n_pairs"``, and ``"checkpoint"``.
    """
    logger.info("Evaluating checkpoint: %s", checkpoint_path)

    ckpt: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config: dict[str, Any] = ckpt.get("config", {})

    # --- Reconstruct model with LoRA ---
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Build LoRA config from checkpoint metadata
    lora_rank: int = ckpt_config.get("lora_rank", 8)
    lora_alpha: int = ckpt_config.get("lora_alpha", 16)
    lora_dropout: float = ckpt_config.get("lora_dropout", 0.05)

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["c_fc", "c_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.base_model.model.logit_scale.requires_grad_(True)

    # Load LoRA weights — handle both full state_dict and adapter-only formats
    lora_state = ckpt["model_state_dict"]
    ls_key = "base_model.model.logit_scale"
    if ls_key in lora_state:
        model.base_model.model.logit_scale.data.copy_(lora_state.pop(ls_key))
    try:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, lora_state)
    except Exception:
        # Fallback: try loading as full state dict (legacy checkpoints)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    model = model.to(device)

    tokenizer = open_clip.get_tokenizer(model_name)

    # Deterministic subsample (same seed as baseline for consistency)
    subset = _subsample_rows(val_rows, max_eval_pairs, seed=42)

    image_feats, text_feats = encode_dataset(model, subset, preprocess, tokenizer, device)
    metrics = compute_retrieval_metrics(image_feats, text_feats)

    # Extract logit_scale — for PEFT-wrapped models, dig into base model
    logit_scale: float = 100.0
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        inner = model.base_model.model
        if hasattr(inner, "logit_scale"):
            logit_scale = float(inner.logit_scale.exp().item())
    elif hasattr(model, "logit_scale"):
        logit_scale = float(model.logit_scale.exp().item())

    loss = _contrastive_loss(image_feats, text_feats, logit_scale=logit_scale)

    result: dict[str, Any] = {
        "metrics": metrics,
        "loss": loss,
        "n_pairs": len(subset),
        "checkpoint": str(checkpoint_path),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }

    # Compute deltas vs baseline
    if base_metrics is not None:
        deltas: dict[str, float] = {}
        base_m: dict[str, float] = base_metrics.get("metrics", {})
        for key, value in metrics.items():
            if key in base_m:
                deltas[key] = value - base_m[key]
        deltas["loss"] = loss - base_metrics.get("loss", 0.0)
        result["deltas"] = deltas

    logger.info(
        "LoRA — loss=%.4f  i2t_r@1=%.3f  i2t_r@5=%.3f  t2i_r@1=%.3f  t2i_r@5=%.3f",
        loss,
        metrics["i2t_r@1"],
        metrics["i2t_r@5"],
        metrics["t2i_r@1"],
        metrics["t2i_r@5"],
    )
    if "deltas" in result:
        d = result["deltas"]
        logger.info(
            "Deltas — loss=%+.4f  i2t_r@1=%+.3f  i2t_r@5=%+.3f  t2i_r@1=%+.3f  t2i_r@5=%+.3f",
            d.get("loss", 0.0),
            d.get("i2t_r@1", 0.0),
            d.get("i2t_r@5", 0.0),
            d.get("t2i_r@1", 0.0),
            d.get("t2i_r@5", 0.0),
        )

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Probe selection
# ---------------------------------------------------------------------------


def select_probes(
    val_rows: list[tuple[str, str]],
    n_probes: int = 50,
    seed: int = 42,
) -> list[int]:
    """Select diverse probe indices from the validation set.

    Strategy: extract scene IDs from filenames and pick at most one frame per
    unique scene.  If there are fewer scenes than *n_probes*, fill the
    remainder with random samples from the remaining val indices.

    Args:
        val_rows: Validation set rows ``(filepath, caption)``.
        n_probes: Desired number of probe indices.
        seed: Random seed for reproducibility.

    Returns:
        Sorted list of indices into *val_rows*.
    """
    rng = random.Random(seed)

    # Group indices by scene ID
    scene_to_indices: dict[str, list[int]] = {}
    for idx, (filepath, _) in enumerate(val_rows):
        filename = Path(filepath).name
        match = re.match(r"s(\d+)_f", filename)
        scene_id = match.group(1) if match else filename
        scene_to_indices.setdefault(scene_id, []).append(idx)

    # Pick one index per scene (random within each scene's frames)
    scene_ids = sorted(scene_to_indices.keys())
    rng.shuffle(scene_ids)

    selected: list[int] = []
    used_indices: set[int] = set()
    for scene_id in scene_ids:
        if len(selected) >= n_probes:
            break
        idx = rng.choice(scene_to_indices[scene_id])
        selected.append(idx)
        used_indices.add(idx)

    # Fill remaining slots if we have fewer scenes than n_probes
    if len(selected) < n_probes:
        remaining = [i for i in range(len(val_rows)) if i not in used_indices]
        rng.shuffle(remaining)
        needed = n_probes - len(selected)
        selected.extend(remaining[:needed])

    return sorted(selected)


# ---------------------------------------------------------------------------
# Nearest-neighbour search
# ---------------------------------------------------------------------------


def find_nearest_neighbors(
    query_features: torch.Tensor,
    all_features: torch.Tensor,
    all_paths: list[str],
    top_k: int = 5,
    exclude_index: int | None = None,
) -> list[tuple[str, float]]:
    """Find top-K nearest neighbours by cosine similarity.

    Args:
        query_features: L2-normalised query vector ``(D,)`` or ``(1, D)``.
        all_features: L2-normalised feature matrix ``(N, D)``.
        all_paths: File paths corresponding to rows of *all_features*.
        top_k: Number of neighbours to return.
        exclude_index: If given, exclude this index (the query itself).

    Returns:
        List of ``(path, similarity_score)`` tuples, sorted descending
        by similarity.
    """
    if query_features.dim() == 1:
        query_features = query_features.unsqueeze(0)

    similarities: torch.Tensor = (query_features @ all_features.T).squeeze(0)  # (N,)

    if exclude_index is not None:
        similarities[exclude_index] = -1.0  # push self-match to bottom

    topk_values, topk_indices = similarities.topk(min(top_k, len(all_paths)))

    results: list[tuple[str, float]] = []
    for score, idx in zip(topk_values.tolist(), topk_indices.tolist()):
        results.append((all_paths[idx], float(score)))

    return results


# ---------------------------------------------------------------------------
# Gallery generation
# ---------------------------------------------------------------------------


def generate_nn_gallery(
    probe_idx: int,
    probe_image_path: str,
    base_neighbors: list[tuple[str, float]],
    lora_neighbors: list[tuple[str, float]],
    output_path: Path,
    top_k: int = 5,
) -> None:
    """Create a composite gallery image: query | base top-K | LoRA top-K.

    Layout: ``(1 + top_k + top_k)`` images in a row, each resized to
    224x224.  Labels are drawn above each section, and similarity scores
    below each neighbour thumbnail.

    Args:
        probe_idx: Probe index (for labelling).
        probe_image_path: Path to the query image.
        base_neighbors: Base model nearest neighbours ``(path, score)``.
        lora_neighbors: LoRA model nearest neighbours ``(path, score)``.
        output_path: Where to save the composite JPEG.
        top_k: Number of neighbours per model (should match list lengths).
    """
    thumb_size = 224
    label_height = 24
    score_height = 20
    total_images = 1 + top_k + top_k
    canvas_width = total_images * thumb_size
    canvas_height = label_height + thumb_size + score_height
    separator_width = 2  # visual separator between sections

    # Account for separators
    canvas_width += separator_width * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 20, 31))
    draw = ImageDraw.Draw(canvas)

    # Try to load a readable font; fall back to PIL default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    def _paste_thumb(img_path: str, x: int) -> None:
        """Load, resize, and paste a thumbnail at position (x, label_height)."""
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        except Exception:
            # Placeholder for unreadable images
            img = Image.new("RGB", (thumb_size, thumb_size), color=(40, 40, 60))
            d = ImageDraw.Draw(img)
            d.text((thumb_size // 4, thumb_size // 2), "N/A", fill=(180, 180, 180), font=font)
        canvas.paste(img, (x, label_height))

    x_offset = 0

    # --- Query image ---
    draw.text((x_offset + 4, 4), f"Query #{probe_idx}", fill=(200, 200, 255), font=font)
    _paste_thumb(probe_image_path, x_offset)
    x_offset += thumb_size

    # --- Separator ---
    draw.rectangle(
        [(x_offset, 0), (x_offset + separator_width, canvas_height)],
        fill=(80, 80, 120),
    )
    x_offset += separator_width

    # --- Base model top-K ---
    draw.text((x_offset + 4, 4), "Base Top-5", fill=(255, 180, 100), font=font)
    for i, (path, score) in enumerate(base_neighbors[:top_k]):
        img_x = x_offset + i * thumb_size
        _paste_thumb(path, img_x)
        # Draw similarity score below thumbnail
        score_text = f"{score:.3f}"
        draw.text(
            (img_x + 4, label_height + thumb_size + 2),
            score_text,
            fill=(255, 180, 100),
            font=font_small,
        )
    x_offset += top_k * thumb_size

    # --- Separator ---
    draw.rectangle(
        [(x_offset, 0), (x_offset + separator_width, canvas_height)],
        fill=(80, 80, 120),
    )
    x_offset += separator_width

    # --- LoRA model top-K ---
    draw.text((x_offset + 4, 4), "LoRA Top-5", fill=(100, 200, 255), font=font)
    for i, (path, score) in enumerate(lora_neighbors[:top_k]):
        img_x = x_offset + i * thumb_size
        _paste_thumb(path, img_x)
        score_text = f"{score:.3f}"
        draw.text(
            (img_x + 4, label_height + thumb_size + 2),
            score_text,
            fill=(100, 200, 255),
            font=font_small,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), "JPEG", quality=90)
    logger.debug("Saved gallery: %s", output_path)


def generate_text_query_gallery(
    query_idx: int,
    query_text: str,
    base_neighbors: list[tuple[str, float]],
    lora_neighbors: list[tuple[str, float]],
    output_path: Path,
    top_k: int = 5,
) -> None:
    """Create a composite gallery image: text query | base top-K | LoRA top-K.

    Same layout as :func:`generate_nn_gallery` but replaces the query image
    with a dark panel containing the query text, rendered with word wrapping
    and a green accent colour.

    Args:
        query_idx: Query index (for labelling).
        query_text: The text query string.
        base_neighbors: Base model nearest neighbours ``(path, score)``.
        lora_neighbors: LoRA model nearest neighbours ``(path, score)``.
        output_path: Where to save the composite JPEG.
        top_k: Number of neighbours per model (should match list lengths).
    """
    thumb_size = 224
    label_height = 24
    score_height = 20
    total_images = 1 + top_k + top_k
    canvas_width = total_images * thumb_size
    canvas_height = label_height + thumb_size + score_height
    separator_width = 2

    # Account for separators
    canvas_width += separator_width * 2

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(18, 20, 31))
    draw = ImageDraw.Draw(canvas)

    # Try to load a readable font; fall back to PIL default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_query = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_small = font
        font_query = font

    def _paste_thumb(img_path: str, x: int) -> None:
        """Load, resize, and paste a thumbnail at position (x, label_height)."""
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (thumb_size, thumb_size), color=(40, 40, 60))
            d = ImageDraw.Draw(img)
            d.text((thumb_size // 4, thumb_size // 2), "N/A", fill=(180, 180, 180), font=font)
        canvas.paste(img, (x, label_height))

    x_offset = 0

    # --- Query text panel ---
    draw.text((x_offset + 4, 4), f"Query #{query_idx}", fill=(100, 220, 160), font=font)
    # Dark panel for text — render query text word-wrapped and centered
    text_panel = Image.new("RGB", (thumb_size, thumb_size), color=(20, 25, 35))
    tp_draw = ImageDraw.Draw(text_panel)
    # Word-wrap the query text
    words = query_text.split()
    lines: list[str] = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = tp_draw.textbbox((0, 0), test_line, font=font_query)
        if bbox[2] - bbox[0] > thumb_size - 20:
            if current_line:
                lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    # Center the text block vertically
    line_height = 20
    total_text_height = len(lines) * line_height
    y_start = (thumb_size - total_text_height) // 2
    for i, line in enumerate(lines):
        bbox = tp_draw.textbbox((0, 0), line, font=font_query)
        text_width = bbox[2] - bbox[0]
        x_text = (thumb_size - text_width) // 2
        tp_draw.text(
            (x_text, y_start + i * line_height),
            line,
            fill=(100, 220, 160),
            font=font_query,
        )
    canvas.paste(text_panel, (x_offset, label_height))
    x_offset += thumb_size

    # --- Separator ---
    draw.rectangle(
        [(x_offset, 0), (x_offset + separator_width, canvas_height)],
        fill=(80, 80, 120),
    )
    x_offset += separator_width

    # --- Base model top-K ---
    draw.text((x_offset + 4, 4), "Base Top-5", fill=(255, 180, 100), font=font)
    for i, (path, score) in enumerate(base_neighbors[:top_k]):
        img_x = x_offset + i * thumb_size
        _paste_thumb(path, img_x)
        score_text = f"{score:.3f}"
        draw.text(
            (img_x + 4, label_height + thumb_size + 2),
            score_text,
            fill=(255, 180, 100),
            font=font_small,
        )
    x_offset += top_k * thumb_size

    # --- Separator ---
    draw.rectangle(
        [(x_offset, 0), (x_offset + separator_width, canvas_height)],
        fill=(80, 80, 120),
    )
    x_offset += separator_width

    # --- LoRA model top-K ---
    draw.text((x_offset + 4, 4), "LoRA Top-5", fill=(100, 200, 255), font=font)
    for i, (path, score) in enumerate(lora_neighbors[:top_k]):
        img_x = x_offset + i * thumb_size
        _paste_thumb(path, img_x)
        score_text = f"{score:.3f}"
        draw.text(
            (img_x + 4, label_height + thumb_size + 2),
            score_text,
            fill=(100, 200, 255),
            font=font_small,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), "JPEG", quality=90)
    logger.debug("Saved text query gallery: %s", output_path)


# ---------------------------------------------------------------------------
# Full evaluation with gallery generation
# ---------------------------------------------------------------------------


def evaluate_with_gallery(
    checkpoint_path: Path,
    val_rows: list[tuple[str, str]],
    model_name: str,
    pretrained: str,
    run_dir: Path,
    probe_indices: list[int],
    baseline: dict[str, Any],
    device: str = "cuda",
    max_eval_pairs: int = 500,
) -> dict[str, Any]:
    """Evaluate a checkpoint and generate nearest-neighbour gallery images.

    This is the main evaluation entry point that combines metric computation
    with visual gallery generation for qualitative analysis.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        val_rows: Full validation set rows.
        model_name: OpenCLIP model architecture name.
        pretrained: Pretrained weights tag.
        run_dir: Root directory for the training run.
        probe_indices: List of val-set indices to use as gallery probes.
        baseline: Baseline metrics dict (from :func:`compute_baseline`).
        device: Target device.
        max_eval_pairs: Maximum number of val pairs for metric computation.

    Returns:
        Evaluation results dict (metrics + gallery info).
    """
    # --- Step 1: Compute metrics ---
    eval_result = evaluate_checkpoint(
        checkpoint_path,
        val_rows,
        model_name,
        pretrained,
        base_metrics=baseline,
        max_eval_pairs=max_eval_pairs,
        device=device,
    )

    # --- Step 2: Encode full val set with both base and LoRA models for NN search ---
    # Subsample for gallery (use all val rows, not just max_eval_pairs subset)
    gallery_rows = val_rows

    # Encode with base model
    logger.info("Encoding val set with base model for NN gallery ...")
    base_model, _, base_preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    base_model = base_model.to(device)
    base_model.eval()
    base_tokenizer = open_clip.get_tokenizer(model_name)
    base_img_feats, _ = encode_dataset(
        base_model, gallery_rows, base_preprocess, base_tokenizer, device
    )

    # Encode text queries with base model (before cleanup)
    tq_path = run_dir / "eval" / "text_queries.json"
    text_queries: list[str] = []
    base_tq_feats: torch.Tensor | None = None
    if tq_path.exists():
        text_queries = json.loads(tq_path.read_text(encoding="utf-8"))
        tokens = base_tokenizer(text_queries).to(device)
        with torch.no_grad():
            base_tq_feats = F.normalize(base_model.encode_text(tokens), dim=-1).cpu()
        del tokens

    del base_model
    torch.cuda.empty_cache()

    # Encode with LoRA model
    logger.info("Encoding val set with LoRA model for NN gallery ...")
    ckpt: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config: dict[str, Any] = ckpt.get("config", {})

    lora_model, _, lora_preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    lora_model.eval()
    for p in lora_model.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=ckpt_config.get("lora_rank", 8),
        lora_alpha=ckpt_config.get("lora_alpha", 16),
        lora_dropout=ckpt_config.get("lora_dropout", 0.05),
        target_modules=["c_fc", "c_proj"],
    )
    lora_model = get_peft_model(lora_model, lora_cfg)
    lora_model.base_model.model.logit_scale.requires_grad_(True)
    lora_state = ckpt["model_state_dict"]
    ls_key = "base_model.model.logit_scale"
    if ls_key in lora_state:
        lora_model.base_model.model.logit_scale.data.copy_(lora_state.pop(ls_key))
    try:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(lora_model, lora_state)
    except Exception:
        lora_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    lora_model.eval()
    lora_model = lora_model.to(device)
    lora_tokenizer = open_clip.get_tokenizer(model_name)
    lora_img_feats, _ = encode_dataset(
        lora_model, gallery_rows, lora_preprocess, lora_tokenizer, device
    )

    # Encode text queries with LoRA model (before cleanup)
    lora_tq_feats: torch.Tensor | None = None
    if text_queries:
        tokens = lora_tokenizer(text_queries).to(device)
        with torch.no_grad():
            lora_tq_feats = F.normalize(lora_model.encode_text(tokens), dim=-1).cpu()
        del tokens

    del lora_model
    torch.cuda.empty_cache()

    # --- Step 3: Generate gallery images for each probe ---
    all_paths = [filepath for filepath, _ in gallery_rows]

    # Determine epoch label from checkpoint filename
    ckpt_stem = checkpoint_path.stem  # e.g. "epoch_001" or "best"
    gallery_dir = run_dir / "eval" / f"{ckpt_stem}_nn"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    n_generated = 0
    for i, probe_idx in enumerate(probe_indices):
        if probe_idx >= len(gallery_rows):
            logger.warning(
                "Probe index %d out of range (val set has %d rows), skipping",
                probe_idx,
                len(gallery_rows),
            )
            continue

        probe_path = gallery_rows[probe_idx][0]

        base_nn = find_nearest_neighbors(
            base_img_feats[probe_idx],
            base_img_feats,
            all_paths,
            top_k=5,
            exclude_index=probe_idx,
        )
        lora_nn = find_nearest_neighbors(
            lora_img_feats[probe_idx],
            lora_img_feats,
            all_paths,
            top_k=5,
            exclude_index=probe_idx,
        )

        # Use sequential index (i) for consistent naming with train_lora.py
        gallery_path = gallery_dir / f"probe_{i:03d}.jpg"
        generate_nn_gallery(probe_idx, probe_path, base_nn, lora_nn, gallery_path)
        n_generated += 1

    eval_result["gallery_dir"] = str(gallery_dir)
    eval_result["n_gallery_images"] = n_generated
    logger.info("Generated %d gallery images in %s", n_generated, gallery_dir)

    # --- Step 3b: Generate text query gallery images ---
    if text_queries and base_tq_feats is not None and lora_tq_feats is not None:
        tq_gallery_dir = run_dir / "eval" / f"{ckpt_stem}_tq"
        tq_gallery_dir.mkdir(parents=True, exist_ok=True)
        for qi, query_text in enumerate(text_queries):
            base_nn = find_nearest_neighbors(
                base_tq_feats[qi], base_img_feats, all_paths, top_k=5,
            )
            lora_nn = find_nearest_neighbors(
                lora_tq_feats[qi], lora_img_feats, all_paths, top_k=5,
            )
            generate_text_query_gallery(
                qi, query_text, base_nn, lora_nn,
                tq_gallery_dir / f"query_{qi:03d}.jpg",
            )
        eval_result["text_query_count"] = len(text_queries)
        logger.info("Generated %d text query galleries in %s", len(text_queries), tq_gallery_dir)

    # --- Step 4: Save eval results ---
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_json_path = eval_dir / f"{ckpt_stem}.json"
    with eval_json_path.open("w") as f:
        json.dump(eval_result, f, indent=2)
    logger.info("Saved eval results to %s", eval_json_path)

    return eval_result


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


def watch_mode(
    run_dir: Path,
    model_name: str,
    pretrained: str,
    max_eval_pairs: int = 500,
    n_probes: int = 50,
    device: str = "cuda",
    poll_interval: float = 30.0,
) -> None:
    """Watch a run directory for new checkpoints and evaluate them.

    Polls ``run_dir/checkpoints/`` every *poll_interval* seconds.  A
    checkpoint is considered "new" if no corresponding
    ``eval/<stem>.json`` exists yet.  The baseline is computed (or loaded)
    once on first invocation.

    Args:
        run_dir: Root directory for the training run.
        model_name: OpenCLIP model architecture name.
        pretrained: Pretrained weights tag.
        max_eval_pairs: Maximum val pairs for metrics.
        n_probes: Number of probe images for galleries.
        device: Target device.
        poll_interval: Seconds between directory polls.
    """
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load run config to get CSV path and val fraction
    config_path = run_dir / "config.json"
    if not config_path.exists():
        logger.error("Run config not found: %s", config_path)
        sys.exit(1)

    with config_path.open() as f:
        run_config: dict[str, Any] = json.load(f)

    csv_path = run_config.get("csv_path", "")
    val_fraction = run_config.get("val_fraction", 0.1)

    if not csv_path or not Path(csv_path).exists():
        logger.error("CSV path from run config not found: %s", csv_path)
        sys.exit(1)

    # Split dataset (same as training)
    _, val_rows = scene_level_split(csv_path, val_fraction=val_fraction, seed=42)
    logger.info("Loaded %d validation rows from %s", len(val_rows), csv_path)

    # Compute or load baseline
    baseline_path = eval_dir / "baseline.json"
    if baseline_path.exists():
        with baseline_path.open() as f:
            baseline: dict[str, Any] = json.load(f)
        logger.info("Loaded cached baseline from %s", baseline_path)
    else:
        baseline = compute_baseline(val_rows, model_name, pretrained, max_eval_pairs, device)
        with baseline_path.open("w") as f:
            json.dump(baseline, f, indent=2)
        logger.info("Saved baseline to %s", baseline_path)

    # Select or load probes
    probes_path = eval_dir / "probes.json"
    if probes_path.exists():
        with probes_path.open() as f:
            probe_indices: list[int] = json.load(f)
        logger.info("Loaded %d cached probe indices", len(probe_indices))
    else:
        probe_indices = select_probes(val_rows, n_probes=n_probes)
        with probes_path.open("w") as f:
            json.dump(probe_indices, f)
        logger.info("Selected %d probe indices, saved to %s", len(probe_indices), probes_path)

    logger.info(
        "Watching %s for new checkpoints (poll every %.0fs) ...",
        ckpt_dir,
        poll_interval,
    )

    try:
        while True:
            if not ckpt_dir.exists():
                time.sleep(poll_interval)
                continue

            # Find all .pt files, excluding symlinks (like best.pt)
            checkpoint_files = sorted(
                p for p in ckpt_dir.glob("*.pt") if not p.is_symlink()
            )

            for ckpt_path in checkpoint_files:
                stem = ckpt_path.stem
                eval_json = eval_dir / f"{stem}.json"
                if eval_json.exists():
                    continue  # already evaluated

                logger.info("New checkpoint detected: %s", ckpt_path)
                try:
                    evaluate_with_gallery(
                        ckpt_path,
                        val_rows,
                        model_name,
                        pretrained,
                        run_dir,
                        probe_indices,
                        baseline,
                        device=device,
                        max_eval_pairs=max_eval_pairs,
                    )
                except Exception:
                    logger.exception("Failed to evaluate %s", ckpt_path)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Watch mode stopped by user.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subsample_rows(
    rows: list[tuple[str, str]],
    max_rows: int,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Deterministically subsample rows if there are more than *max_rows*.

    Args:
        rows: Full row list.
        max_rows: Maximum number of rows to return.
        seed: Random seed for reproducibility.

    Returns:
        Subsampled (or original) row list.
    """
    if len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:max_rows])
    return [rows[i] for i in selected]


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load the training run's config.json.

    Args:
        run_dir: Root directory for the training run.

    Returns:
        Parsed config dictionary.

    Raises:
        SystemExit: If the config file is missing.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        logger.error("Run config not found: %s", config_path)
        sys.exit(1)

    with config_path.open() as f:
        config: dict[str, Any] = json.load(f)
    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for LoRA checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA checkpoints with R@K metrics and NN galleries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a specific checkpoint
  uv run python tools/training/eval_lora.py \\
      --checkpoint assets/lora_training/runs/my-run/checkpoints/epoch_001.pt \\
      --run my-run

  # Watch for new checkpoints and evaluate automatically
  uv run python tools/training/eval_lora.py --watch --run my-run

  # Evaluate with more pairs and probes
  uv run python tools/training/eval_lora.py \\
      --checkpoint best.pt --run my-run \\
      --max-pairs 1000 --n-probes 100
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pt checkpoint to evaluate. "
        "If a bare filename is given (e.g. 'epoch_001.pt'), it is resolved "
        "relative to the run's checkpoints/ directory.",
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run name (subdirectory under assets/lora_training/runs/).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        default=False,
        help="Watch the run's checkpoints/ directory for new files and evaluate "
        "each one automatically.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=500,
        help="Maximum number of image-text pairs for metric computation (default: 500).",
    )
    parser.add_argument(
        "--n-probes",
        type=int,
        default=50,
        help="Number of probe images for nearest-neighbour galleries (default: 50).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between checkpoint directory polls in watch mode (default: 30).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(DEFAULT_RUNS_DIR),
        help="Base directory containing training runs (default: assets/lora_training/runs)",
    )
    parser.add_argument(
        "--text-queries",
        nargs="*",
        default=None,
        help="Text queries for text-to-image galleries. If provided, overwrites "
        "text_queries.json in the eval directory.",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    runs_dir = Path(args.base_dir)
    run_dir = runs_dir / args.run
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    # Load run config for model info and dataset path
    run_config = _load_run_config(run_dir)
    model_name: str = run_config.get("model_name", "ViT-H-14")
    pretrained: str = run_config.get("pretrained", "laion2b_s32b_b79k")
    csv_path: str = run_config.get("csv_path", "")
    val_fraction: float = run_config.get("val_fraction", 0.1)

    if args.watch:
        watch_mode(
            run_dir,
            model_name,
            pretrained,
            max_eval_pairs=args.max_pairs,
            n_probes=args.n_probes,
            device=args.device,
            poll_interval=args.poll_interval,
        )
        return

    # --- Single checkpoint evaluation ---
    if args.checkpoint is None:
        logger.error("Either --checkpoint or --watch must be specified.")
        parser.print_help()
        sys.exit(1)

    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute() and not ckpt_path.exists():
        # Try relative to run's checkpoints directory
        ckpt_path = run_dir / "checkpoints" / ckpt_path
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    if not csv_path or not Path(csv_path).exists():
        logger.error("CSV path from run config not found: %s", csv_path)
        sys.exit(1)

    # Load validation set
    _, val_rows = scene_level_split(csv_path, val_fraction=val_fraction, seed=42)
    logger.info("Loaded %d validation rows from %s", len(val_rows), csv_path)

    # Compute or load baseline
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Handle text queries: overwrite text_queries.json if --text-queries provided
    if args.text_queries is not None:
        tq_path = eval_dir / "text_queries.json"
        tq_path.write_text(json.dumps(args.text_queries), encoding="utf-8")
        logger.info("Saved %d text queries to %s", len(args.text_queries), tq_path)

    baseline_path = eval_dir / "baseline.json"
    if baseline_path.exists():
        with baseline_path.open() as f:
            baseline: dict[str, Any] = json.load(f)
        logger.info("Loaded cached baseline from %s", baseline_path)
    else:
        baseline = compute_baseline(val_rows, model_name, pretrained, args.max_pairs, args.device)
        with baseline_path.open("w") as f:
            json.dump(baseline, f, indent=2)
        logger.info("Saved baseline to %s", baseline_path)

    # Select or load probes
    probes_path = eval_dir / "probes.json"
    if probes_path.exists():
        with probes_path.open() as f:
            probe_indices: list[int] = json.load(f)
        logger.info("Loaded %d cached probe indices", len(probe_indices))
    else:
        probe_indices = select_probes(val_rows, n_probes=args.n_probes)
        with probes_path.open("w") as f:
            json.dump(probe_indices, f)
        logger.info("Selected %d probe indices", len(probe_indices))

    # Evaluate
    result = evaluate_with_gallery(
        ckpt_path,
        val_rows,
        model_name,
        pretrained,
        run_dir,
        probe_indices,
        baseline,
        device=args.device,
        max_eval_pairs=args.max_pairs,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Val pairs:  {result['n_pairs']}")
    print(f"Loss:       {result['loss']:.4f}")
    print()
    print("Retrieval metrics:")
    for key, value in sorted(result["metrics"].items()):
        delta_str = ""
        if "deltas" in result and key in result["deltas"]:
            delta = result["deltas"][key]
            delta_str = f"  ({delta:+.3f})"
        print(f"  {key:12s} = {value:.3f}{delta_str}")
    if "deltas" in result:
        print(f"\n  loss delta   = {result['deltas'].get('loss', 0.0):+.4f}")
    print(f"\nGallery: {result.get('gallery_dir', 'N/A')}")
    print(f"Gallery images: {result.get('n_gallery_images', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
