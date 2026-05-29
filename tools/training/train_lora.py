#!/usr/bin/env python3
"""LoRA fine-tuning for OpenCLIP models using PEFT.

Applies low-rank adaptation to the attention output projections of both the
vision and text encoders, training with CLIP contrastive loss (InfoNCE).
Supports VRAM auto-detection, gradient accumulation, mixed-precision training,
early stopping, and live dashboard control via ``control.json``.

Usage:
    uv run python tools/training/train_lora.py --csv path/to/train_clean.csv
    uv run python tools/training/train_lora.py --csv data.csv --epochs 5 --lr 1e-4
    uv run python tools/training/train_lora.py --csv data.csv --resume path/to/checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.training.eval_lora import (
    compute_retrieval_metrics,
    encode_dataset,
    find_nearest_neighbors,
    generate_nn_gallery,
    generate_text_query_gallery,
    select_probes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VRAM_TIER_HIGH_MB: int = 28 * 1024   # >= 28 GB  (A100, 5090)
_VRAM_TIER_MID_MB: int = 16 * 1024    # >= 16 GB  (4080, 3090)
_VRAM_TIER_LOW_MB: int = 14 * 1024    # >= 14 GB  (needs headroom for ViT-H-14 backward pass)
_WARMUP_STEPS: int = 200
_LOG_EVERY: int = 10
_GRAD_MAX_NORM: float = 1.0
_WEIGHT_DECAY: float = 0.01

DEFAULT_TEXT_QUERIES: list[str] = [
    "asshole closeup", "anal sex", "creampie", "anal creampie",
    "big ass closeup", "cum in mouth", "cum leaking",
]

# ---------------------------------------------------------------------------
# VRAM auto-detection
# ---------------------------------------------------------------------------


def detect_vram_tier() -> tuple[int, int, int]:
    """Select batch size and gradient-accumulation steps based on GPU VRAM.

    Tiers (effective batch size is always 64):
        - >= 28 GB  -> batch_size=64, grad_accum=1
        - >= 16 GB  -> batch_size=16, grad_accum=4
        - >= 14 GB  -> batch_size=8,  grad_accum=8
        - <  14 GB  -> batch_size=4,  grad_accum=16  (12GB GPUs like 4070S)
        - No CUDA   -> batch_size=4,  grad_accum=16  (CPU fallback)

    Returns:
        Tuple of ``(batch_size, grad_accum, vram_mb)`` where *vram_mb* is the
        total device memory in MiB (0 when CUDA is unavailable).
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available -- falling back to CPU defaults (batch=4, accum=16)")
        return 4, 16, 0

    props = torch.cuda.get_device_properties(0)
    vram_mb: int = props.total_memory // (1024 * 1024)

    if vram_mb >= _VRAM_TIER_HIGH_MB:
        batch_size, grad_accum = 64, 1
    elif vram_mb >= _VRAM_TIER_MID_MB:
        batch_size, grad_accum = 16, 4
    elif vram_mb >= _VRAM_TIER_LOW_MB:
        batch_size, grad_accum = 8, 8
    else:
        batch_size, grad_accum = 4, 16

    logger.info(
        "VRAM detected: %d MiB (%s) -> batch_size=%d, grad_accum=%d",
        vram_mb,
        props.name,
        batch_size,
        grad_accum,
    )
    return batch_size, grad_accum, vram_mb


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def setup_lora_model(
    model_name: str,
    pretrained: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    device: torch.device,
) -> tuple[Any, Any, Any, Any]:
    """Load an OpenCLIP model and inject LoRA adapters via PEFT.

    LoRA is applied to the MLP layers (``c_fc``, ``c_proj``) in every
    transformer block of both the visual and text encoders.

    Note: We target MLP layers instead of attention ``out_proj`` because
    OpenCLIP uses ``nn.MultiheadAttention`` which passes ``out_proj.weight``
    as a raw tensor to ``F.multi_head_attention_forward()``, bypassing PEFT's
    module-level ``forward()`` hook entirely.  MLP layers are standard
    ``nn.Linear`` modules called through ``forward()`` and work correctly.

    The ``logit_scale`` parameter is manually unfrozen after PEFT wrapping
    (it is a scalar ``nn.Parameter``, not an ``nn.Module``, so PEFT's
    ``modules_to_save`` cannot handle it).

    Args:
        model_name: OpenCLIP architecture name (e.g. ``"ViT-H-14"``).
        pretrained: Pretrained checkpoint tag (e.g. ``"laion2b_s32b_b79k"``).
        lora_rank: LoRA rank (*r*).
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability on LoRA layers.
        device: Target torch device.

    Returns:
        Tuple of ``(peft_model, preprocess_train, preprocess_eval, tokenizer)``.
    """
    import open_clip
    from peft import LoraConfig, get_peft_model

    logger.info("Loading OpenCLIP model %s (pretrained=%s)", model_name, pretrained)
    model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )

    # Freeze the entire base model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Configure LoRA — target MLP layers (c_fc, c_proj) in both encoders.
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["c_fc", "c_proj"],
    )

    peft_model = get_peft_model(model, lora_config)

    # Manually unfreeze logit_scale (nn.Parameter, not a module)
    peft_model.base_model.model.logit_scale.requires_grad_(True)

    # Report trainable parameters
    trainable_params: int = 0
    total_params: int = 0
    for param in peft_model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(
        "LoRA applied: %s trainable / %s total parameters (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100.0 * trainable_params / total_params if total_params > 0 else 0.0,
    )

    tokenizer = open_clip.get_tokenizer(model_name)
    return peft_model, preprocess_train, preprocess_eval, tokenizer


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def clip_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric CLIP contrastive loss (InfoNCE).

    Args:
        image_features: ``(B, D)`` image embeddings (not necessarily normalized).
        text_features: ``(B, D)`` text embeddings (not necessarily normalized).
        logit_scale: Scalar log-temperature parameter (``model.logit_scale``).

    Returns:
        Scalar loss tensor.
    """
    # L2-normalize
    image_norm = F.normalize(image_features, p=2, dim=1)
    text_norm = F.normalize(text_features, p=2, dim=1)

    # Scaled cosine similarity
    scale = logit_scale.exp()
    logits_per_image = scale * (image_norm @ text_norm.T)
    logits_per_text = logits_per_image.T

    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    loss = (
        F.cross_entropy(logits_per_image, labels)
        + F.cross_entropy(logits_per_text, labels)
    ) / 2.0

    return loss


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Complete training configuration.

    Serialisable via ``dataclasses.asdict`` and written to ``config.json``
    at the start of each run.
    """

    csv_path: str
    model_name: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lr: float = 2e-4
    epochs: int = 10
    patience: int = 3
    val_fraction: float = 0.1
    batch_size: int = 16
    grad_accum: int = 4
    eval_interval: int = 0  # 0 = epoch-only, >0 = validate every N optimizer steps
    num_workers: int = 4
    run_name: str = ""
    resume_from: str = ""
    path_remap: str = ""  # "OLD_PREFIX=NEW_PREFIX" to rewrite CSV image paths
    text_queries: list[str] = field(default_factory=lambda: list(DEFAULT_TEXT_QUERIES))
    base_dir: str = "assets/lora_training/runs"


# ---------------------------------------------------------------------------
# Atomic JSON writer
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as JSON to *path* atomically.

    Writes to a temporary file in the same directory and then uses
    ``os.replace`` to move it into place.  This prevents corrupt JSON if the
    process crashes or is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------


def _warmup_cosine_lambda(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Learning-rate multiplier: linear warmup then cosine decay to zero."""
    if current_step < warmup_steps:
        return current_step / max(warmup_steps, 1)
    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Control file helpers
# ---------------------------------------------------------------------------


def _read_control(control_path: Path) -> dict[str, Any]:
    """Read and parse the control file, returning defaults on any error."""
    default: dict[str, Any] = {"command": "none", "value": None}
    if not control_path.exists():
        return default
    try:
        with control_path.open("r") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError):
        return default


def _reset_control(control_path: Path) -> None:
    """Reset the control file to the default ``none`` command."""
    atomic_write_json(control_path, {"command": "none", "value": None})


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: TrainConfig) -> None:  # noqa: C901 — complexity is inherent to training loops
    """Execute the full LoRA training loop.

    Creates a run directory, sets up the model, dataloaders, optimizer, and
    scheduler, then trains for up to ``config.epochs`` epochs with early
    stopping.  State is checkpointed after each epoch and metrics are logged
    to ``train_log.jsonl`` every ``_LOG_EVERY`` steps.

    The training process can be controlled at runtime by writing commands to
    ``control.json`` in the run directory (e.g. from the training dashboard).

    Args:
        config: Fully-populated training configuration.
    """
    from tools.training.clip_dataset import CLIPDataset, scene_level_split

    # -----------------------------------------------------------------------
    # Run directory
    # -----------------------------------------------------------------------
    run_dir = Path(config.base_dir) / config.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    control_path = run_dir / "control.json"
    state_path = run_dir / "train_state.json"
    log_path = run_dir / "train_log.jsonl"

    # Persist config
    atomic_write_json(config_path, asdict(config))
    # Initialise control file
    _reset_control(control_path)

    logger.info("Run directory: %s", run_dir)

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    peft_model, preprocess_train, preprocess_eval, tokenizer = setup_lora_model(
        model_name=config.model_name,
        pretrained=config.pretrained,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Data loaders
    # -----------------------------------------------------------------------
    train_rows, val_rows = scene_level_split(
        config.csv_path, val_fraction=config.val_fraction,
    )

    # Rewrite image paths if --path-remap was provided (e.g. for remote training)
    if config.path_remap and "=" in config.path_remap:
        old_prefix, new_prefix = config.path_remap.split("=", 1)
        train_rows = [(fp.replace(old_prefix, new_prefix), cap) for fp, cap in train_rows]
        val_rows = [(fp.replace(old_prefix, new_prefix), cap) for fp, cap in val_rows]
        logger.info("Path remap: '%s' -> '%s' (%d train + %d val paths rewritten)",
                     old_prefix, new_prefix, len(train_rows), len(val_rows))

    train_dataset = CLIPDataset(train_rows, preprocess_train, tokenizer)
    val_dataset = CLIPDataset(val_rows, preprocess_train, tokenizer)

    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    logger.info(
        "Dataset: %d train samples (%d batches), %d val samples (%d batches)",
        len(train_dataset),
        len(train_loader),
        len(val_dataset),
        len(val_loader),
    )

    # -----------------------------------------------------------------------
    # Baseline eval (base model, no LoRA) — reuses the in-memory model via
    # PEFT's disable_adapter(), avoiding a second ~7.8 GB model load.
    # -----------------------------------------------------------------------
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    n_probes = min(50, len(val_rows))

    logger.info("Computing baseline metrics (LoRA disabled)...")
    eval_batch = max(config.batch_size // 2, 1)
    peft_model.eval()
    with torch.no_grad(), peft_model.disable_adapter():
        base_img_feats, base_txt_feats = encode_dataset(
            peft_model, val_rows, preprocess_eval, tokenizer,
            device=device, batch_size=eval_batch,
        )

    base_metrics = compute_retrieval_metrics(base_img_feats, base_txt_feats)
    # Contrastive loss on base
    logit_scale_val = peft_model.base_model.model.logit_scale.exp()
    sim = base_img_feats @ base_txt_feats.T * logit_scale_val.item()
    labels_base = torch.arange(sim.shape[0])
    base_loss = (
        F.cross_entropy(sim, labels_base) + F.cross_entropy(sim.T, labels_base)
    ).item() / 2

    baseline_data: dict[str, Any] = {
        "metrics": base_metrics,
        "loss": base_loss,
        "n_pairs": len(val_rows),
        "model_name": config.model_name,
        "pretrained": config.pretrained,
    }
    atomic_write_json(eval_dir / "baseline.json", baseline_data)
    logger.info(
        "Baseline — loss=%.4f i2t_r@1=%.3f i2t_r@5=%.3f",
        base_loss, base_metrics["i2t_r@1"], base_metrics["i2t_r@5"],
    )

    # Select and cache probe indices for galleries
    probe_indices = select_probes(val_rows, n_probes=n_probes)
    atomic_write_json(eval_dir / "probes.json", probe_indices)

    if config.text_queries:
        atomic_write_json(eval_dir / "text_queries.json", config.text_queries)

    peft_model.train()

    # -----------------------------------------------------------------------
    # Optimizer & scheduler
    # -----------------------------------------------------------------------
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=_WEIGHT_DECAY)

    total_steps = len(train_loader) * config.epochs // config.grad_accum
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_lambda(step, _WARMUP_STEPS, total_steps),
    )

    # -----------------------------------------------------------------------
    # Mixed-precision scaler
    # -----------------------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------
    start_epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0

    if config.resume_from:
        resume_path = Path(config.resume_from)
        if resume_path.exists():
            logger.info("Resuming from checkpoint: %s", resume_path)
            ckpt: dict[str, Any] = torch.load(resume_path, map_location=device, weights_only=False)
            from peft import set_peft_model_state_dict

            lora_state = ckpt["model_state_dict"]
            # Restore logit_scale if saved separately
            ls_key = "base_model.model.logit_scale"
            if ls_key in lora_state:
                peft_model.base_model.model.logit_scale.data.copy_(lora_state.pop(ls_key))
            set_peft_model_state_dict(peft_model, lora_state)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            logger.info(
                "Resumed at epoch %d, step %d (best_val_loss=%.4f)",
                start_epoch, global_step, best_val_loss,
            )
        else:
            logger.warning("Checkpoint %s not found, starting from scratch", resume_path)

    # -----------------------------------------------------------------------
    # Training state helper
    # -----------------------------------------------------------------------
    t_start = time.monotonic()

    def _write_state(
        status: str,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: float,
    ) -> None:
        elapsed = time.monotonic() - t_start
        if step > 0 and total_steps > 0:
            steps_remaining = total_steps - step
            eta = elapsed / step * steps_remaining
        else:
            eta = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        state: dict[str, Any] = {
            "status": status,
            "epoch": epoch,
            "total_epochs": config.epochs,
            "step": step,
            "total_steps": total_steps,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "best_val_loss": round(best_val_loss, 6),
            "lr": current_lr,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        atomic_write_json(state_path, state)

    def _append_log(
        epoch: int,
        step: int,
        train_loss: float,
        *,
        val_loss: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "train_loss": round(train_loss, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        if val_loss is not None:
            entry["val_loss"] = round(val_loss, 6)
        if metrics is not None:
            entry["metrics"] = {k: round(v, 6) for k, v in metrics.items()}
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------
    _write_state("training", start_epoch, global_step, 0.0, best_val_loss)

    stopped_early = False

    for epoch in range(start_epoch, config.epochs):
        # ---- TRAIN --------------------------------------------------------
        peft_model.train()
        epoch_loss: float = 0.0
        epoch_batches: int = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, texts) in enumerate(train_loader):
            # -- Check control file for dashboard commands ---
            control = _read_control(control_path)
            command: str = control.get("command", "none")

            if command == "stop":
                logger.info("Stop command received via control.json")
                _write_state("stopped", epoch, global_step, epoch_loss / max(epoch_batches, 1), best_val_loss)
                _reset_control(control_path)
                stopped_early = True
                break

            if command == "pause":
                logger.info("Pause command received -- waiting for resume...")
                _write_state("paused", epoch, global_step, epoch_loss / max(epoch_batches, 1), best_val_loss)
                while True:
                    time.sleep(1.0)
                    ctrl = _read_control(control_path)
                    if ctrl.get("command") != "pause":
                        break
                _reset_control(control_path)
                _write_state("training", epoch, global_step, epoch_loss / max(epoch_batches, 1), best_val_loss)
                logger.info("Resumed training")

            if command == "set_lr":
                new_lr = control.get("value")
                if isinstance(new_lr, (int, float)) and new_lr > 0:
                    for pg in optimizer.param_groups:
                        pg["lr"] = float(new_lr)
                    logger.info("Learning rate updated to %g via control.json", new_lr)
                _reset_control(control_path)

            # -- Forward pass ------------------------------------------------
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                image_features = peft_model.encode_image(images)
                text_features = peft_model.encode_text(texts)
                loss = clip_contrastive_loss(
                    image_features, text_features, peft_model.base_model.model.logit_scale,
                )
                # Scale loss for gradient accumulation
                loss = loss / config.grad_accum

            scaler.scale(loss).backward()

            # -- Optimizer step (every grad_accum batches) -------------------
            if (batch_idx + 1) % config.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=_GRAD_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            # Accumulate un-scaled loss for logging
            epoch_loss += loss.item() * config.grad_accum
            epoch_batches += 1

            # -- Periodic logging (every _LOG_EVERY optimizer steps) -----------
            is_step_boundary = (batch_idx + 1) % config.grad_accum == 0
            if is_step_boundary and global_step % _LOG_EVERY == 0:
                avg_loss = epoch_loss / epoch_batches
                _write_state("training", epoch + 1, global_step, avg_loss, best_val_loss)
                _append_log(epoch + 1, global_step, avg_loss)
                logger.info(
                    "Epoch %d/%d  step %d  batch %d/%d  loss=%.4f  lr=%.2e",
                    epoch + 1,
                    config.epochs,
                    global_step,
                    batch_idx + 1,
                    len(train_loader),
                    avg_loss,
                    optimizer.param_groups[0]["lr"],
                )

            # -- Mid-epoch validation (every eval_interval optimizer steps) --
            if (config.eval_interval > 0
                    and is_step_boundary
                    and global_step > 0
                    and global_step % config.eval_interval == 0):
                logger.info("Mid-epoch eval at step %d — computing val metrics...", global_step)
                peft_model.eval()
                mid_val_loss = 0.0
                mid_val_batches = 0
                mid_img_feats: list[torch.Tensor] = []
                mid_txt_feats: list[torch.Tensor] = []
                with torch.no_grad():
                    for val_images, val_texts in val_loader:
                        val_images = val_images.to(device, non_blocking=True)
                        val_texts = val_texts.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                            vi = peft_model.encode_image(val_images)
                            vt = peft_model.encode_text(val_texts)
                            vloss = clip_contrastive_loss(
                                vi, vt, peft_model.base_model.model.logit_scale,
                            )
                        mid_val_loss += vloss.item()
                        mid_val_batches += 1
                        mid_img_feats.append(F.normalize(vi, dim=-1).cpu().float())
                        mid_txt_feats.append(F.normalize(vt, dim=-1).cpu().float())
                        del vi, vt, vloss

                avg_mid_val = mid_val_loss / max(mid_val_batches, 1)

                # R@K metrics from accumulated features
                img_feats_cat = torch.cat(mid_img_feats)
                txt_feats_cat = torch.cat(mid_txt_feats)
                del mid_img_feats, mid_txt_feats
                lora_metrics = compute_retrieval_metrics(img_feats_cat, txt_feats_cat)
                deltas = {k: lora_metrics[k] - base_metrics[k] for k in base_metrics}
                deltas["loss"] = avg_mid_val - base_loss

                logger.info(
                    "Eval step %d: val_loss=%.4f  i2t_r@1=%.3f (%+.3f)  i2t_r@5=%.3f (%+.3f)",
                    global_step, avg_mid_val,
                    lora_metrics["i2t_r@1"], deltas["i2t_r@1"],
                    lora_metrics["i2t_r@5"], deltas["i2t_r@5"],
                )

                # Write eval JSON (dashboard reads eval/epoch_NNN.json)
                eval_result: dict[str, Any] = {
                    "metrics": lora_metrics,
                    "loss": avg_mid_val,
                    "n_pairs": img_feats_cat.shape[0],
                    "checkpoint": f"step_{global_step}",
                    "lora_rank": config.lora_rank,
                    "lora_alpha": config.lora_alpha,
                    "deltas": deltas,
                }
                # Mid-epoch probe galleries
                logger.info("Generating mid-epoch probe galleries (step %d)...", global_step)
                mid_gallery_dir = eval_dir / f"epoch_{epoch + 1:03d}_nn"
                mid_gallery_dir.mkdir(parents=True, exist_ok=True)
                all_val_paths_mid = [fp for fp, _ in val_rows]
                for pi, probe_idx in enumerate(probe_indices):
                    if probe_idx >= img_feats_cat.shape[0]:
                        continue
                    base_nn = find_nearest_neighbors(
                        base_img_feats[probe_idx], base_img_feats,
                        all_val_paths_mid, top_k=5, exclude_index=probe_idx,
                    )
                    lora_nn = find_nearest_neighbors(
                        img_feats_cat[probe_idx], img_feats_cat,
                        all_val_paths_mid, top_k=5, exclude_index=probe_idx,
                    )
                    generate_nn_gallery(
                        probe_idx=probe_idx,
                        probe_image_path=val_rows[probe_idx][0],
                        base_neighbors=base_nn,
                        lora_neighbors=lora_nn,
                        output_path=mid_gallery_dir / f"probe_{pi:03d}.jpg",
                    )
                eval_result["gallery_dir"] = str(mid_gallery_dir)
                eval_result["n_gallery_images"] = len(probe_indices)

                # Mid-epoch text query galleries
                if config.text_queries:
                    logger.info("Generating mid-epoch text query galleries (step %d)...", global_step)
                    mid_tq_dir = eval_dir / f"epoch_{epoch + 1:03d}_tq"
                    mid_tq_dir.mkdir(parents=True, exist_ok=True)
                    mid_tq_tokens = tokenizer(config.text_queries).to(device)
                    with torch.no_grad():
                        mid_lora_tq = F.normalize(peft_model.encode_text(mid_tq_tokens), dim=-1).cpu().float()
                        with peft_model.disable_adapter():
                            mid_base_tq = F.normalize(peft_model.encode_text(mid_tq_tokens), dim=-1).cpu().float()
                    for qi, query_text in enumerate(config.text_queries):
                        base_nn = find_nearest_neighbors(
                            mid_base_tq[qi], base_img_feats, all_val_paths_mid, top_k=5,
                        )
                        lora_nn = find_nearest_neighbors(
                            mid_lora_tq[qi], img_feats_cat, all_val_paths_mid, top_k=5,
                        )
                        generate_text_query_gallery(
                            qi, query_text, base_nn, lora_nn,
                            mid_tq_dir / f"query_{qi:03d}.jpg",
                        )
                    del mid_tq_tokens, mid_lora_tq, mid_base_tq
                    eval_result["text_query_count"] = len(config.text_queries)
                    logger.info("Mid-epoch galleries: %d probes + %d text queries (step %d)",
                                len(probe_indices), len(config.text_queries), global_step)
                else:
                    logger.info("Mid-epoch galleries: %d probes (step %d)",
                                len(probe_indices), global_step)

                atomic_write_json(eval_dir / f"epoch_{epoch + 1:03d}.json", eval_result)

                _write_state("training", epoch + 1, global_step,
                             epoch_loss / max(epoch_batches, 1), avg_mid_val)
                _append_log(epoch + 1, global_step, epoch_loss / max(epoch_batches, 1),
                            val_loss=avg_mid_val, metrics=lora_metrics)
                peft_model.train()

        if stopped_early:
            break

        avg_train_loss = epoch_loss / max(epoch_batches, 1)

        # ---- VALIDATE (with R@K feature collection) --------------------------
        peft_model.eval()
        val_loss_sum: float = 0.0
        val_batches: int = 0
        epoch_img_feats: list[torch.Tensor] = []
        epoch_txt_feats: list[torch.Tensor] = []

        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    image_features = peft_model.encode_image(images)
                    text_features = peft_model.encode_text(texts)
                    loss = clip_contrastive_loss(
                        image_features, text_features, peft_model.base_model.model.logit_scale,
                    )

                val_loss_sum += loss.item()
                val_batches += 1
                epoch_img_feats.append(F.normalize(image_features, dim=-1).cpu().float())
                epoch_txt_feats.append(F.normalize(text_features, dim=-1).cpu().float())
                del image_features, text_features, loss

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        # R@K metrics for end-of-epoch
        epoch_img_cat = torch.cat(epoch_img_feats)
        epoch_txt_cat = torch.cat(epoch_txt_feats)
        del epoch_img_feats, epoch_txt_feats
        epoch_metrics = compute_retrieval_metrics(epoch_img_cat, epoch_txt_cat)
        epoch_deltas = {k: epoch_metrics[k] - base_metrics[k] for k in base_metrics}
        epoch_deltas["loss"] = avg_val_loss - base_loss

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  best=%.4f  "
            "i2t_r@1=%.3f (%+.3f)  i2t_r@5=%.3f (%+.3f)",
            epoch + 1, config.epochs, avg_train_loss, avg_val_loss, best_val_loss,
            epoch_metrics["i2t_r@1"], epoch_deltas["i2t_r@1"],
            epoch_metrics["i2t_r@5"], epoch_deltas["i2t_r@5"],
        )
        _append_log(epoch + 1, global_step, avg_train_loss,
                     val_loss=avg_val_loss, metrics=epoch_metrics)

        # ---- CHECKPOINT ----------------------------------------------------
        ckpt_name = f"epoch_{epoch + 1:03d}.pt"
        ckpt_path = checkpoint_dir / ckpt_name

        # Save only LoRA adapter weights + logit_scale (not the frozen base model)
        # to keep checkpoints small (~20 MB vs ~4 GB).
        from peft import get_peft_model_state_dict

        lora_state = get_peft_model_state_dict(peft_model)
        # Also save logit_scale which is manually unfrozen (not in PEFT's state)
        lora_state["base_model.model.logit_scale"] = (
            peft_model.base_model.model.logit_scale.data.clone()
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": lora_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "config": asdict(config),
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved: %s", ckpt_path)

        # ---- Eval JSON + NN galleries (epoch boundary) ---------------------
        logger.info("Generating epoch %d probe galleries...", epoch + 1)
        epoch_eval: dict[str, Any] = {
            "metrics": epoch_metrics,
            "loss": avg_val_loss,
            "n_pairs": epoch_img_cat.shape[0],
            "checkpoint": ckpt_name,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "deltas": epoch_deltas,
        }

        # Generate gallery images comparing base vs LoRA nearest neighbours
        gallery_dir = eval_dir / f"epoch_{epoch + 1:03d}_nn"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        all_val_paths = [fp for fp, _ in val_rows]

        for i, probe_idx in enumerate(probe_indices):
            if probe_idx >= epoch_img_cat.shape[0]:
                continue
            base_nn = find_nearest_neighbors(
                base_img_feats[probe_idx], base_img_feats,
                all_val_paths, top_k=5, exclude_index=probe_idx,
            )
            lora_nn = find_nearest_neighbors(
                epoch_img_cat[probe_idx], epoch_img_cat,
                all_val_paths, top_k=5, exclude_index=probe_idx,
            )
            generate_nn_gallery(
                probe_idx=probe_idx,
                probe_image_path=val_rows[probe_idx][0],
                base_neighbors=base_nn,
                lora_neighbors=lora_nn,
                output_path=gallery_dir / f"probe_{i:03d}.jpg",
            )

        epoch_eval["gallery_dir"] = str(gallery_dir)
        epoch_eval["n_gallery_images"] = len(probe_indices)
        logger.info("Generated %d gallery images in %s", len(probe_indices), gallery_dir)

        # ---- Text query galleries -------------------------------------------
        if config.text_queries:
            logger.info("Generating epoch %d text query galleries...", epoch + 1)
            tq_dir = eval_dir / f"epoch_{epoch + 1:03d}_tq"
            tq_dir.mkdir(parents=True, exist_ok=True)
            text_tokens = tokenizer(config.text_queries).to(device)
            with torch.no_grad():
                lora_tq = F.normalize(peft_model.encode_text(text_tokens), dim=-1).cpu().float()
                with peft_model.disable_adapter():
                    base_tq = F.normalize(peft_model.encode_text(text_tokens), dim=-1).cpu().float()
            for qi, query_text in enumerate(config.text_queries):
                base_nn = find_nearest_neighbors(
                    base_tq[qi], base_img_feats, all_val_paths, top_k=5,
                )
                lora_nn = find_nearest_neighbors(
                    lora_tq[qi], epoch_img_cat, all_val_paths, top_k=5,
                )
                generate_text_query_gallery(
                    qi, query_text, base_nn, lora_nn,
                    tq_dir / f"query_{qi:03d}.jpg",
                )
            del text_tokens, lora_tq, base_tq
            epoch_eval["text_query_count"] = len(config.text_queries)
            logger.info("Generated %d text query galleries in %s", len(config.text_queries), tq_dir)

        atomic_write_json(eval_dir / f"epoch_{epoch + 1:03d}.json", epoch_eval)

        # ---- Early stopping ------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Update best.pt symlink
            best_link = checkpoint_dir / "best.pt"
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(ckpt_name)
            logger.info("New best model (val_loss=%.4f) -> %s", best_val_loss, ckpt_name)
        else:
            patience_counter += 1
            logger.info(
                "No improvement (patience %d/%d)",
                patience_counter,
                config.patience,
            )
            if patience_counter >= config.patience:
                logger.info("Early stopping triggered after %d epochs", epoch + 1)
                _write_state("completed", epoch + 1, global_step, avg_train_loss, avg_val_loss)
                break

        _write_state("training", epoch + 1, global_step, avg_train_loss, avg_val_loss)
    else:
        # Loop completed without break — all epochs finished
        _write_state("completed", config.epochs, global_step, avg_train_loss, avg_val_loss)

    if stopped_early:
        logger.info("Training stopped at epoch %d, step %d", epoch + 1, global_step)
    else:
        logger.info("Training complete. Best val_loss=%.4f", best_val_loss)

    logger.info("Run directory: %s", run_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for OpenCLIP models using PEFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to training CSV (columns: filepath,caption)",
    )
    parser.add_argument("--model", default="ViT-H-14", help="OpenCLIP architecture name")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k", help="Pretrained weights tag")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience (epochs)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of scenes for validation")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Override batch size (0 = auto-detect from VRAM)")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="Run validation every N optimizer steps (0 = epoch-only)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--run-name", default="", help="Run name (auto-generated if omitted)")
    parser.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    parser.add_argument("--path-remap", default="",
                        help="Rewrite CSV image paths: OLD_PREFIX=NEW_PREFIX (e.g. /home/user=H:)")
    parser.add_argument("--text-queries", nargs="*", default=None,
                        help="Text queries for text-to-image galleries (default: built-in list)")
    parser.add_argument("--base-dir", default="assets/lora_training/runs",
                        help="Base directory for training run output (checkpoints, logs, gallery)")

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Auto-detect VRAM tier
    batch_size, grad_accum, vram_mb = detect_vram_tier()

    if args.batch_size > 0:
        batch_size = args.batch_size
        grad_accum = max(1, 64 // batch_size)  # Keep effective batch = 64
        logger.info("Batch size overridden: batch_size=%d, grad_accum=%d", batch_size, grad_accum)

    # Generate run name if not provided
    run_name = args.run_name or datetime.now(tz=timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    config = TrainConfig(
        csv_path=args.csv,
        model_name=args.model,
        pretrained=args.pretrained,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        val_fraction=args.val_fraction,
        batch_size=batch_size,
        grad_accum=grad_accum,
        eval_interval=args.eval_interval,
        num_workers=args.num_workers,
        run_name=run_name,
        resume_from=args.resume,
        path_remap=args.path_remap,
        text_queries=args.text_queries if args.text_queries is not None else list(DEFAULT_TEXT_QUERIES),
        base_dir=args.base_dir,
    )

    logger.info("Training config: %s", json.dumps(asdict(config), indent=2))
    train(config)


if __name__ == "__main__":
    main()
