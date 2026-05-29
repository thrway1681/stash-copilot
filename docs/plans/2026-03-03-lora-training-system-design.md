# LoRA Training System Design

**Date:** 2026-03-03
**Status:** Approved

## Overview

A three-component system for fine-tuning the OpenCLIP ViT-H-14 embedding model with LoRA adapters, using the project's 63K clean image-caption dataset. Designed to prevent over/underfitting with continuous evaluation and visual sanity checks.

## Components

### 1. `tools/training/train_lora.py` — Training Script

**Model setup:**
- Base: OpenCLIP ViT-H-14 (`laion2b_s32b_b79k`), frozen
- LoRA via HuggingFace PEFT applied to BOTH vision and text encoder attention projections
- Target modules: `q_proj`, `k_proj` (packed as `in_proj_weight`), `v_proj`, `out_proj` in all transformer blocks of both encoders
- Default: rank=8, alpha=16, dropout=0.05

**VRAM auto-detection:**
- >= 28 GB (5090): batch_size=64, grad_accum=1
- >= 10 GB (4070S): batch_size=16, grad_accum=4
- < 10 GB: batch_size=8, grad_accum=8
- Effective batch size always 64 regardless of GPU

**Training loop:**
- Loss: CLIP contrastive loss (InfoNCE)
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: linear warmup (200 steps) -> cosine decay to 0
- Precision: fp16 via torch.cuda.amp
- Early stopping: patience=3 epochs on val loss
- State file: `train_state.json` updated every 10 steps
- Log file: `train_log.jsonl` append-only per-step metrics
- Control file: `control.json` — checked each step for pause/stop/lr-override commands from dashboard

**CLI:**
```
uv run python tools/training/train_lora.py
    --csv assets/lora_dataset/train_clean.csv
    --model ViT-H-14
    --lora-rank 8
    --lora-alpha 16
    --lr 2e-4
    --epochs 10
    --patience 3
    --val-fraction 0.1
    --run-name <optional>
```

### 2. `tools/training/eval_lora.py` — Evaluation Script

**Metrics per checkpoint:**
1. **Val contrastive loss** — same InfoNCE as training
2. **Retrieval R@K** — image->text and text->image R@1, R@5, R@10 on 500-pair val subset
3. **Nearest-neighbor gallery** — 50 diverse probe images (one per top tag), top-5 neighbors from base model vs LoRA model, saved as composite JPEGs

**Base model baseline:**
- Computed once at the start of evaluation
- Stored in eval/ directory
- All LoRA metrics displayed as deltas vs base

**Probe selection:**
- 50 images from different tag categories (body types, acts, positions, styles, etc.)
- No two probes from the same scene
- Fixed at run init for consistent comparison across epochs

**Output:** `eval/epoch_NNN.json` with all metrics + `eval/epoch_NNN_nn/` directory with gallery images

**Modes:**
- `--watch --run <id>` — watches for new checkpoints, evaluates automatically
- `--checkpoint <path> --run <id>` — evaluate specific checkpoint
- `--compare <run_a> <run_b>` — side-by-side run comparison

### 3. `tools/training/training_dashboard.py` + `.html` — Dashboard

**Architecture:** Threaded HTTP server serving static HTML + JSON API, polling state files. Same pattern as `caption_dashboard.py`.

**UI sections (top to bottom):**
1. **Header** — Gradient orb (pulsing when training), run selector dropdown, status pill
2. **Control bar** — Start/Stop/Pause buttons, LR slider with live adjustment
3. **Metric cards (6-grid)** — Train Loss, Val Loss, R@1, R@5, Learning Rate, VRAM
4. **GPU monitor strip** — Utilization %, VRAM bar, temperature, power draw (polled every 5s via nvidia-smi)
5. **Progress bar** — Epoch + step progress with ETA
6. **Charts row** — Loss curves (train vs val, canvas-drawn) | Retrieval R@K bars (base vs LoRA)
7. **Nearest-neighbor gallery** — Paginated: query | base top-5 | LoRA top-5, epoch selector
8. **Run history** — Past runs with summary metrics, clickable to browse

**Aesthetic:** Matches caption_dashboard.html exactly:
- Dark theme: `--bg-deep: #080a10`, `--surface: #12141f`
- Fonts: Outfit (display) + JetBrains Mono (metrics)
- Gradient primary: blue -> indigo -> purple
- Animated metric cards with stagger
- Pulsing orb for active state
- Glow effects on hover
- Status pills with animated dots

**API endpoints:**
- `GET /` — serve HTML
- `GET /api/status` — train_state.json + GPU stats + eval results list
- `GET /api/eval/<epoch>` — eval metrics for specific epoch
- `GET /api/gallery/<epoch>/<probe_idx>` — serve gallery image
- `GET /api/image/<path>` — serve dataset images for the gallery
- `GET /api/runs` — list all runs with summary
- `POST /api/control` — start/stop/pause/set-lr commands
- `GET /api/loss-history` — full train_log.jsonl for chart rendering

**Polling:** 2s during active training, 10s when idle.

## Directory Structure

```
tools/training/
├── __init__.py
├── train_lora.py
├── eval_lora.py
├── training_dashboard.py
├── training_dashboard.html
└── clip_dataset.py          # PyTorch Dataset for CSV + OpenCLIP preprocessing

assets/lora_training/
├── runs/
│   └── <run_id>/
│       ├── config.json
│       ├── control.json
│       ├── train_state.json
│       ├── train_log.jsonl
│       ├── checkpoints/
│       │   ├── epoch_001.pt
│       │   └── best.pt -> epoch_001.pt
│       └── eval/
│           ├── baseline.json
│           ├── epoch_001.json
│           ├── epoch_001_nn/
│           │   ├── probe_001.jpg
│           │   └── ...
│           └── probes.json     # Fixed probe image list
├── val_split.csv
└── train_split.csv
```

## Dependencies (new)

```
peft >= 0.12            # HuggingFace PEFT for LoRA injection
```

Already available: `open-clip-torch`, `torch`, `numpy`, `Pillow`.

## GPU Support

| GPU | VRAM | Batch Size | Grad Accum | Effective BS | Est. Time (5 epochs) |
|-----|------|-----------|------------|-------------|---------------------|
| RTX 4070 Super | 12 GB | 16 | 4 | 64 | ~2-3 hours |
| RTX 5090 | 32 GB | 64 | 1 | 64 | ~30-60 min |
| Cloud (A100 40GB) | 40 GB | 64 | 1 | 64 | ~20-40 min |

## Overfitting Prevention

1. **Early stopping** (patience=3) — primary guard
2. **LoRA rank constraint** — low rank (8) limits model capacity
3. **Weight decay** (0.01) — L2 regularization
4. **Dropout** (0.05) on LoRA layers
5. **Scene-level val split** — no data leakage between train/val
6. **Visual sanity checks** — nearest-neighbor galleries catch quality degradation
7. **Dashboard LR control** — manually reduce LR if loss curves diverge
8. **Base model comparison** — every eval shows delta vs base, easy to spot regression
