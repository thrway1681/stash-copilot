#!/usr/bin/env python3
"""Automated caption runner — selects diverse frames, captions via Gemini, tracks budget.

Phase 1: Load pre-computed CLIP embeddings, run SmartFrameSelector per scene.
Phase 2: Caption selected frames via Gemini API (single frame per call).
         Every call goes through ApiBudget for rate limiting + cost tracking.
Phase 3: Dashboard + checkpoint after each scene.

Usage:
    uv run python tools/dataset/caption_runner.py
    uv run python tools/dataset/caption_runner.py --limit 10 --max-cost 5.00
    uv run python tools/dataset/caption_runner.py --max-frames 256 --workers 5
    uv run python tools/dataset/caption_runner.py --dry-run
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable

from tools.dataset.gemini_api import caption_frame as gemini_caption_frame, CaptionResult
from tools.dataset.openrouter_api import caption_frame as openrouter_caption_frame
from tools.dataset.api_budget import ApiBudget, BudgetExhausted, DailyLimitReached, PRICING, count_tokens
from tools.dataset.frame_selector import (
    DEFAULT_ASSETS_DIR,
    DEFAULT_FRAMES_DIR,
    EmbeddingIndex,
    load_embedding_index,
    select_frames_for_scene,
)
from tools.dataset.constants import CAPTION_PROMPT as PROMPT
from tools.dataset.io_utils import dataset_image_name

# ── Constants ───────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "lora_dataset"
CHECKPOINT_FILE = "caption_progress.json"
DASHBOARD_INTERVAL = 10  # scenes between dashboard prints

# Error streak detection: if this many consecutive frames fail with non-
# content-moderation errors, assume the API is rate-limited and stop.
# Content moderation errors (blocked, prohibited) are excluded because
# they prove the API is responsive — just filtering content.
ERROR_STREAK_LIMIT = 20
_CONTENT_MOD_KEYWORDS = ("blocked", "prohibited", "content_filter")

# Model fallback chain: when RPD limit is hit, switch to the next model.
# Each model has its own independent rate limits on the Gemini API.
# Models containing "/" use the OpenRouter API; others use Gemini direct.
MODEL_FALLBACK_CHAIN: list[str] = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "google/gemini-3-flash-preview",     # OpenRouter fallback (same models, separate quota)
    "google/gemini-3.1-pro-preview",     # OpenRouter fallback
    "google/gemini-3-pro-preview",       # OpenRouter fallback
]


def _is_openrouter_model(model: str) -> bool:
    """OpenRouter models use the ``provider/model`` naming convention."""
    return "/" in model


# ── Scene result ─────────────────────────────────────────────────────────


@dataclass
class SceneResult:
    """Result from processing one scene."""
    image_names: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    errors: int = 0


# ── Scene processing ────────────────────────────────────────────────────


def _caption_one_frame(
    frame_path: Path,
    scene_id: str,
    images_dir: Path,
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    budget: ApiBudget,
    openrouter_key: str | None = None,
) -> tuple[str, str, bool]:
    """Caption a single frame: copy image, call API via budget gate, write .txt.

    Returns (image_name, caption_text, was_error).
    Skips API call if .txt already exists (idempotent).
    """
    image_name = dataset_image_name(scene_id, frame_path)
    dest_img = images_dir / image_name
    dest_txt = images_dir / image_name.replace(".jpg", ".txt")

    # Copy image if not already there
    if not dest_img.exists():
        shutil.copy2(frame_path, dest_img)

    # Skip if caption already exists (and is not an error from a previous run)
    if dest_txt.exists():
        existing = dest_txt.read_text(encoding="utf-8")
        if not existing.startswith("[ERROR"):
            return image_name, existing, False

    # Gate through budget (blocks until rate limits allow)
    budget.acquire()

    # Dispatch to the right API based on model name
    effective_key = openrouter_key if _is_openrouter_model(model) else api_key
    caption_fn = openrouter_caption_frame if _is_openrouter_model(model) else gemini_caption_frame

    try:
        result = caption_fn(
            base64.b64encode(frame_path.read_bytes()).decode("utf-8"),
            prompt, model, effective_key, temperature,
        )
        caption = result.caption
        # Record ACTUAL token usage from API response
        budget.record_usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        was_error = False
    except (BudgetExhausted, DailyLimitReached):
        raise  # propagate budget/rate stops to trigger fallback
    except Exception as e:
        caption = f"[ERROR: {e}]"
        budget.record_error()
        was_error = True
        _log(f"    ERROR {image_name}: {e}")

    dest_txt.write_text(caption, encoding="utf-8")
    # Flush budget to disk after every frame so the dashboard stays current
    budget.save_state()
    return image_name, caption, was_error


def process_scene_frames(
    scene_id: str,
    frame_paths: list[str],
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    images_dir: Path,
    budget: ApiBudget,
    workers: int = 10,
    on_frame_done: Callable[[str, str, bool], None] | None = None,
    openrouter_key: str | None = None,
) -> SceneResult:
    """Caption a list of frame paths for one scene using a thread pool.

    Returns SceneResult with image_names, captions, and error count.
    ``on_frame_done`` is called after each frame completes (for live progress).
    """
    results: dict[str, tuple[str, str, bool]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _caption_one_frame,
                Path(fp), scene_id, images_dir, prompt, model, api_key,
                temperature, budget, openrouter_key,
            ): fp
            for fp in frame_paths
        }
        budget_stop: Exception | None = None
        for future in as_completed(futures):
            fp = futures[future]
            try:
                results[fp] = future.result()
            except (BudgetExhausted, DailyLimitReached) as e:
                # Cancel remaining futures and break out
                budget_stop = e
                for f in futures:
                    f.cancel()
                break
            except Exception as e:
                img_name = dataset_image_name(scene_id, Path(fp))
                results[fp] = (img_name, f"[ERROR: {e}]", True)
            if on_frame_done:
                img_name, caption, was_error = results[fp]
                on_frame_done(img_name, caption, was_error)

    if budget_stop is not None:
        raise budget_stop

    # Return in original frame order
    scene_result = SceneResult()
    for fp in frame_paths:
        img_name, caption, was_error = results[fp]
        scene_result.image_names.append(img_name)
        scene_result.captions.append(caption)
        if was_error:
            scene_result.errors += 1

    return scene_result


# ── Checkpoint ──────────────────────────────────────────────────────────


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed_scenes": [], "total_frames_captioned": 0, "errors": 0}


def _save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    data["last_updated"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Metadata ────────────────────────────────────────────────────────────


def _append_metadata(
    jsonl_path: Path,
    scene_id: int,
    image_names: list[str],
    captions: list[str],
    selection_stats: dict[str, Any],
) -> None:
    record = {
        "scene_id": scene_id,
        "image_names": image_names,
        "captions": {name: cap for name, cap in zip(image_names, captions)},
        "selection": selection_stats,
        "captioned_at": datetime.now(UTC).isoformat(),
        "method": "gemini-vlm+smart-select",
    }
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Main loop ───────────────────────────────────────────────────────────


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    model: str = "gemini-3-flash-preview",
    temperature: float = 1.0,
    api_key: str | None = None,
    openrouter_key: str | None = None,
    skip_gemini_fallback: bool = False,
    max_frames: int = 20,
    workers: int = 10,
    limit: int | None = None,
    max_cost: float | None = None,
    dry_run: bool = False,
) -> None:
    """Run the caption pipeline: select frames then caption via Gemini or OpenRouter."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY")

    # Need at least one key for the starting model
    if not dry_run:
        if _is_openrouter_model(model) and not openrouter_key:
            _log("ERROR: No OpenRouter API key. Set OPENROUTER_API_KEY env var or pass --openrouter-key.")
            sys.exit(1)
        elif not _is_openrouter_model(model) and not api_key:
            _log("ERROR: No Gemini API key. Set GEMINI_API_KEY env var or pass --api-key.")
            sys.exit(1)

    # Phase 1: Load embeddings
    index = load_embedding_index(assets_dir)
    all_scenes = index.scene_id_list

    # Load checkpoint to skip completed scenes
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_FILE
    checkpoint = _load_checkpoint(checkpoint_path)
    completed = set(checkpoint["completed_scenes"])

    # Filter to pending scenes
    pending = [s for s in all_scenes if s not in completed]
    if limit:
        pending = pending[:limit]

    # Count total frames to caption
    total_frames = 0
    for sid in pending:
        start, end = index._scene_ranges[sid]
        n = end - start
        total_frames += min(n, max_frames)

    # ── Measure real token cost (not estimate) ──────────────────────────

    budget = ApiBudget(
        model=model,
        rpm_limit=900,     # safety margin below 1,000
        tpm_limit=900_000, # safety margin below 1,000,000
        rpd_limit=9_500,   # safety margin below 10,000
        max_cost=max_cost,
        state_file=output_dir / "budget_state.json",
    )
    budget._total_frames = total_frames

    if not dry_run and api_key and not _is_openrouter_model(model):
        # countTokens is a Gemini-only API — skip for OpenRouter models
        sample_scene = pending[0] if pending else all_scenes[0]
        sample_start, _ = index._scene_ranges[sample_scene]
        sample_fidx = index.frame_indices[sample_start]
        sample_frame = frames_dir / f"scene_{sample_scene}" / f"frame_{int(sample_fidx):04d}.jpg"

        if sample_frame.exists():
            _log("Measuring prompt token count via countTokens API...")
            sample_b64 = base64.b64encode(sample_frame.read_bytes()).decode("utf-8")
            measured = count_tokens(model, api_key, PROMPT, sample_b64)
            budget.measured_input_tokens_per_call = measured
            _log(f"  Measured input tokens: {measured:,} per call")

            # Show cost estimate before starting
            est_cost = budget.estimate_total_cost(total_frames)
            _log(f"  Estimated total cost:  ${est_cost:,.2f} "
                 f"(for {total_frames:,} frames)")
            if max_cost:
                _log(f"  Budget cap:            ${max_cost:,.2f}")
            _log("")

    # Build fallback chain starting from the requested model
    if model in MODEL_FALLBACK_CHAIN:
        fallback_idx = MODEL_FALLBACK_CHAIN.index(model)
        fallback_models = MODEL_FALLBACK_CHAIN[fallback_idx:]
    else:
        fallback_models = [model]

    # When skip_gemini_fallback is set, remove direct Gemini models from the
    # fallback tail so the runner jumps straight to OpenRouter on rate limit.
    if skip_gemini_fallback:
        fallback_models = [fallback_models[0]] + [
            m for m in fallback_models[1:] if _is_openrouter_model(m)
        ]

    provider_label = "OpenRouter" if _is_openrouter_model(model) else "Gemini"
    _log(f"Caption Runner ({provider_label} API)")
    _log(f"  Model:       {model}")
    if len(fallback_models) > 1:
        _log(f"  Fallback:    {' -> '.join(fallback_models[1:])}")
    else:
        _log(f"  Fallback:    none")
    _log(f"  OpenRouter:  {'key set' if openrouter_key else 'no key (OpenRouter fallbacks disabled)'}")
    if skip_gemini_fallback:
        _log(f"  Skip Gemini: yes (jumping straight to OpenRouter on rate limit)")
    _log(f"  Temperature: {temperature}")
    _log(f"  Max frames:  {max_frames}/scene")
    _log(f"  Workers:     {workers}")
    _log(f"  Scenes:      {len(pending)} pending ({len(completed)} done)")
    _log(f"  Total frames:{total_frames:,}")
    _log(f"  Dry run:     {dry_run}")
    _log("")

    jsonl_path = output_dir / "metadata.jsonl"
    total_captioned = 0
    total_errors = 0
    start_time = time.monotonic()

    # ── Fill uncaptioned frames first (.jpg without .txt) ─────────
    # Repairs gaps from deleted errors, interrupted runs, or frame
    # selection differences.  Runs BEFORE new scenes so repairs are
    # prioritised over fresh work.
    # NOTE: Skipped when max_frames is low (e.g. 20) because the images/
    # dir may contain hundreds of thousands of orphan .jpg files from
    # previous runs with higher max_frames.  The main scene loop handles
    # partial scenes correctly by skipping existing .txt files.
    if not dry_run and max_frames > 100:
        uncaptioned = _find_uncaptioned_frames(images_dir)
        if uncaptioned:
            n_orphans = sum(len(v) for v in uncaptioned.values())
            _log(f"  Filling {n_orphans} uncaptioned frames across "
                 f"{len(uncaptioned)} scenes...\n")

            fill_fixed = 0
            fill_errors = 0
            fill_stopped = False

            effective_key = openrouter_key if _is_openrouter_model(model) else api_key
            caption_fn = openrouter_caption_frame if _is_openrouter_model(model) else gemini_caption_frame

            for sid, orphan_jpgs in sorted(uncaptioned.items()):
                if fill_stopped:
                    break

                _log(f"    Scene {sid}: {len(orphan_jpgs)} frames")

                def _fill_one(
                    img_p: Path, fn: Any = caption_fn,
                    key: str = effective_key, mdl: str = model,
                ) -> str:
                    budget.acquire()
                    result = fn(
                        base64.b64encode(img_p.read_bytes()).decode("utf-8"),
                        PROMPT, mdl, key, temperature,
                    )
                    budget.record_usage(
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                    )
                    budget.save_state()
                    return result.caption

                try:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {
                            pool.submit(_fill_one, jpg): jpg
                            for jpg in orphan_jpgs
                        }
                        budget_stop: Exception | None = None
                        for fut in as_completed(futures):
                            jpg = futures[fut]
                            try:
                                caption = fut.result()
                                jpg.with_suffix(".txt").write_text(caption, encoding="utf-8")
                                fill_fixed += 1
                                total_captioned += 1
                                _log(f"      FILLED {jpg.name}")
                                _log(f"FRAME_DONE {jpg.name} OK {caption[:200].replace(chr(10), ' ')}")
                            except (BudgetExhausted, DailyLimitReached) as e:
                                budget_stop = e
                                for f in futures:
                                    f.cancel()
                                break
                            except Exception as e:
                                jpg.with_suffix(".txt").write_text(f"[ERROR: {e}]", encoding="utf-8")
                                budget.record_error()
                                fill_errors += 1
                                total_errors += 1
                                checkpoint["errors"] = checkpoint.get("errors", 0) + 1
                                _log(f"      FILL ERROR {jpg.name}: {e}")
                                _log(f"FRAME_DONE {jpg.name} ERROR {str(e)[:200].replace(chr(10), ' ')}")

                        if budget_stop is not None:
                            raise budget_stop

                except (BudgetExhausted, DailyLimitReached) as e:
                    _log(f"\n  Fill stopped: {e}")
                    fill_stopped = True

                _save_checkpoint(checkpoint_path, checkpoint)

            _log(f"\n  Fill complete: {fill_fixed} captioned, {fill_errors} errors"
                 + (", stopped (budget)" if fill_stopped else ""))

            if fill_stopped:
                _log(f"\n{budget.dashboard()}")
                elapsed = time.monotonic() - start_time
                _log(f"\nDone (fill only): {total_captioned:,} frames in {elapsed:.0f}s")
                budget.save_state()
                return
            _log("")

    scene_iter = iter(enumerate(pending))
    stopped = False
    error_streak = 0  # consecutive non-content-mod frame errors

    while not stopped:
        try:
            for i, scene_id in scene_iter:
                # Select frames
                selections = select_frames_for_scene(
                    index, scene_id, max_frames=max_frames, frames_dir=frames_dir,
                )
                if not selections:
                    _log(f"  [{i+1}/{len(pending)}] Scene {scene_id}: no frames, skipping")
                    continue

                frame_paths = [s.path for s in selections]
                n_frames = len(frame_paths)

                from stash_ai.tasks.smart_frame_selector import SmartFrameSelector
                stats = SmartFrameSelector().get_selection_stats(selections)

                _log(f"  [{i+1}/{len(pending)}] Scene {scene_id} "
                     f"({n_frames} selected, {stats['novelty_count']} novelty)")

                if dry_run:
                    _log(f"    [dry-run] Would caption {n_frames} frames")
                    continue

                # Track frames completed within this scene for live progress
                frames_done_in_scene = 0

                def _on_frame_done(img_name: str, caption: str, was_error: bool) -> None:
                    nonlocal frames_done_in_scene, error_streak
                    frames_done_in_scene += 1
                    if was_error:
                        # Content moderation errors prove the API is alive
                        is_content_mod = any(
                            kw in caption.lower() for kw in _CONTENT_MOD_KEYWORDS
                        )
                        if is_content_mod:
                            error_streak = 0
                        else:
                            error_streak += 1
                    else:
                        error_streak = 0
                    # Flush checkpoint with in-progress frame count
                    checkpoint["total_frames_captioned"] = (
                        base_frames_captioned + frames_done_in_scene
                    )
                    _save_checkpoint(checkpoint_path, checkpoint)
                    # Structured frame event for dashboard streaming
                    status = "ERROR" if was_error else "OK"
                    # Truncate caption to 200 chars to keep log lines bounded
                    cap_preview = caption[:200].replace("\n", " ")
                    _log(f"FRAME_DONE {img_name} {status} {cap_preview}")

                base_frames_captioned = checkpoint.get("total_frames_captioned", 0)
                scene_start = time.monotonic()
                scene_result = process_scene_frames(
                    scene_id=str(scene_id),
                    frame_paths=frame_paths,
                    prompt=PROMPT,
                    model=model,
                    api_key=api_key,  # type: ignore[arg-type]
                    temperature=temperature,
                    images_dir=images_dir,
                    budget=budget,
                    workers=workers,
                    on_frame_done=_on_frame_done,
                    openrouter_key=openrouter_key,
                )
                scene_time = time.monotonic() - scene_start

                total_captioned += len(scene_result.image_names)
                total_errors += scene_result.errors

                _append_metadata(
                    jsonl_path, scene_id, scene_result.image_names,
                    scene_result.captions, stats,
                )

                checkpoint["completed_scenes"].append(scene_id)
                checkpoint["errors"] = checkpoint.get("errors", 0) + scene_result.errors
                _save_checkpoint(checkpoint_path, checkpoint)

                _log(f"    {len(scene_result.image_names)} captions in {scene_time:.1f}s"
                     + (f" ({scene_result.errors} errors)" if scene_result.errors else ""))

                # Detect sustained non-content-mod errors → likely rate limited
                if error_streak >= ERROR_STREAK_LIMIT:
                    _log(f"\n  RATE LIMITED: {error_streak} consecutive API errors "
                         f"(excluding content moderation)")
                    _log(f"  Stopping to avoid wasting budget.\n")
                    stopped = True
                    break

                # Dashboard every N scenes
                if (i + 1) % DASHBOARD_INTERVAL == 0:
                    _log(f"\n{budget.dashboard()}\n")

            # Reached end of pending scenes normally
            stopped = True

        except BudgetExhausted as e:
            _log(f"\n  STOPPED: {e}")
            _log(f"  Increase --max-cost to continue.\n")
            stopped = True

        except DailyLimitReached as e:
            # Try fallback to next model
            fallback_models = fallback_models[1:]
            # Skip OpenRouter models if no key is available
            while fallback_models and _is_openrouter_model(fallback_models[0]) and not openrouter_key:
                _log(f"  Skipping {fallback_models[0]} (no OpenRouter key)")
                fallback_models = fallback_models[1:]
            if not fallback_models:
                _log(f"\n  STOPPED: {e}")
                _log(f"  All fallback models exhausted. Resume tomorrow.\n")
                stopped = True
            else:
                next_model = fallback_models[0]
                provider_label = "OpenRouter" if _is_openrouter_model(next_model) else "Gemini"
                _log(f"\n  RPD limit hit for {model}: {e}")
                _log(f"  Switching to fallback model: {next_model} ({provider_label})")
                model = next_model
                budget.model = model
                budget.pricing = PRICING.get(model, budget.pricing)
                # Reset RPD counter — new model has its own quota
                budget._rpd_count = 0
                budget._rpd_date = date.today().isoformat()
                _log(f"  New pricing: ${budget.pricing.input_per_m}/M input, "
                     f"${budget.pricing.output_per_m}/M output")
                _log(f"  Continuing...\n")

    # Final dashboard
    _log(f"\n{budget.dashboard()}")

    elapsed = time.monotonic() - start_time
    _log(f"\nDone: {total_captioned:,} frames across {len(pending)} scenes in {elapsed:.0f}s")
    if total_errors:
        _log(f"  Errors: {total_errors}")

    budget.save_state()


def _find_error_captions(images_dir: Path) -> list[Path]:
    """Find all .txt files in images_dir that contain error captions."""
    errors: list[Path] = []
    for txt in sorted(images_dir.glob("*.txt")):
        try:
            content = txt.read_text(encoding="utf-8")
            if content.startswith("[ERROR"):
                errors.append(txt)
        except OSError:
            continue
    return errors


def _find_uncaptioned_frames(images_dir: Path) -> dict[int, list[Path]]:
    """Find .jpg files in images_dir that lack a .txt companion.

    Uses os.scandir for fast enumeration over 300K+ files.
    Returns {scene_id: [jpg_paths]} for scenes with gaps.
    """
    jpg_stems: set[str] = set()
    txt_stems: set[str] = set()
    for entry in os.scandir(images_dir):
        name = entry.name
        if name.startswith("s") and "_f" in name:
            if name.endswith(".jpg"):
                jpg_stems.add(name[:-4])
            elif name.endswith(".txt"):
                txt_stems.add(name[:-4])

    orphan_stems = sorted(jpg_stems - txt_stems)
    if not orphan_stems:
        return {}

    incomplete: dict[int, list[Path]] = {}
    for stem in orphan_stems:
        try:
            sid = int(stem.split("_f")[0][1:])
        except (ValueError, IndexError):
            continue
        incomplete.setdefault(sid, []).append(images_dir / (stem + ".jpg"))
    return incomplete


def reset_errors(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Delete all error .txt files and reset the error count in the checkpoint.

    This restores the state as if errored frames were never attempted,
    so the next ``run()`` will detect and re-caption them.
    """
    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        _log("No images directory found — nothing to reset.")
        return

    _log("Scanning for error captions...")
    error_files = _find_error_captions(images_dir)
    if not error_files:
        _log("No error captions found.")
        return

    _log(f"  Found {len(error_files)} error files — deleting...")
    deleted = 0
    for txt in error_files:
        try:
            txt.unlink()
            deleted += 1
        except OSError:
            pass

    _log(f"  Deleted {deleted} error files.")

    # Reset error count in checkpoint
    checkpoint_path = output_dir / CHECKPOINT_FILE
    checkpoint = _load_checkpoint(checkpoint_path)
    old_errors = checkpoint.get("errors", 0)
    checkpoint["errors"] = 0
    _save_checkpoint(checkpoint_path, checkpoint)
    _log(f"  Reset checkpoint errors: {old_errors} → 0")
    _log(f"  Next run() will detect and re-caption the {deleted} uncaptioned frames.")


def retry_errors(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str = "gemini-3-flash-preview",
    temperature: float = 1.0,
    api_key: str | None = None,
    openrouter_key: str | None = None,
    skip_gemini_fallback: bool = False,
    workers: int = 10,
    max_cost: float | None = None,
) -> None:
    """Retry all errored captions in the images directory.

    Uses the same model fallback chain, skip-gemini logic, and concurrent
    workers as the main pipeline.

    Usage:
        uv run python tools/dataset/caption_runner.py --retry-errors
        uv run python tools/dataset/caption_runner.py --retry-errors --workers 15 --max-cost 1.00
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY")

    if _is_openrouter_model(model) and not openrouter_key:
        _log("ERROR: No OpenRouter API key. Set OPENROUTER_API_KEY env var or pass --openrouter-key.")
        sys.exit(1)
    elif not _is_openrouter_model(model) and not api_key:
        _log("ERROR: No API key. Set GEMINI_API_KEY env var or pass --api-key.")
        sys.exit(1)

    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        _log(f"No images directory at {images_dir}")
        return

    error_files = _find_error_captions(images_dir)
    if not error_files:
        _log("No errored captions found — nothing to retry.")
        return

    budget = ApiBudget(
        model=model,
        rpm_limit=900,
        tpm_limit=900_000,
        rpd_limit=9_500,
        max_cost=max_cost,
        state_file=output_dir / "budget_state.json",
    )

    # Build fallback chain (same logic as run())
    if model in MODEL_FALLBACK_CHAIN:
        fallback_idx = MODEL_FALLBACK_CHAIN.index(model)
        fallback_models = MODEL_FALLBACK_CHAIN[fallback_idx:]
    else:
        fallback_models = [model]
    if skip_gemini_fallback:
        fallback_models = [fallback_models[0]] + [
            m for m in fallback_models[1:] if _is_openrouter_model(m)
        ]
    # Skip OpenRouter models if no key available
    while len(fallback_models) > 1 and _is_openrouter_model(fallback_models[0]) and not openrouter_key:
        fallback_models = fallback_models[1:]

    provider_label = "OpenRouter" if _is_openrouter_model(model) else "Gemini"
    _log(f"Retry Errors Mode ({provider_label} API)")
    _log(f"  Model:       {model}")
    if len(fallback_models) > 1:
        _log(f"  Fallback:    {' -> '.join(fallback_models[1:])}")
    _log(f"  Workers:     {workers}")
    _log(f"  Temperature: {temperature}")
    _log(f"  Errors found:{len(error_files)}")
    if skip_gemini_fallback:
        _log(f"  Skip Gemini: yes")
    if max_cost:
        _log(f"  Budget cap:  ${max_cost:.2f}")
    _log("")

    # Load checkpoint so we can keep its error count in sync as we fix frames
    checkpoint_path = output_dir / CHECKPOINT_FILE
    checkpoint = _load_checkpoint(checkpoint_path)

    start_time = time.monotonic()
    fixed = 0
    still_failed = 0
    error_iter = iter(enumerate(error_files))
    stopped = False
    error_streak = 0  # consecutive non-content-mod frame errors

    while not stopped:
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures: dict[Any, tuple[int, Path, Path]] = {}
                for i, txt_path in error_iter:
                    img_name = txt_path.name.replace(".txt", ".jpg")
                    img_path = images_dir / img_name
                    if not img_path.exists():
                        _log(f"  [{i+1}/{len(error_files)}] SKIP {img_name} (image missing)")
                        continue

                    effective_key = openrouter_key if _is_openrouter_model(model) else api_key
                    caption_fn = openrouter_caption_frame if _is_openrouter_model(model) else gemini_caption_frame

                    def _retry_one(
                        img_p: Path, txt_p: Path, fn: Any = caption_fn,
                        key: str = effective_key, mdl: str = model,
                    ) -> str:
                        budget.acquire()
                        result = fn(
                            base64.b64encode(img_p.read_bytes()).decode("utf-8"),
                            PROMPT, mdl, key, temperature,
                        )
                        budget.record_usage(
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                        )
                        budget.save_state()
                        return result.caption

                    fut = pool.submit(_retry_one, img_path, txt_path)
                    futures[fut] = (i, txt_path, img_path)

                budget_stop: Exception | None = None
                for fut in as_completed(futures):
                    idx, txt_path, img_path = futures[fut]
                    img_name = txt_path.name.replace(".txt", ".jpg")
                    try:
                        caption = fut.result()
                        txt_path.write_text(caption, encoding="utf-8")
                        fixed += 1
                        error_streak = 0
                        # Keep checkpoint error count in sync with disk
                        checkpoint["errors"] = max(0, checkpoint.get("errors", 0) - 1)
                        if fixed % 50 == 0:
                            _save_checkpoint(checkpoint_path, checkpoint)
                        _log(f"  [{idx+1}/{len(error_files)}] FIXED {img_name}")
                        _log(f"FRAME_DONE {img_name} OK {caption[:200].replace(chr(10), ' ')}")
                    except (BudgetExhausted, DailyLimitReached) as e:
                        budget_stop = e
                        for f in futures:
                            f.cancel()
                        break
                    except Exception as e:
                        still_failed += 1
                        err_str = str(e).lower()
                        is_content_mod = any(
                            kw in err_str for kw in _CONTENT_MOD_KEYWORDS
                        )
                        if is_content_mod:
                            error_streak = 0
                        else:
                            error_streak += 1
                        old_error = txt_path.read_text(encoding="utf-8").strip()
                        new_error = f"[ERROR: {e}]"
                        if old_error != new_error:
                            txt_path.write_text(new_error, encoding="utf-8")
                        budget.record_error()
                        _log(f"  [{idx+1}/{len(error_files)}] FAILED {img_name}: {e}")
                        _log(f"FRAME_DONE {img_name} ERROR {str(e)[:200].replace(chr(10), ' ')}")

                if budget_stop is not None:
                    raise budget_stop

                # Detect sustained non-content-mod errors → likely rate limited
                if error_streak >= ERROR_STREAK_LIMIT:
                    _log(f"\n  RATE LIMITED: {error_streak} consecutive API errors "
                         f"(excluding content moderation)")
                    _log(f"  Stopping to avoid wasting budget.\n")
                    stopped = True

            # Finished all error files normally (or rate limited)
            stopped = True

        except BudgetExhausted as e:
            _log(f"\n  STOPPED: {e}")
            stopped = True

        except DailyLimitReached as e:
            fallback_models = fallback_models[1:]
            while fallback_models and _is_openrouter_model(fallback_models[0]) and not openrouter_key:
                fallback_models = fallback_models[1:]
            if not fallback_models:
                _log(f"\n  STOPPED: {e}")
                _log(f"  All fallback models exhausted.\n")
                stopped = True
            else:
                next_model = fallback_models[0]
                provider = "OpenRouter" if _is_openrouter_model(next_model) else "Gemini"
                _log(f"\n  RPD limit hit for {model}: {e}")
                _log(f"  Switching to fallback model: {next_model} ({provider})")
                model = next_model
                budget.model = model
                budget.pricing = PRICING.get(model, budget.pricing)
                budget._rpd_count = 0
                budget._rpd_date = date.today().isoformat()
                _log(f"  Continuing...\n")

    elapsed = time.monotonic() - start_time
    budget.save_state()
    _save_checkpoint(checkpoint_path, checkpoint)

    _log(f"\nRetry complete in {elapsed:.1f}s: {fixed} fixed, {still_failed} still errored")
    remaining = len(error_files) - fixed - still_failed
    if remaining > 0:
        _log(f"  {remaining} not attempted (budget exhausted)")
    _log(f"\n{budget.dashboard()}")


_log_file: Any = None


def _init_log_file(output_dir: Path) -> None:
    """Open a persistent log file alongside the runner output."""
    global _log_file
    log_path = output_dir / "runner.log"
    _log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    if _log_file is not None:
        try:
            _log_file.write(msg + "\n")
            _log_file.flush()
        except OSError:
            pass


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select diverse frames then caption via Gemini / OpenRouter VLM.",
    )
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="Model name (default: gemini-3-flash-preview). "
                             "Use provider/model format for OpenRouter (e.g. google/gemini-3-flash-preview)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--api-key",
                        default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                        help="Gemini API key (default: GEMINI_API_KEY env var)")
    parser.add_argument("--openrouter-key",
                        default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key for fallback (default: OPENROUTER_API_KEY env var)")
    parser.add_argument("--max-frames", type=int, default=20,
                        help="Max frames per scene via SmartFrameSelector (default: 20)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent API calls per scene (default: 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max scenes to process (default: all)")
    parser.add_argument("--max-cost", type=float, default=None,
                        help="Budget cap in USD — stops when reached (default: no cap)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview frame selection without API calls")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Only retry errored captions, skip main pipeline")
    parser.add_argument("--skip-gemini-fallback", action="store_true",
                        help="Skip direct Gemini fallback models, jump straight to OpenRouter on rate limit")
    parser.add_argument("--reset-errors", action="store_true",
                        help="Delete all error captions and reset error count, then exit. "
                             "Next run() will re-caption the uncaptioned frames.")
    args = parser.parse_args()

    _init_log_file(DEFAULT_OUTPUT_DIR)

    if args.reset_errors:
        reset_errors()
        return

    if args.retry_errors:
        retry_errors(
            model=args.model,
            temperature=args.temperature,
            api_key=args.api_key,
            openrouter_key=args.openrouter_key,
            skip_gemini_fallback=args.skip_gemini_fallback,
            workers=args.workers,
            max_cost=args.max_cost,
        )
    else:
        run(
            model=args.model,
            temperature=args.temperature,
            api_key=args.api_key,
            openrouter_key=args.openrouter_key,
            skip_gemini_fallback=args.skip_gemini_fallback,
            max_frames=args.max_frames,
            workers=args.workers,
            limit=args.limit,
            max_cost=args.max_cost,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
