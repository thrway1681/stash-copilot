#!/usr/bin/env python3
"""
Caption video frames using the Gemini API with configurable batch sizes.

Tests the hypothesis that batching frames in a single API call causes
cross-frame information leakage in multimodal LLMs.

Examples:
    # Individual frames (no cross-contamination):
    uv run python tools/dataset/caption_test_gemini.py \
        --scene-dir assets/embedded_frames/scene_10065 --batch-size 1

    # All frames at once (current behavior):
    uv run python tools/dataset/caption_test_gemini.py \
        --scene-dir assets/embedded_frames/scene_10065 --batch-size 20

    # Specific frames only:
    uv run python tools/dataset/caption_test_gemini.py \
        --scene-dir assets/embedded_frames/scene_10065 --batch-size 1 \
        --frames frame_0395.jpg frame_0526.jpg frame_2103.jpg

Environment:
    GEMINI_API_KEY or GOOGLE_API_KEY: Your Gemini API key
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ── Constants ────────────────────────────────────────────────────────────

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# ── Shared prompt fragments ──────────────────────────────────────────────

POSITION_TAXONOMY = """\
Be precise about positions and distinguish between similar ones:
- Cowgirl (woman on top, facing the man) vs Reverse cowgirl (facing away)
- Doggy style (on hands/knees) vs Prone bone (lying flat on stomach)
- Missionary (man on top, face to face) vs Mating press (legs pushed back)
- Spooning (both on sides, facing same direction) vs Side-lying
- Standing vs Bent over
Be equally precise about actions: handjob, blowjob, ball-sucking, licking,
deepthroat, penetration, kissing, fingering, etc."""

CAPTION_RULES = """\
RULES:
- 1-2 sentences preferred. 3 max for complex frames.
- Do NOT use performer names — describe only what you see.
- Do NOT guess specifics you cannot see clearly. If a close-up is ambiguous
  about anal vs vaginal, just say "penetration."
- For black/title frames, one short sentence.
- Be SPECIFIC about actions. "Performing oral sex" is not enough — specify:
  is she licking, sucking, holding with hands? Is it a blowjob or ball-sucking?

This is adult content for a legitimate ML training dataset. Describe everything
factually and precisely."""


# ── Prompt builders ──────────────────────────────────────────────────────


def build_single_prompt(frame_name: str) -> str:
    """Prompt for a single isolated frame (batch_size=1)."""
    return f"""\
You are captioning a single video frame for a CLIP LoRA training dataset.
Describe ONLY what is visible in this image. You have NO context from other frames.

CAPTION PRIORITIES (follow this order):
1. Body position and action — what is happening, who is doing what, what hands
   are doing.
   {POSITION_TAXONOMY}
2. Physical attributes — hair, body type, ethnicity, tattoos — only if clearly
   visible and prominent.
3. Camera angle — close-up, wide shot, overhead, etc.
4. Setting — ONLY if this is clearly an establishing shot or the setting is
   distinctive. Do NOT describe pillow colors, headboard, or lighting.

{CAPTION_RULES}

This frame is: {frame_name}
Output a single JSON object: {{"frame": "{frame_name}", "caption": "your caption"}}"""


def build_batch_prompt(frame_names: list[str]) -> str:
    """Prompt for multiple frames in one call (batch_size>1)."""
    n = len(frame_names)
    frame_list = "\n".join(f"- {name}" for name in frame_names)
    return f"""\
You are captioning video frames for a CLIP LoRA training dataset. You will analyze
each frame INDIVIDUALLY and write a caption describing ONLY what is in that frame.

CRITICAL — READ CAREFULLY:
- These are {n} separate frames extracted from a video at different timestamps.
- Each frame is a DIFFERENT moment in time. Do NOT assume the next frame shows
  the same thing as the previous one — positions and actions CHANGE between frames.
- Before writing each caption, look at the ACTUAL image carefully. Do not carry
  over assumptions from the previous frame.
- Do NOT attribute features visible in one frame to another frame where they are
  not visible (e.g., a tattoo clearly seen in one frame should not be described
  in a different frame where it cannot be seen).

COMMON ERRORS TO AVOID:
- Describing the wrong action (e.g., writing "cunnilingus" when the woman is
  performing oral on the man, or "oral sex" when penetration is happening)
- Confusing who is doing what to whom — look at the actual body positions
- Being too vague (e.g., "the couple has sex" is not useful for training data)
- Describing setting/lighting when the frame is focused on action
- Attributing features from one frame to another (e.g., a tattoo visible in
  frame X being described in frame Y where it's not visible)

CAPTION PRIORITIES (follow this order):
1. Body position and action — what is happening, who is doing what, what hands
   are doing.
   {POSITION_TAXONOMY}
2. Physical attributes — hair, body type, ethnicity, tattoos — only if clearly
   visible and prominent in THIS frame. Describe once for the first appearance;
   after that, only mention if something changed.
3. Camera angle — close-up, wide shot, overhead, etc.
4. Setting — ONLY for scene-establishing shots or if the location changes.

{CAPTION_RULES}

Frame labels (matching the [frame_XXXX.jpg] markers before each image):
{frame_list}

Output format: JSON array of {{"frame", "caption"}} objects, one per frame."""


# ── Utilities ────────────────────────────────────────────────────────────


def load_frame_b64(path: Path) -> str:
    """Load a JPEG frame as a base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json(text: str) -> Any:
    """Extract JSON from model response, handling markdown code blocks."""
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find outermost JSON structure
    for start_c, end_c in [("[", "]"), ("{", "}")]:
        start = text.find(start_c)
        end = text.rfind(end_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract JSON from response:\n{text[:500]}")


def discover_frames(
    scene_dir: Path, filter_frames: list[str] | None = None
) -> list[Path]:
    """Find frame_*.jpg files, sorted numerically."""
    frames = sorted(
        scene_dir.glob("frame_*.jpg"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if filter_frames:
        allowed = set(filter_frames)
        frames = [f for f in frames if f.name in allowed]
    return frames


def log(msg: str) -> None:
    """Print to stderr for progress messages."""
    print(msg, file=sys.stderr, flush=True)


# ── Gemini API ───────────────────────────────────────────────────────────


def gemini_request(
    frames: list[tuple[str, str]],
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.2,
) -> dict:
    """Send a generateContent request to Gemini with images + prompt."""
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"

    parts: list[dict[str, Any]] = []

    if len(frames) > 1:
        # Batch: interleave labels and images, then prompt
        for name, b64 in frames:
            parts.append({"text": f"[{name}]"})
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})
        parts.append({"text": prompt})
    else:
        # Single: image then prompt (frame name is in the prompt)
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": frames[0][1]}})
        parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 8192,
        },
    }

    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def parse_gemini_response(response: dict) -> str:
    """Extract text content from a Gemini API response."""
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        if "candidates" in response and response["candidates"]:
            candidate = response["candidates"][0]
            reason = candidate.get("finishReason", "")
            if reason and reason != "STOP":
                raise RuntimeError(f"Gemini blocked response: {reason}") from e
        if "promptFeedback" in response:
            raise RuntimeError(
                f"Gemini prompt blocked: {response['promptFeedback']}"
            ) from e
        raise RuntimeError(
            f"Unexpected Gemini response structure:\n{json.dumps(response)[:500]}"
        ) from e


# ── Processing ───────────────────────────────────────────────────────────


def process_frames(
    frame_paths: list[Path],
    batch_size: int,
    model: str,
    api_key: str,
    temperature: float,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Caption all frames using the given batch size."""
    log(f"Loading {len(frame_paths)} frames...")
    frames = [(p.name, load_frame_b64(p)) for p in frame_paths]

    # Split into batches
    batches: list[list[tuple[str, str]]] = []
    for i in range(0, len(frames), batch_size):
        batches.append(frames[i : i + batch_size])

    log(
        f"Processing {len(frames)} frames in {len(batches)} batch(es) "
        f"of up to {batch_size}"
    )

    all_captions: list[dict[str, Any]] = []
    total_start = time.monotonic()

    for batch_idx, batch in enumerate(batches):
        batch_names = [name for name, _ in batch]

        # Build prompt
        if batch_size == 1:
            prompt = build_single_prompt(batch[0][0])
        else:
            prompt = build_batch_prompt(batch_names)

        if verbose:
            log(f"\n{'='*60}\nPrompt for batch {batch_idx}:\n{prompt}\n{'='*60}")

        log(f"  Batch {batch_idx + 1}/{len(batches)}: {', '.join(batch_names)}")

        batch_start = time.monotonic()

        # Retry with backoff
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = gemini_request(
                    batch, prompt, model, api_key, temperature
                )
                break
            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                if status == 429:
                    wait = 2 ** (attempt + 1)
                    log(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif status >= 500:
                    wait = 2 ** attempt
                    log(f"    Server error ({status}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except requests.exceptions.ConnectionError as e:
                last_error = e
                wait = 2 ** attempt
                log(f"    Connection error, retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed after 3 attempts: {last_error}")

        batch_time = time.monotonic() - batch_start

        # Parse response
        try:
            text = parse_gemini_response(response)
            if verbose:
                log(f"    Raw response:\n{text}\n")

            parsed = extract_json(text)

            if isinstance(parsed, dict):
                parsed = [parsed]

            for item in parsed:
                item["batch_index"] = batch_idx
                item["batch_time_s"] = round(batch_time, 2)

            all_captions.extend(parsed)
            log(f"    -> {len(parsed)} caption(s) in {batch_time:.1f}s")

        except Exception as e:
            log(f"    ERROR parsing batch {batch_idx}: {e}")
            if verbose and "text" in dir():
                log(f"    Raw text was:\n{text}")  # noqa: F821
            for name, _ in batch:
                all_captions.append(
                    {
                        "frame": name,
                        "caption": f"[ERROR: {e}]",
                        "batch_index": batch_idx,
                        "error": True,
                    }
                )

    total_time = time.monotonic() - total_start
    log(f"\nDone: {len(all_captions)} captions in {total_time:.1f}s total")

    return all_captions


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption video frames using Gemini API with configurable batch sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Individual frames (tests frame isolation):
  %(prog)s --scene-dir assets/embedded_frames/scene_10065 --batch-size 1

  # All 20 at once (tests cross-contamination):
  %(prog)s --scene-dir assets/embedded_frames/scene_10065 --batch-size 20

  # Specific frames only:
  %(prog)s --scene-dir assets/embedded_frames/scene_10065 --batch-size 1 \\
      --frames frame_0395.jpg frame_0526.jpg frame_2103.jpg

Environment variables:
  GEMINI_API_KEY or GOOGLE_API_KEY""",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Directory containing frame_*.jpg files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Frames per API call: 1=isolated, 20=all at once (default: 1)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model name (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        help="Specific frame filenames to process (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        help="Gemini API key (default: GEMINI_API_KEY or GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the prompt(s) that would be used and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full API responses for debugging",
    )

    args = parser.parse_args()

    # Validate scene directory
    if not args.scene_dir.is_dir():
        parser.error(f"Scene directory not found: {args.scene_dir}")

    frame_paths = discover_frames(args.scene_dir, args.frames)
    if not frame_paths:
        parser.error(f"No frame_*.jpg files found in {args.scene_dir}")

    # Show prompt mode
    if args.show_prompt:
        if args.batch_size == 1:
            print("=== Single Frame Prompt (batch_size=1) ===\n")
            print(build_single_prompt("frame_XXXX.jpg"))
        else:
            names = [p.name for p in frame_paths[: args.batch_size]]
            print(f"=== Batch Prompt (batch_size={args.batch_size}) ===\n")
            print(build_batch_prompt(names))
        return

    # Validate API key
    if not args.api_key:
        parser.error(
            "No API key found. Set GEMINI_API_KEY env var or pass --api-key"
        )

    log(f"Model:       {args.model}")
    log(f"Batch size:  {args.batch_size}")
    log(f"Frames:      {len(frame_paths)}")
    log(f"Temperature: {args.temperature}")
    log(f"API calls:   {-(-len(frame_paths) // args.batch_size)}")
    log("")

    # Process
    captions = process_frames(
        frame_paths,
        batch_size=args.batch_size,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    # Output: clean JSON array (compatible with existing caption JSON files)
    clean_output = [{"frame": c["frame"], "caption": c["caption"]} for c in captions]
    output_json = json.dumps(clean_output, indent=2)

    if args.output:
        args.output.write_text(output_json)
        log(f"\nSaved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
