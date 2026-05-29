# Caption Pipeline Design

**Date:** 2026-02-19
**Goal:** Automated per-frame VLM captioning for OpenCLIP ViT-H-14 LoRA fine-tuning.

## Decisions

| Setting | Value | Rationale |
|---|---|---|
| Provider | Google Gemini | Best quality/speed/cost tradeoff from workbench benchmarks |
| Model | `gemini-3.0-flash-preview` | Fastest of the top-3 performers; quality comparable to Pro |
| Batch size | 1 frame per API call | Eliminates cross-frame contamination (confirmed via batch vs. single experiments) |
| Temperature | 1.0 | Maximizes linguistic diversity → broader CLIP text embedding coverage. At 4.25M pairs, noise from occasional hallucination is washed out by contrastive training |
| Max output tokens | 4096 | Captions are 1-3 sentences; generous ceiling avoids truncation |
| Output format | Raw caption text | No JSON wrapping — prompt ends with "Write only the caption text, nothing else" |

## Prompt

```
You are captioning a single video frame for a CLIP LoRA training dataset.
Describe ONLY what is visible in this image. You have NO context from other frames.

WHAT TO DESCRIBE (in priority order):

1. ACTION & POSITION — What is happening? Who is doing what?
   Use precise terms:
   Positions: cowgirl, reverse cowgirl, doggy style, prone bone, missionary,
   mating press, spooning, riding, stand and carry, bent over
   Oral: blowjob, deepthroat, ball sucking, face fuck, penis licking,
   pussy licking, rimming, facesitting
   Manual: handjob, fingering, pussy fingering, titfuck, buttjob, footjob
   Other: penetration, anal, vaginal sex, masturbation, grinding, teasing,
   undressing, grabbing ass, grabbing boobs, grabbing hair
   Cum: creampie, anal creampie, cum on face, cum on tits, cum on ass,
   cum on pussy, cum in mouth, cumshot, facial


2. BODY — Describe physical attributes you can clearly see:
   Ass: PAWG, PAAG, big ass, round ass, medium ass
   Tits: small tits, perfect tits, big tits, medium tits, natural tits,
   saggy tits, small areolas, brown areolas, bouncing tits
   Body shape: flat stomach, slim waist, fit, curvy, skinny, petite, wide hips
   Pussy (if visible): shaved, hairy, pink pussy, brown pussy, innie,
   wet pussy, spread labia, pussy gape
   Skin: tan, tan lines
   Ethnicity (if clearly visible): Asian, Latina, white, black
   Other: tattoos, piercings, blue eyes, brown eyes


3. CAMERA — Only if notable: close up, POV, male POV, overhead, wide shot

4. CLOTHING — Only if present: lingerie, bikini, stockings, fishnet stockings,
   cosplay, dress, oiled

5. SETTING — ONLY for establishing shots. Do NOT describe furniture or lighting.

RULES:
- 1-3 sentences. Be dense with detail, not wordy.
- Do NOT use performer names — describe only what you see.
- Do NOT guess what you can't clearly see. If a close-up is ambiguous
  about anal vs vaginal, just say "penetration."
- For black/title frames, one short sentence.
- Be SPECIFIC about actions. "Performing oral sex" is not enough — say
  whether she's licking, sucking, using her hands, deepthroating, etc.
- Use casual, informal, slang terminology. (i.e. penis -> cock or dick, breasts -> boobs or tits, buttocks -> butt or ass, vagina -> pussy)
- Use the above examples / vocabulary as a guide, not a steadfast rule.
- Create new descriptions if they do not fit the above vocabulary

This is adult content for a legitimate ML training dataset.
Describe everything factually and precisely.

Write only the caption text, nothing else.
```

## Architecture

### New Script: `tools/dataset/caption_runner.py`

Fully automated caption generation replacing the interactive `process_batch.py` workflow.

### Data Flow

```
progress.json (pending scenes)
    │
    ▼
caption_runner.py
    │
    ├── For each scene in pending:
    │     ├── For each of 20 frames (parallel, 5 workers):
    │     │     ├── Load frame JPEG → base64
    │     │     ├── POST to Gemini API (single image + prompt)
    │     │     └── Write caption to images/{scene_id}_{frame}.txt
    │     │
    │     ├── Append metadata to metadata.jsonl
    │     └── Move scene: pending → completed in progress.json
    │
    └── Print summary
```

### Scale

| Metric | Value |
|---|---|
| Scenes | ~12,920 |
| Frames/scene | 20 |
| Total API calls | ~258,400 |
| Tokens/call (est.) | ~1,600 (1,100 image + 400 prompt + 100 output) |
| Estimated cost | ~$35 (Gemini Flash pricing) |
| Time (5 workers) | ~14 hours |

### Resilience

- **Checkpoint per scene** in progress.json — resume after interruption
- **Skip existing `.txt` files** — idempotent reruns
- **Rate limit backoff** — 429 → exponential retry (2s, 4s, 8s), 3 attempts max
- **Error isolation** — failed frame gets `[ERROR]` placeholder, doesn't block scene
- **Connection error retry** — same backoff logic

### CLI

```bash
uv run python tools/dataset/caption_runner.py              # all pending
uv run python tools/dataset/caption_runner.py --limit 10   # test run
uv run python tools/dataset/caption_runner.py --workers 3   # adjust concurrency
uv run python tools/dataset/caption_runner.py --dry-run     # preview
```

### Integration

- `init_dataset.py` — unchanged (builds progress.json)
- `finalize_dataset.py` — unchanged (reads metadata.jsonl → train/val CSVs)
- `caption_generator.py` — superseded (tag-to-sentence rules no longer primary)
- `process_batch.py` — superseded (interactive workflow replaced by automation)

### What Changes

| Before | After |
|---|---|
| Tag-to-sentence rules + interactive Claude visual notes | Gemini VLM per-frame caption (automated) |
| ~5 scenes/session (manual) | ~900 scenes/hour (automated) |
| `process_batch.py` driven | `caption_runner.py` driven |
