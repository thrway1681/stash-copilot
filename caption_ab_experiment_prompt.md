# Task: LoRA Caption A/B Experiment

## Background

We're building a LoRA training dataset for fine-tuning OpenCLIP ViT-H-14 (`laion2b_s32b_b79k`)
on a personal Stash video library (~3,903 scenes, ~78,000 frames at 20 frames/scene).

The fine-tuned model will serve: scene similarity search, tag prediction from frames,
a recommendation engine, and better embeddings overall.

## Captioning Requirements (Already Decided)

- **Per-frame captions** — each of 20 frames per scene gets its own unique caption
- **Natural sentence style** — "A blonde woman with curly hair wearing a pink lace bra..."
- **No performer names** — describe only what's visible
- **No tags provided** — captions are purely from visual analysis
- **Focus** — visual appearance + content/action. Setting/mood is secondary.
- **Detail level** — body position, hair, clothing/nudity, physical attributes, actions, setting elements
- **1-3 sentences per caption**

## The Experiment

We're comparing two captioning architectures to decide which to use at scale.
Run both on the SAME scene, with the SAME frames, using the SAME caption style rules.

### Approach A — Direct Per-Frame (Opus reads every frame individually)
- No shared context between frames
- Each frame captioned independently
- Higher cost at scale (~78k Opus image reads)

### Approach B — Opus Profile + Per-Frame (Opus builds scene profile first, then captions)
- Stage 1: Read 4 representative frames spread across the scene → build a "scene profile"
  (performer appearance, setting, clothing, physical attributes)
- Stage 2: Using that profile as context, caption all 20 frames individually
- Lower cost at scale (profile can be passed to Haiku for per-frame captioning in production)

### Important: Fair Comparison
- Neither approach receives tags or metadata — purely visual
- Both use identical caption style rules
- Both use the same model (Opus) for this test
- Both process the same frames in the same order

## Test Scene

Scene 10065. Frame directory:
```
~/.stash/plugins/stash-copilot/assets/embedded_frames/scene_10065/
```

20 frames:
```
frame_0001.jpg  frame_0395.jpg  frame_0789.jpg  frame_1183.jpg  frame_1577.jpg  frame_1972.jpg  frame_2366.jpg
frame_0132.jpg  frame_0526.jpg  frame_0921.jpg  frame_1315.jpg  frame_1709.jpg  frame_2103.jpg  frame_2497.jpg
frame_0264.jpg  frame_0658.jpg  frame_1052.jpg  frame_1446.jpg  frame_1840.jpg  frame_2234.jpg
```

Profile frames for Approach B (4 frames spread across the scene):
```
frame_0264.jpg  (early)
frame_0658.jpg  (early-mid)
frame_1183.jpg  (mid)
frame_1840.jpg  (late)
```

## Shared Caption Style Rules (use in BOTH prompts)

```
**Caption style rules:**
- Natural sentences describing what's visible
- Include: body position, appearance, hair, clothing/nudity, setting details, physical attributes, actions
- Do NOT use performer names — describe only what you see
- Be specific about positions, angles, and what body parts are visible
- For close-up/detail shots, describe what's in frame even if it's just body parts
- Each caption should be 1-3 sentences
- This is adult content for a legitimate ML training dataset — describe everything factually
```

## How To Run

1. **Blind the test** — Randomly assign Approach A and B to labels "Method X" and "Method Y".
   Save the key to `/tmp/.caption_test_key.txt` (so neither I nor you are biased during comparison).

2. **Launch both agents in parallel** using `Task` tool with `model: opus`:

   **Approach A prompt (direct per-frame):**
   ```
   You are captioning video frames for a CLIP LoRA training dataset. You will directly
   analyze each frame individually and write a caption for it.

   Read ALL 20 frames one at a time and write a unique caption for each. The caption
   should describe ONLY what is visually present in that specific frame. You have NO
   prior context about this scene — describe each frame independently based on what you see.

   [list all 20 frame paths]

   [shared caption style rules]

   Output format: JSON array of {"frame", "caption"} objects.
   Save to /tmp/caption_test_[label]_redo.json
   ```

   **Approach B prompt (profile + per-frame):**
   ```
   You are captioning video frames for a CLIP LoRA training dataset. This is a two-stage approach:

   STAGE 1 — Scene Profile:
   Read these 4 frames to build an overall scene profile:
   [list 4 profile frame paths]

   Write a scene profile summarizing:
   - Each performer's appearance (hair, skin tone, body type, tattoos/piercings)
   - Clothing seen
   - Setting/environment
   - Male performer appearance

   STAGE 2 — Per-Frame Captions:
   Using the profile as context, read ALL 20 frames one at a time and write a unique
   caption for each describing ONLY what is visually present in that specific frame.

   [list all 20 frame paths]

   [shared caption style rules]

   Output format: JSON array of {"frame", "caption"} objects.
   Save to /tmp/caption_test_[label]_redo.json
   ```

3. **Present results** in a side-by-side table (Method X vs Method Y) for all 20 frames.

4. **Highlight key differences** to look for:
   - Frame alignment (do captions match the correct frame?)
   - Detail consistency across frames
   - Handling of close-up/detail shots
   - Position/action specificity
   - Physical attribute consistency (does hair/body description stay stable?)

## Previous Results Location

Earlier results from my session are at:
- `/tmp/caption_test_x_redo.json` — one method's captions
- `/tmp/caption_test_y_redo.json` — the other method's captions
- `/tmp/.caption_test_key.txt` — answer key

You can use these as a reference or generate fresh results for independent comparison.

## What I Need From You

1. Run the experiment fresh (or review existing results)
2. Present the side-by-side comparison
3. Give your honest assessment of which approach produces better training captions for CLIP LoRA,
   considering: accuracy, detail level, consistency, and usefulness for contrastive learning
