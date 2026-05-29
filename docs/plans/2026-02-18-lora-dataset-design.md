# LoRA Training Dataset Design

**Date:** 2026-02-18
**Status:** Approved
**Author:** Claude + user session

---

## Overview

Build a self-contained image-caption dataset from the Stash library for LoRA fine-tuning of OpenCLIP ViT-H-14 (`laion2b_s32b_b79k`). The dataset will improve embedding quality for three tasks: text→image retrieval, image→image similarity search, and image→tag classification.

---

## Library Stats (at design time)

| Metric | Value |
|---|---|
| Total scenes | 12,006 |
| Scenes with embedded frames | 12,920 |
| Total available frames | 4,264,870 |
| Avg frames per scene | 330 |
| Embedded frames disk usage | 175 GB |
| Free disk space | 1.2 TB |
| **Scenes selected (5+ content tags)** | **3,903** |
| **Target image-caption pairs** | **~78,060** |

---

## Scope

### Scene Selection
Include only scenes with **5 or more content tags** (excluding admin/workflow tags). This yields 3,903 scenes with enough tag signal to generate meaningful captions.

**Excluded admin tags:**
`Embedded`, `To Embed`, `To Script`, `Funscript`, `Missing Performer (Male)`, `HD Available`, `FS: Action`, `FS: Beat`, `Start`, `Free stroke`, `OG beat comes back`, `Funk Beat`, `Funk Beat comes back`, `Jiggle Fuck`, `Hip Sway`, `Mixed Audio`, `Music Only`, `Event 2024`, `Event 2025`, `Custom Marker A`, `Custom Marker B`, `Remix`, `Cumpilation`, `[AVN Award Winner]`, `[Award Winner]`, `[MiscTags: Skip]`, `[SIT: Multi-Script]`, `[Set Profile Image]`, `[Stashbox Performer Gallery]`, `[TPDB: Skip Marker]`, `[Timestamp: Skip Sync]`

### Frame Sampling
**20 frames per scene**, uniformly sampled from the scene's frame timeline (not random — evenly spaced indices to maximize temporal diversity). All frames from the same scene share the same caption, since the caption describes scene-level content.

### Visual Analysis
For each scene, Claude (the AI assistant, not an external model) reads **2 representative frames** — at the 1/3 and 1/2 marks of the scene timeline — to:
- Observe what is actually happening (acts, positions, clothing state, setting, camera angle)
- Compare observations to existing Stash tags
- Identify and log missing tags
- Produce an augmented natural language caption

This is a **multi-session process** (~26 sessions at ~150 scenes/session).

---

## Output Directory Structure

```
assets/lora_dataset/
├── images/
│   ├── s{scene_id}_f{frame_num:04d}.jpg   ← copied frame (source NOT deleted)
│   ├── s{scene_id}_f{frame_num:04d}.txt   ← caption (same basename, .txt extension)
│   └── ...
├── train.csv             ← 90% split: columns: filepath, caption
├── val.csv               ← 10% holdout: columns: filepath, caption
├── metadata.jsonl        ← one JSON record per scene (not per frame):
│                            scene_id, frame_paths, tags, caption,
│                            visual_notes, missing_tags, analyzed_at
├── progress.json         ← checkpoint for multi-session resume
└── README.md             ← dataset card with stats, split info, tag coverage
```

**Flat image directory** (not nested by scene) — required by `open_clip_train.main` CSV loader and `clipora`.

---

## Caption Format

**Natural language sentence(s)** describing scene-level content, suitable for CLIP contrastive training.

### Template
```
A {production_style} scene featuring {performer_description} with {physical_attributes}.
{acts_and_positions}. {setting_details}.
```

### Examples
```
An amateur OnlyFans scene featuring a PAWG performer with big ass, small tits, and a tan
complexion. The performer is giving a blowjob in an indoor setting, POV perspective.

A non-nude solo scene featuring a petite Asian performer with small tits and blue eyes,
twerking and dancing in lingerie.

A hardcore scene featuring a curvy performer with big ass and big tits. Doggy style vaginal
sex with a white male performer, indoor bedroom setting.
```

### Tag-to-sentence mapping rules
- **Body type tags** → "a [body_type] performer with [attributes]"
- **Act tags** → sentence describing the act + position if known
- **Content style tags** → leading modifier (amateur, hardcore, POV, etc.)
- **Setting tags** → closing clause (indoor/outdoor, location type)
- **Ethnicity tags** → included in performer description
- **Admin tags** → excluded entirely

---

## Visual Analysis Protocol

For each scene, I will:

1. **Read frame at 1/3 of timeline** — typically captures the setup/beginning of the main act
2. **Read frame at 1/2 of timeline** — typically captures the peak activity
3. **Observe and note:**
   - Primary sexual act (if any): intercourse, oral, manual, solo, non-nude
   - Position (if applicable): missionary, doggy, cowgirl, POV, etc.
   - Clothing state: nude, partially clothed, fully clothed
   - Setting: indoor/outdoor, specific location type
   - Camera angle: POV, wide, close-up, overhead
   - Performer count: solo, couple, group
   - Any body attributes visible that aren't in existing tags
4. **Compare to existing tags** — log anything observed but not tagged
5. **Write caption** combining existing tags + observations

### Missing Tag Log
Every observation not covered by existing tags is written to `metadata.jsonl` under `missing_tags`. At end of each session a summary is printed showing systemic gaps.

---

## Progress / Resume Architecture

`progress.json` structure:
```json
{
  "total_scenes": 3903,
  "completed": ["19", "21", "22", ...],
  "pending": ["25", "26", ...],
  "last_updated": "2026-02-18T17:00:00Z",
  "pairs_written": 1240,
  "sessions": 1
}
```

Each session:
1. Load `progress.json`
2. Determine remaining scenes
3. Process until context approaches limit
4. Save checkpoint
5. Report: N scenes done, N remaining, estimated sessions left

---

## Train/Val Split

- **90% train** / **10% validation**
- Split at the **scene level** (not frame level) — all frames from a scene go to the same split, preventing data leakage where the model sees the same scene in both train and val
- Stratified by content type where possible

---

## Generated CSV Format (for clipora / open_clip_train)

`train.csv`:
```csv
filepath,caption
~/.stash/plugins/stash-copilot/assets/lora_dataset/images/s19_f0001.jpg,"An amateur non-nude scene featuring a PAWG performer with big ass, small tits, and a tan complexion. The performer is posing and dancing in an indoor setting."
```

---

## Missing Tag Report

At the end of full dataset construction, a `missing_tags_report.md` will be generated in the dataset directory summarizing:
- Which act tags are most commonly missing
- Which position tags are never present
- Recommended tags to add to the Stash taxonomy
- Scenes where the visual content doesn't match existing tags

---

## Hardware / Training Notes

- **Local training**: RTX 5090 (32 GB) — LoRA rank 64, gradient checkpointing, 8-bit Adam, SigLIP loss
- **Cloud training**: RunPod A100 80 GB (~$1.74/hr) for full fine-tune — estimated $30–80 total
- **Dataset size**: ~78,000 pairs, ~10–15 GB copied images
- **Training time estimate**: 6–9 hours per 10-epoch run on A100 80 GB

---

## Implementation Phases

### Phase 1: Infrastructure (current session)
- Write Python script to select scenes, sample frame paths, initialize `progress.json`
- Create dataset directory structure
- Write caption generation functions

### Phase 2: Visual Analysis + Caption Writing (multi-session)
- Each session: load progress, process ~150 scenes, read 2 frames each, write captions + copy 20 frames, save checkpoint
- ~26 sessions to complete all 3,903 scenes

### Phase 3: Finalization
- Generate `train.csv` / `val.csv` from completed metadata
- Write `missing_tags_report.md`
- Write `README.md` dataset card

### Phase 4: Training
- Launch LoRA training locally on RTX 5090
- Optional: full fine-tune on RunPod A100 80 GB
