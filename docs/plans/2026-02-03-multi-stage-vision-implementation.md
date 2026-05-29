# Multi-Stage Vision Analysis: Implementation Design

## Overview

This document details the implementation strategy for the multi-stage vision analysis pipeline, building on the architectural design in `2026-02-02-multi-stage-vision-analysis-design.md`.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Default behavior | Multi-stage (3 stages) |
| Escape hatch | "Quick mode" for single-pass |
| UI location | Settings gear (⚙️) in Analyze tab |
| Options scope | All per-scene |
| Existing data | Drop all VisionHistory records |
| Code location | Keep in `scene_vision.py` |
| Prompt customization | User-editable, persists across scenes |

## Error Handling

| Stage | On Failure |
|-------|------------|
| Stage 1 (Classification) | Abort with error message |
| Stage 2 (Description) | Return classification only (partial results) |
| Stage 3 (Verification) | Return unverified description with indicator |

## UI Design

### Settings Panel

Accessed via ⚙️ icon next to "Analyze" button:

```
┌─────────────────────────────────┐
│ Analysis Options                │
├─────────────────────────────────┤
│ ☐ Quick mode (faster)           │
│ ☐ Skip verification             │
│                                 │
│ Frames: [Auto ▾]                │
│         - Auto (smart)          │
│         - 16 frames             │
│         - 32 frames             │
│         - 64 frames             │
│                                 │
│ ▶ Edit prompts                  │
└─────────────────────────────────┘
```

### Prompt Editor (expanded)

```
┌─────────────────────────────────┐
│ ▼ Edit prompts    [↻ Reset All] │
├─────────────────────────────────┤
│ Classification prompt:          │
│ ┌─────────────────────────────┐ │
│ │ Examine ALL frames...       │ │
│ └─────────────────────────────┘ │
│                      [↻ Reset]  │
│                                 │
│ Description constraints:        │
│ ┌─────────────────────────────┐ │
│ │ ## VERIFIED CLASSIFICATION  │ │
│ └─────────────────────────────┘ │
│                      [↻ Reset]  │
│                                 │
│ Verification prompt:            │
│ ┌─────────────────────────────┐ │
│ │ You are a fact-checker...   │ │
│ └─────────────────────────────┘ │
│                      [↻ Reset]  │
└─────────────────────────────────┘
```

Custom prompts persist in plugin config and apply to all scenes until reset.

### Classification Badges

Displayed above description after analysis:

```
👫 Couple  🎬 Live Action  👤×2  🏠 Indoor→Outdoor  🔥 Mixed
```

Badge mappings:

| Attribute | Badge Examples |
|-----------|----------------|
| Scene type | 👤 Solo Female, 👫 Couple, 👥 Threesome, 👥👥 Group |
| Content type | 🎬 Live Action, 🎨 Animated |
| Performer count | 👤×1, 👤×2, 👤×3 |
| Setting | 🏠 Indoor, 🌳 Outdoor, 🏠→🌳 Indoor→Outdoor |
| Primary activity | Text label, no emoji (e.g., "Oral", "Mixed") |

### Verification Status

Below badges:

- ✓ Verified (green) — All claims verified
- ⚠ N corrections (yellow) — Click to expand
- ⚡ Unverified (gray) — Stage 3 failed/skipped

Corrections displayed as inline callouts:

```
> ⚠️ **Correction**: Description said "blonde hair" but frames show brunette.
```

### Progress Indicator

```
Analyzing... Stage 1/3: Classification
Analyzing... Stage 2/3: Description
Analyzing... Stage 3/3: Verification
```

## Data Model

### VisionHistory Schema

```python
@dataclass
class VisionHistory:
    scene_id: int

    # Stage 1
    classification: dict          # JSON: scene_type, performer_count, etc.
    classification_evidence: dict # Frame citations for each attribute

    # Stage 2
    description: str              # Full description text
    suggested_tags: list[dict]    # Tag suggestions with confidence

    # Stage 3
    verification_status: str      # "verified" | "corrections" | "skipped" | "failed"
    corrections: list[dict]       # [{claim, verdict, correction_text}, ...]

    # Metadata
    frames_used: int
    frame_selection: str          # "smart" | "uniform" | "manual"
    quick_mode: bool              # True if single-pass was used
    model: str
    created_at: str
```

### Migration

Drop existing vision history table and recreate with new schema. No migration of old records.

### Custom Prompt Storage

Stored in plugin config:

```json
{
  "custom_prompts": {
    "classification": "...",
    "description_constraints": "...",
    "verification": "..."
  }
}
```

Null values indicate "use default".

## Backend Implementation

### New Methods

```python
def _run_classification_stage(
    self,
    frames: list[str],  # base64 images
    model: str,
    custom_prompt: str | None = None
) -> dict:
    """
    Stage 1: Classify scene fundamentals.
    Returns classification JSON with evidence citations.
    Raises on failure (no fallback).
    """

def _run_description_stage_constrained(
    self,
    frames: list[str],
    classification: dict,
    model: str,
    custom_constraints: str | None = None
) -> tuple[str, list[dict]]:
    """
    Stage 2: Generate description constrained by classification.
    Returns (description_text, suggested_tags).
    """

def _run_verification_stage(
    self,
    frames: list[str],
    description: str,
    model: str,
    custom_prompt: str | None = None
) -> dict:
    """
    Stage 3: Verify claims against frames.
    Returns {status: str, corrections: list[dict]}.
    """

def run_multi_stage_analysis(
    self,
    scene_id: int,
    options: AnalysisOptions
) -> VisionHistory:
    """
    Orchestrates all stages with error handling:
    - Stage 1 fail → raise error
    - Stage 2 fail → return partial (classification only)
    - Stage 3 fail → return with verification_status="failed"
    """
```

### AnalysisOptions

```python
@dataclass
class AnalysisOptions:
    quick_mode: bool = False
    skip_verification: bool = False
    frame_count: int | None = None  # None = auto/smart
    custom_prompts: dict | None = None
```

## Stage Prompts

### Stage 1: Classification

```
Examine ALL frames carefully. For each question, cite specific frame numbers.

1. Is this live-action or animated content?
2. How many distinct performers appear?
3. What genders are present?
4. Is this solo or does it involve partners?
5. Does the scene have non-sexual intro segments?
6. What settings appear (indoor/outdoor)?
7. What is the primary sexual activity?

Output as JSON:
{
  "content_type": "live_action|animated|mixed",
  "scene_type": "solo_female|solo_male|couple|threesome|group|other",
  "performer_count": <number>,
  "genders_present": "female_only|male_only|mixed",
  "setting_progression": "outdoor_only|indoor_only|outdoor_to_indoor|indoor_to_outdoor|mixed",
  "primary_activity": "softcore|masturbation|oral|vaginal|anal|mixed",
  "has_intro_segments": true|false,
  "evidence": { <frame citations for each field> }
}
```

### Stage 2: Description Constraints (prepended to existing prompt)

```
## VERIFIED CLASSIFICATION (do not contradict)
- Scene type: {classification.scene_type}
- Performers: {classification.performer_count} ({classification.genders_present})
- Content: {classification.content_type}
- Setting: {classification.setting_progression}

Your description MUST be consistent with these facts.
```

### Stage 3: Verification

```
You are a fact-checker. Verify each claim against the frames.

## Description to verify:
{stage_2_description}

For each claim, report:
<claim text="..." frames="..." verdict="CORRECT|INCORRECT">
  If incorrect, explain what you actually see.
</claim>

Verify: performer count, genders, physical descriptions, clothing,
positions, activities, props, settings.
```

## Retry Behavior

When retrying after Stage 2 failure:
- Reuse cached Stage 1 classification from VisionHistory
- Only re-run Stages 2 and 3
- Saves cost and time

## Files to Modify

| File | Changes |
|------|---------|
| `stash_ai/tasks/scene_vision.py` | New stage methods, `run_multi_stage_analysis()`, updated `VisionHistory` |
| `stash_ai/storage.py` | Drop/recreate vision history table |
| `stash-copilot.js` | Settings gear, prompt editor, badges, verification display, progress |
| `stash-copilot.css` | Badge styles, settings panel, correction callouts |
| Plugin config schema | Custom prompt storage fields |

## Implementation Order

1. Backend: New VisionHistory schema + storage migration
2. Backend: Three stage methods + orchestrator
3. Backend: Error handling and partial results
4. Frontend: Settings gear with options
5. Frontend: Prompt editor with reset
6. Frontend: Classification badges
7. Frontend: Verification status + corrections display
8. Frontend: Stage progress indicator
9. Testing: Validate on scene 13368

## Success Criteria

1. Scene 13368 correctly identified as couple scene (not solo)
2. Classification stage catches scene type errors
3. Verification stage catches description hallucinations
4. Corrections displayed clearly to user
5. Quick mode provides fast single-pass fallback
6. Custom prompts persist and work across scenes
