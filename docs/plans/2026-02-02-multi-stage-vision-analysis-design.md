# Multi-Stage Vision Analysis Pipeline

## Problem

The current single-pass scene vision analysis suffers from significant hallucination issues. The VLM can generate coherent but completely wrong descriptions - for example, describing a couple scene as solo masturbation, or inventing content that doesn't exist in any frame.

**Example case (scene 13368)**:
- VLM described: "Solo female masturbation with dildo"
- Actual content: Couple scene with outdoor intro, male partner clearly visible
- The VLM hallucinated an entire narrative that contradicted the visual evidence

## Solution

A three-stage verification pipeline that forces the VLM to:
1. Commit to fundamental scene attributes before detailed description
2. Generate descriptions constrained by verified classification
3. Self-verify claims against visual evidence

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1: Classification                   │
│  Input: All frames                                          │
│  Output: Structured JSON (content_type, scene_type, etc.)   │
│  Cost: ~500-800 tokens                                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 2: Description                      │
│  Input: All frames + Classification constraints             │
│  Output: Detailed description with progression table        │
│  Cost: ~2000-3000 tokens                                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 3: Verification                     │
│  Input: All frames + Stage 2 description                    │
│  Output: Verified claims with corrections appended          │
│  Cost: ~1000-1500 tokens                                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Final Output                            │
│  Stage 2 description + any correction notes from Stage 3    │
└─────────────────────────────────────────────────────────────┘
```

## Stage 1: Classification

### Purpose

Force the VLM to commit to fundamental scene attributes before detailed description. This prevents errors where the VLM describes the wrong scene type entirely.

### Attributes Captured

| Attribute | Values | Purpose |
|-----------|--------|---------|
| `content_type` | live_action, animated, mixed | Distinguish real vs cartoon/hentai/3D |
| `scene_type` | solo_female, solo_male, couple, threesome, group, other | Core scene classification |
| `performer_count` | integer | Number of distinct people/characters |
| `genders_present` | female_only, male_only, mixed | Gender composition |
| `setting_progression` | outdoor_only, indoor_only, outdoor_to_indoor, indoor_to_outdoor, mixed | Location flow |
| `primary_activity` | softcore, masturbation, oral, vaginal, anal, mixed | Main sexual activity |
| `has_intro_segments` | boolean | Non-sexual intro/conversation segments |

### Prompt Design

```
Examine ALL frames carefully. For each question, cite specific frame numbers as evidence.

1. Is this live-action or animated content? (cite frames)
2. How many distinct performers appear? (cite frames showing each)
3. What genders are present? (cite frames)
4. Is this solo or does it involve partners? (cite frames showing interaction)
5. Does the scene have non-sexual intro segments? (cite frames)
6. What settings appear (indoor/outdoor)? (cite frames)
7. What is the primary sexual activity? (cite frames)

Output as JSON:
{
  "content_type": "live_action|animated|mixed",
  "scene_type": "solo_female|solo_male|couple|threesome|group|other",
  "performer_count": <number>,
  "genders_present": "female_only|male_only|mixed",
  "setting_progression": "outdoor_only|indoor_only|outdoor_to_indoor|indoor_to_outdoor|mixed",
  "primary_activity": "softcore|masturbation|oral|vaginal|anal|mixed",
  "has_intro_segments": true|false,
  "evidence": {
    "performer_count": "Frames X, Y show performer A; frames Z, W show performer B",
    "scene_type": "Male visible in frames X, Y, Z interacting with female",
    ...
  }
}
```

### Output

Structured JSON stored in `VisionHistory.classification` for use in Stage 2.

---

## Stage 2: Description

### Purpose

Generate detailed description constrained by Stage 1 classification results. The classification acts as guardrails that prevent contradictory hallucinations.

### Constraint Injection

Classification results are injected into the prompt as verified facts:

```
## Classification Results (VERIFIED - do not contradict)
- Content type: live_action
- Scene type: couple
- Performers: 2 (1 female, 1 male)
- Setting: outdoor_to_indoor
- Has intro: true

Your description MUST be consistent with these verified facts.
If you see something that contradicts these, note it as an observation
but do not change the classification.
```

### Prompt Modifications

The existing description prompt is modified to:
1. Include classification constraints at the top
2. Require frame number citations for progression timestamps
3. Add explicit instruction: "Describe what you ACTUALLY SEE, not what you expect"

### Output

Detailed description with:
- Overview (constrained by classification)
- Performer descriptions
- Progression table with timestamps
- Suggested question

---

## Stage 3: Verification

### Purpose

Re-examine frames against the generated description to catch any hallucinations or errors that slipped through despite classification constraints.

### Verification Process

1. Show the same frames again
2. Provide the Stage 2 description
3. Ask VLM to verify each major claim against visual evidence

### Prompt Design

```
You are a fact-checker verifying a scene description against video frames.

## Description to verify:
{stage_2_description}

## Your task:
For each claim in the description, check if it matches what you see in the frames.

Report in this format:
<verification>
  <claim text="performer has dark hair" frames_cited="1,5,12" verdict="INCORRECT">
    Actually has light brown/blonde hair visible in frames 5, 12, 32
  </claim>
  <claim text="using pink dildo" frames_cited="32,42" verdict="INCORRECT">
    No dildo visible. Male partner present - this is penetrative sex.
  </claim>
  <claim text="wearing red lingerie" frames_cited="40-45" verdict="CORRECT">
    Red bodysuit/lingerie visible in cited frames
  </claim>
</verification>

Verify these aspects:
- Performer count and genders
- Physical descriptions (hair, body type, etc.)
- Clothing/state of undress
- Positions and activities
- Props/toys mentioned
- Setting descriptions
```

### Correction Handling

When verification finds errors:
- Corrections are appended as notes to the final description
- Format: `**[Correction]** Original claim was X, but frames show Y`
- Simple append approach (no full rewrite) for cost efficiency

### Output

- If all claims CORRECT: Return Stage 2 description unchanged
- If errors found: Return Stage 2 description + correction notes appended

---

## Implementation Details

### Location

- New method `_run_multi_stage_description()` in `stash_ai/tasks/scene_vision.py`
- Replaces current single-pass as the default behavior
- Old single-pass code retained but not called by default

### Data Persistence

Store in `VisionHistory`:
- `classification`: Stage 1 JSON output
- `classification_evidence`: Frame citations from Stage 1
- `verification_results`: Stage 3 verification output
- `corrections_applied`: List of corrections appended

### Frame Handling

- All 3 stages see the same frame selection (smart or uniform)
- No re-extraction or re-selection between stages
- Images sent 3 times (once per stage)

### Fallback Behavior

- If Stage 1 fails to parse → Fall back to current single-pass
- If Stage 3 fails to parse → Return Stage 2 description without verification
- Errors logged for debugging

### Cost Estimate

| Stage | Text Tokens | Image Tokens (64 frames) | Total |
|-------|-------------|--------------------------|-------|
| Stage 1 | ~500-800 | ~25,600 | ~26,400 |
| Stage 2 | ~2,000-3,000 | ~25,600 | ~28,600 |
| Stage 3 | ~1,000-1,500 | ~25,600 | ~27,100 |
| **Total** | ~4,500 | ~76,800 | ~81,300 |

Compared to current single-pass (~28,000 tokens), this is approximately **3x cost** for significantly improved accuracy.

---

## Files to Modify

| File | Changes |
|------|---------|
| `stash_ai/tasks/scene_vision.py` | Add `_run_multi_stage_description()`, classification prompts, verification logic |
| `stash_ai/tasks/scene_vision.py` | Update `VisionHistory` dataclass with new fields |
| `stash_ai/tasks/scene_vision.py` | Modify `_run_description_stage()` to call multi-stage by default |

---

## Success Criteria

1. Scene 13368 correctly identified as couple scene (not solo)
2. Classification stage catches fundamental scene type errors
3. Verification stage catches description hallucinations
4. Corrections appended provide transparency about what was fixed
5. No regression on scenes that were previously described correctly

---

## Future Enhancements

- **Configurable strictness**: Option to require full rewrite on errors vs append
- **Confidence thresholds**: Skip verification for high-confidence classifications
- **Caching**: Cache Stage 1 results for re-analysis workflows
- **User override**: Allow manual classification override before Stage 2
