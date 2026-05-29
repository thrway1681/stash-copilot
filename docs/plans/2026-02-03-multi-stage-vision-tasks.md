# Multi-Stage Vision Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a 3-stage vision analysis pipeline (classification → description → verification) to reduce VLM hallucinations.

**Architecture:** The existing single-pass `_run_description_stage()` becomes the fallback "quick mode". New methods `_run_classification_stage()`, `_run_constrained_description_stage()`, and `_run_verification_stage()` implement the multi-stage pipeline. The orchestrator `_run_multi_stage_analysis()` chains them with error handling. VisionHistory gets new fields for classification/verification data.

**Tech Stack:** Python (scene_vision.py), JavaScript (stash-copilot.js), CSS (stash-copilot.css)

---

## Task 1: Add Classification Data Structures to VisionHistory

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py:449-600` (VisionHistory class)

**Step 1: Add new fields to VisionHistory.__init__()**

In `stash_ai/tasks/scene_vision.py`, find the `VisionHistory.__init__()` method (line ~452) and add these fields after `self.tool_calls`:

```python
        # Multi-stage analysis fields (Stage 1: Classification)
        self.classification: Optional[Dict[str, Any]] = None  # JSON from Stage 1
        self.classification_evidence: Optional[Dict[str, str]] = None  # Frame citations

        # Multi-stage analysis fields (Stage 3: Verification)
        self.verification_status: str = "pending"  # "pending" | "verified" | "corrections" | "skipped" | "failed"
        self.corrections: List[Dict[str, Any]] = []  # [{claim, verdict, correction_text}, ...]

        # Analysis options
        self.quick_mode: bool = False  # True if single-pass was used
        self.skip_verification: bool = False  # True if Stage 3 was skipped
```

**Step 2: Update VisionHistory.to_dict()**

Find `to_dict()` method (~line 527) and add serialization for new fields after `"tool_calls"`:

```python
            # Multi-stage analysis fields
            "classification": self.classification,
            "classification_evidence": self.classification_evidence,
            "verification_status": self.verification_status,
            "corrections": self.corrections,
            "quick_mode": self.quick_mode,
            "skip_verification": self.skip_verification,
```

**Step 3: Update VisionHistory.from_dict()**

Find `from_dict()` classmethod (~line 588) and add deserialization. After loading existing fields, add:

```python
        # Multi-stage analysis fields
        history.classification = data.get("classification")
        history.classification_evidence = data.get("classification_evidence")
        history.verification_status = data.get("verification_status", "pending")
        history.corrections = data.get("corrections", [])
        history.quick_mode = data.get("quick_mode", False)
        history.skip_verification = data.get("skip_verification", False)
```

**Step 4: Verify imports work**

Run: `cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision && uv run python -c "from stash_ai.tasks.scene_vision import VisionHistory; h = VisionHistory('123'); print(h.classification, h.verification_status)"`

Expected: `None pending`

**Step 5: Commit**

```bash
cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): add multi-stage fields to VisionHistory

Add classification, verification_status, corrections, quick_mode, and
skip_verification fields to support the 3-stage analysis pipeline.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add AnalysisOptions Dataclass

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py:10-35` (imports area)

**Step 1: Add AnalysisOptions dataclass**

After the existing imports (around line 35, before the SCENE_VISION_SUBDIR constant), add:

```python
@dataclass
class AnalysisOptions:
    """Options for multi-stage vision analysis."""
    quick_mode: bool = False  # Use single-pass instead of multi-stage
    skip_verification: bool = False  # Skip Stage 3 verification
    frame_count: Optional[int] = None  # Override auto frame count (None = smart selection)
    custom_prompts: Optional[Dict[str, str]] = None  # Custom prompts for each stage
```

**Step 2: Verify imports work**

Run: `cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision && uv run python -c "from stash_ai.tasks.scene_vision import AnalysisOptions; o = AnalysisOptions(quick_mode=True); print(o.quick_mode, o.frame_count)"`

Expected: `True None`

**Step 3: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): add AnalysisOptions dataclass

Configuration object for multi-stage analysis: quick_mode, skip_verification,
frame_count override, and custom_prompts support.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Classification Stage Prompts

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py:80-100` (prompts section)

**Step 1: Add classification prompt constant**

After `FOLLOWUP_SYSTEM_PROMPT` (around line 79), add:

```python
# ============================================================================
# Multi-Stage Vision Analysis Prompts
# ============================================================================

CLASSIFICATION_PROMPT = """Examine ALL frames carefully. For each question, cite specific frame numbers as evidence.

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
    "content_type": "Frame X shows real human features",
    "performer_count": "Frames X, Y show performer A; frames Z, W show performer B",
    "scene_type": "Male visible in frames X, Y, Z interacting with female",
    "genders_present": "Female in frames X-Y, male in frames Z-W",
    "setting_progression": "Outdoor setting in frames 1-5, indoor from frame 6",
    "primary_activity": "Oral in frames X-Y, vaginal from frame Z",
    "has_intro_segments": "Frames 1-3 show conversation with clothes on"
  }
}"""

CLASSIFICATION_CONSTRAINTS_TEMPLATE = """## VERIFIED CLASSIFICATION (do not contradict)
- Content type: {content_type}
- Scene type: {scene_type}
- Performers: {performer_count} ({genders_present})
- Setting: {setting_progression}
- Has intro: {has_intro_segments}

Your description MUST be consistent with these verified facts.
If you observe something that contradicts these classifications, note it but do not change your description to match the contradiction.

---

"""

VERIFICATION_PROMPT_TEMPLATE = """You are a fact-checker verifying a scene description against video frames.

## Description to verify:
{description}

## Your task:
For each major claim in the description, check if it matches what you see in the frames.

Report in this format:
<verification>
  <claim text="[exact claim from description]" frames_cited="[frame numbers]" verdict="CORRECT|INCORRECT">
    [If incorrect, explain what you actually see in those frames]
  </claim>
</verification>

Verify these aspects:
- Performer count and genders
- Physical descriptions (hair color, body type, etc.)
- Clothing and state of undress
- Positions and activities described
- Props or toys mentioned
- Setting descriptions

Only report claims that are INCORRECT. If everything is correct, output:
<verification>
  <all_correct>true</all_correct>
</verification>"""
```

**Step 2: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision && uv run python -c "from stash_ai.tasks.scene_vision import CLASSIFICATION_PROMPT, VERIFICATION_PROMPT_TEMPLATE; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): add multi-stage analysis prompts

Add CLASSIFICATION_PROMPT for Stage 1, CLASSIFICATION_CONSTRAINTS_TEMPLATE
for Stage 2 constraint injection, and VERIFICATION_PROMPT_TEMPLATE for Stage 3.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement _run_classification_stage()

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py` (add method after `_run_description_stage`)

**Step 1: Find insertion point**

The method `_run_description_stage` ends around line 2200. Add the new method after it.

**Step 2: Add _run_classification_stage method**

```python
    def _run_classification_stage(
        self,
        history: VisionHistory,
        frames_base64: List[str],
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1: Classify fundamental scene attributes.

        Forces the VLM to commit to scene type, performer count, etc. before
        detailed description. This prevents hallucinations where the VLM
        describes the wrong type of scene entirely.

        Args:
            history: VisionHistory to update
            frames_base64: List of base64-encoded frame images
            custom_prompt: Optional custom classification prompt

        Returns:
            Classification dict with scene attributes and evidence

        Raises:
            ValueError: If classification fails to parse
        """
        self.log(f"Stage 1: Running classification ({len(frames_base64)} frames)", "info")
        history.status = "classifying"
        history.status_message = "Stage 1/3: Classifying scene..."
        history.stage_start_time = datetime.now().isoformat()
        self._save_history(history)

        prompt = custom_prompt or CLASSIFICATION_PROMPT

        # Use same system prompt as description stage
        model_name = getattr(self.description_llm, 'model', '').lower()
        is_gemma_model = any(kw in model_name for kw in GEMMA_MODEL_KEYWORDS)

        if is_gemma_model:
            system_prompt = GEMMA_SYSTEM_PROMPT
        else:
            system_prompt = VISION_SYSTEM_PROMPT

        # Build messages for VLM
        messages = [Message(role="user", content=prompt, images=frames_base64)]

        try:
            result = self.description_llm.complete(
                messages=messages,
                system=system_prompt,
            )

            response_text = result.text.strip()
            self.log(f"Classification response length: {len(response_text)}", "debug")

            # Parse JSON from response
            classification = self._parse_classification_json(response_text)

            # Store in history
            history.classification = {
                k: v for k, v in classification.items() if k != "evidence"
            }
            history.classification_evidence = classification.get("evidence", {})

            self.log(f"Classification complete: {history.classification.get('scene_type')}, "
                    f"{history.classification.get('performer_count')} performers", "info")

            return classification

        except Exception as e:
            self.log(f"Classification stage failed: {e}", "error")
            history.status = "error"
            history.status_message = f"Classification failed: {str(e)}"
            self._save_history(history)
            raise ValueError(f"Classification stage failed: {e}")

    def _parse_classification_json(self, response: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from classification response.

        Handles responses that may have text before/after the JSON.
        """
        import re

        # Try to find JSON block in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            raise ValueError("No JSON found in classification response")

        json_str = json_match.group()

        try:
            classification = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common issues
            # Remove trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            classification = json.loads(json_str)

        # Validate required fields
        required_fields = ["content_type", "scene_type", "performer_count", "genders_present"]
        missing = [f for f in required_fields if f not in classification]
        if missing:
            raise ValueError(f"Classification missing required fields: {missing}")

        return classification
```

**Step 3: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): implement _run_classification_stage

Stage 1 of multi-stage pipeline: classify scene type, performer count,
genders, setting, and primary activity with frame evidence citations.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement _run_constrained_description_stage()

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py` (add method after classification stage)

**Step 1: Add constrained description method**

```python
    def _run_constrained_description_stage(
        self,
        history: VisionHistory,
        frames_base64: List[str],
        classification: Dict[str, Any],
        scene_context: Dict[str, Any],
        frame_timestamps: List[float],
        custom_constraints: Optional[str] = None,
    ) -> str:
        """
        Stage 2: Generate description constrained by classification.

        Injects classification results as verified facts that the description
        must be consistent with.

        Args:
            history: VisionHistory to update
            frames_base64: List of base64-encoded frame images
            classification: Classification dict from Stage 1
            scene_context: Dict with performers info
            frame_timestamps: List of timestamps for each frame
            custom_constraints: Optional custom constraints template

        Returns:
            Description text
        """
        self.log(f"Stage 2: Running constrained description ({len(frames_base64)} frames)", "info")
        history.status = "describing"
        history.status_message = "Stage 2/3: Generating description..."
        history.stage_start_time = datetime.now().isoformat()
        self._save_history(history)

        # Build constraints from classification
        constraints_template = custom_constraints or CLASSIFICATION_CONSTRAINTS_TEMPLATE
        constraints = constraints_template.format(
            content_type=classification.get("content_type", "unknown"),
            scene_type=classification.get("scene_type", "unknown"),
            performer_count=classification.get("performer_count", "unknown"),
            genders_present=classification.get("genders_present", "unknown"),
            setting_progression=classification.get("setting_progression", "unknown"),
            has_intro_segments=classification.get("has_intro_segments", "unknown"),
        )

        # Build the full description prompt with constraints prepended
        performer_context = self._build_performer_context(scene_context)
        formatted_timestamps = self._format_frame_timestamps(frame_timestamps)

        # Get base description prompt (reuse existing logic)
        model_name = getattr(self.description_llm, 'model', '').lower()
        is_gemma_model = any(kw in model_name for kw in GEMMA_MODEL_KEYWORDS)
        is_caption_model = any(kw in model_name for kw in CAPTION_MODEL_KEYWORDS)

        try:
            if is_gemma_model:
                template = get_prompt("vision", "description", "gemma")
            elif is_caption_model:
                template = get_prompt("vision", "description", "caption")
            else:
                template = get_prompt("vision", "description", "structured")
        except (FileNotFoundError, KeyError):
            if is_gemma_model:
                template = GEMMA_PROMPT
            elif is_caption_model:
                template = CAPTION_PROMPT
            else:
                template = STRUCTURED_PROMPT

        actual_frame_count = len(frame_timestamps) if frame_timestamps else len(frames_base64)

        base_prompt = template.format(
            performer_context=performer_context,
            frame_count=actual_frame_count,
            frame_timestamps=formatted_timestamps,
        )

        # Prepend constraints to prompt
        full_prompt = constraints + base_prompt

        # Select system prompt
        if is_gemma_model:
            system_prompt = GEMMA_SYSTEM_PROMPT
        else:
            system_prompt = VISION_SYSTEM_PROMPT

        # Build messages
        messages = [Message(role="user", content=full_prompt, images=frames_base64)]

        try:
            result = self.description_llm.complete(
                messages=messages,
                system=system_prompt,
            )

            description = result.text.strip()

            # Extract suggested question if present
            cleaned_description, suggested_question = extract_suggested_question(description)
            if suggested_question:
                history.suggested_question = suggested_question

            history.description = cleaned_description
            history.description_complete = True

            # Add to message history
            user_msg = VisionMessage(
                role="user",
                content="Analyze this scene and describe what you see.",
                has_image=True,
            )
            history.add_message(user_msg)
            assistant_msg = VisionMessage(role="assistant", content=cleaned_description)
            history.add_message(assistant_msg)

            self.log(f"Constrained description complete ({len(cleaned_description)} chars)", "info")
            return cleaned_description

        except Exception as e:
            self.log(f"Description stage failed: {e}", "error")
            history.status = "error"
            history.status_message = f"Description failed: {str(e)}"
            self._save_history(history)
            raise
```

**Step 2: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): implement _run_constrained_description_stage

Stage 2 of multi-stage pipeline: generate description with classification
constraints injected to prevent contradictory hallucinations.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Implement _run_verification_stage()

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py` (add method after description stage)

**Step 1: Add verification method**

```python
    def _run_verification_stage(
        self,
        history: VisionHistory,
        frames_base64: List[str],
        description: str,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 3: Verify description claims against frames.

        Re-examines frames to catch any hallucinations in the description.

        Args:
            history: VisionHistory to update
            frames_base64: List of base64-encoded frame images
            description: Description text from Stage 2
            custom_prompt: Optional custom verification prompt

        Returns:
            Dict with verification status and any corrections
        """
        self.log(f"Stage 3: Running verification ({len(frames_base64)} frames)", "info")
        history.status = "verifying"
        history.status_message = "Stage 3/3: Verifying claims..."
        history.stage_start_time = datetime.now().isoformat()
        self._save_history(history)

        prompt_template = custom_prompt or VERIFICATION_PROMPT_TEMPLATE
        prompt = prompt_template.format(description=description)

        # Use same system prompt
        model_name = getattr(self.description_llm, 'model', '').lower()
        is_gemma_model = any(kw in model_name for kw in GEMMA_MODEL_KEYWORDS)
        system_prompt = GEMMA_SYSTEM_PROMPT if is_gemma_model else VISION_SYSTEM_PROMPT

        messages = [Message(role="user", content=prompt, images=frames_base64)]

        try:
            result = self.description_llm.complete(
                messages=messages,
                system=system_prompt,
            )

            response = result.text.strip()
            verification = self._parse_verification_response(response)

            # Update history
            if verification["all_correct"]:
                history.verification_status = "verified"
                history.corrections = []
                self.log("Verification complete: all claims correct", "info")
            else:
                history.verification_status = "corrections"
                history.corrections = verification["corrections"]
                self.log(f"Verification complete: {len(verification['corrections'])} corrections", "info")

                # Append corrections to description
                if verification["corrections"]:
                    correction_notes = self._format_corrections(verification["corrections"])
                    history.description = history.description + "\n\n" + correction_notes

            return verification

        except Exception as e:
            self.log(f"Verification stage failed: {e}", "warning")
            history.verification_status = "failed"
            self._save_history(history)
            # Don't raise - verification failure is not fatal
            return {"all_correct": None, "corrections": [], "error": str(e)}

    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse verification XML response."""
        import re

        # Check for all_correct
        if "<all_correct>true</all_correct>" in response.lower():
            return {"all_correct": True, "corrections": []}

        # Parse claim elements
        corrections = []
        claim_pattern = r'<claim\s+text="([^"]+)"\s+frames_cited="([^"]+)"\s+verdict="([^"]+)">\s*(.*?)\s*</claim>'

        for match in re.finditer(claim_pattern, response, re.DOTALL | re.IGNORECASE):
            claim_text, frames, verdict, explanation = match.groups()
            if verdict.upper() == "INCORRECT":
                corrections.append({
                    "claim": claim_text,
                    "frames_cited": frames,
                    "verdict": verdict,
                    "correction": explanation.strip(),
                })

        return {
            "all_correct": len(corrections) == 0,
            "corrections": corrections,
        }

    def _format_corrections(self, corrections: List[Dict[str, Any]]) -> str:
        """Format corrections as markdown notes."""
        lines = ["---", "", "**Corrections:**"]
        for i, c in enumerate(corrections, 1):
            lines.append(f"\n{i}. ~~\"{c['claim']}\"~~ → {c['correction']}")
        return "\n".join(lines)
```

**Step 2: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): implement _run_verification_stage

Stage 3 of multi-stage pipeline: verify description claims against frames
and append corrections if hallucinations are detected.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Implement _run_multi_stage_analysis() Orchestrator

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py` (add orchestrator method)

**Step 1: Add orchestrator method**

```python
    def _run_multi_stage_analysis(
        self,
        history: VisionHistory,
        frames_base64: List[str],
        scene_context: Dict[str, Any],
        frame_timestamps: List[float],
        options: Optional[AnalysisOptions] = None,
    ) -> VisionHistory:
        """
        Run multi-stage vision analysis pipeline.

        Stage 1: Classification - Commit to fundamental scene attributes
        Stage 2: Description - Generate description constrained by classification
        Stage 3: Verification - Verify claims against visual evidence

        Error handling:
        - Stage 1 fails → Abort with error
        - Stage 2 fails → Return classification only (partial results)
        - Stage 3 fails → Return unverified description

        Args:
            history: VisionHistory to update
            frames_base64: List of base64-encoded frame images
            scene_context: Dict with performers info
            frame_timestamps: List of timestamps for each frame
            options: Analysis options (quick_mode, skip_verification, etc.)

        Returns:
            Updated VisionHistory
        """
        options = options or AnalysisOptions()

        # Store options in history
        history.quick_mode = options.quick_mode
        history.skip_verification = options.skip_verification

        # Quick mode = existing single-pass behavior
        if options.quick_mode:
            self.log("Using quick mode (single-pass analysis)", "info")
            history.status = "describing"
            history.status_message = "Analyzing scene..."
            self._save_history(history)

            self._run_description_stage(history, frames_base64, scene_context, frame_timestamps)
            history.verification_status = "skipped"
            return history

        # Get custom prompts if provided
        custom_prompts = options.custom_prompts or {}

        # Stage 1: Classification
        try:
            classification = self._run_classification_stage(
                history,
                frames_base64,
                custom_prompt=custom_prompts.get("classification"),
            )
        except ValueError as e:
            # Stage 1 failure is fatal
            self.log(f"Multi-stage analysis aborted: {e}", "error")
            history.status = "error"
            history.status_message = f"Classification failed: {str(e)}"
            self._save_history(history)
            raise

        # Stage 2: Constrained Description
        try:
            description = self._run_constrained_description_stage(
                history,
                frames_base64,
                classification,
                scene_context,
                frame_timestamps,
                custom_constraints=custom_prompts.get("description_constraints"),
            )
        except Exception as e:
            # Stage 2 failure returns partial results (classification only)
            self.log(f"Description failed, returning classification only: {e}", "warning")
            history.status = "partial"
            history.status_message = "Classification complete. Description failed."
            history.description = f"[Description generation failed: {str(e)}]\n\nClassification completed successfully."
            history.verification_status = "skipped"
            self._save_history(history)
            return history

        # Stage 3: Verification (unless skipped)
        if options.skip_verification:
            self.log("Skipping verification stage (user option)", "info")
            history.verification_status = "skipped"
        else:
            self._run_verification_stage(
                history,
                frames_base64,
                description,
                custom_prompt=custom_prompts.get("verification"),
            )
            # Note: verification errors are handled internally and don't abort

        # Complete
        history.status = "complete"
        history.status_message = "Analysis complete"
        history.stage = "complete"
        self._save_history(history)

        return history
```

**Step 2: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): implement _run_multi_stage_analysis orchestrator

Chains classification → description → verification with error handling:
- Stage 1 fail: abort
- Stage 2 fail: return classification only
- Stage 3 fail: return unverified description

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Wire Up Multi-Stage as Default in run_analysis()

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py` (update run_analysis or equivalent entry point)

**Step 1: Find the main analysis entry point**

Search for where `_run_description_stage` is called and modify to use multi-stage by default.

**Step 2: Update to call multi-stage**

Find the method that orchestrates the analysis (likely `run_analysis` or similar) and update it to:

```python
# Replace direct call to _run_description_stage with:
options = AnalysisOptions(
    quick_mode=quick_mode,  # From request params
    skip_verification=skip_verification,
    frame_count=frame_count_override,
    custom_prompts=custom_prompts,
)

self._run_multi_stage_analysis(
    history,
    frames_base64,
    scene_context,
    frame_timestamps,
    options=options,
)
```

**Step 3: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot/.worktrees/multi-stage-vision && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): wire multi-stage analysis as default

Multi-stage is now the default. Quick mode available via options.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Add Settings Gear UI with Options

**Files:**
- Modify: `stash-copilot.js:9229-9303` (renderSidebarAnalyzeContent)

**Step 1: Update the settings button behavior**

Find `renderSidebarAnalyzeContent` and update the settings panel HTML. Replace the existing prompts section with a more comprehensive options panel:

```javascript
                <div class="stash-copilot-sidebar-options" style="display: none;">
                    <div class="stash-copilot-sidebar-section-header">
                        <span>Analysis Options</span>
                        <button class="stash-copilot-sidebar-options-reset" title="Reset All">↻</button>
                    </div>
                    <div class="stash-copilot-sidebar-options-content">
                        <label class="stash-copilot-sidebar-option">
                            <input type="checkbox" class="stash-copilot-option-quick-mode">
                            <span>Quick mode (faster)</span>
                        </label>
                        <label class="stash-copilot-sidebar-option">
                            <input type="checkbox" class="stash-copilot-option-skip-verification">
                            <span>Skip verification</span>
                        </label>
                        <div class="stash-copilot-sidebar-option-group">
                            <label>Frames</label>
                            <select class="stash-copilot-option-frame-count">
                                <option value="auto">Auto (smart)</option>
                                <option value="16">16 frames</option>
                                <option value="32">32 frames</option>
                                <option value="64">64 frames</option>
                            </select>
                        </div>
                        <div class="stash-copilot-sidebar-prompts-toggle">
                            <span>▶ Edit prompts</span>
                        </div>
                        <div class="stash-copilot-sidebar-prompts-content" style="display: none;">
                            <!-- Existing prompt editors -->
                        </div>
                    </div>
                </div>
```

**Step 2: Update settings button click handler**

In `setupSidebarAnalyzeListeners`, update the settings button to toggle the options panel:

```javascript
        // Settings button toggles options panel
        const settingsBtn = container.querySelector('.stash-copilot-sidebar-settings');
        const optionsPanel = container.querySelector('.stash-copilot-sidebar-options');
        if (settingsBtn && optionsPanel) {
            settingsBtn.addEventListener('click', () => {
                const isVisible = optionsPanel.style.display !== 'none';
                optionsPanel.style.display = isVisible ? 'none' : 'block';
            });
        }
```

**Step 3: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(ui): add settings gear with analysis options

Options panel with quick mode, skip verification, frame count selector,
and expandable prompts editor.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Add Classification Badges Display

**Files:**
- Modify: `stash-copilot.js` (add badge rendering function)
- Modify: `stash-copilot.css` (add badge styles)

**Step 1: Add badge rendering function**

```javascript
    function renderClassificationBadges(classification) {
        if (!classification) return '';

        const badges = [];

        // Scene type badge
        const sceneTypeIcons = {
            'solo_female': '👤',
            'solo_male': '👤',
            'couple': '👫',
            'threesome': '👥',
            'group': '👥👥',
        };
        const sceneIcon = sceneTypeIcons[classification.scene_type] || '🎬';
        const sceneLabel = classification.scene_type?.replace('_', ' ') || 'Unknown';
        badges.push(`<span class="stash-copilot-badge scene-type">${sceneIcon} ${sceneLabel}</span>`);

        // Content type badge
        if (classification.content_type === 'animated') {
            badges.push(`<span class="stash-copilot-badge content-type">🎨 Animated</span>`);
        } else if (classification.content_type === 'live_action') {
            badges.push(`<span class="stash-copilot-badge content-type">🎬 Live Action</span>`);
        }

        // Performer count
        if (classification.performer_count) {
            badges.push(`<span class="stash-copilot-badge performer-count">👤×${classification.performer_count}</span>`);
        }

        // Setting
        const settingIcons = {
            'indoor_only': '🏠',
            'outdoor_only': '🌳',
            'outdoor_to_indoor': '🌳→🏠',
            'indoor_to_outdoor': '🏠→🌳',
            'mixed': '🏠🌳',
        };
        if (classification.setting_progression && settingIcons[classification.setting_progression]) {
            badges.push(`<span class="stash-copilot-badge setting">${settingIcons[classification.setting_progression]}</span>`);
        }

        // Primary activity (text only, no emoji)
        if (classification.primary_activity && classification.primary_activity !== 'unknown') {
            const activity = classification.primary_activity.charAt(0).toUpperCase() +
                           classification.primary_activity.slice(1);
            badges.push(`<span class="stash-copilot-badge activity">${activity}</span>`);
        }

        return `<div class="stash-copilot-classification-badges">${badges.join('')}</div>`;
    }
```

**Step 2: Add CSS styles**

```css
.stash-copilot-classification-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 12px;
}

.stash-copilot-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(139, 92, 246, 0.15);
    color: #c4b5fd;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.stash-copilot-badge.scene-type {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border-color: rgba(16, 185, 129, 0.3);
}

.stash-copilot-badge.activity {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border-color: rgba(245, 158, 11, 0.3);
}
```

**Step 3: Update displayAnalysisResults to include badges**

Find where the description is displayed and add the badges above it.

**Step 4: Commit**

```bash
git add stash-copilot.js stash-copilot.css
git commit -m "feat(ui): add classification badges display

Shows scene type, content type, performer count, setting, and activity
as color-coded badges above the description.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Add Verification Status Display

**Files:**
- Modify: `stash-copilot.js` (add verification status rendering)
- Modify: `stash-copilot.css` (add verification styles)

**Step 1: Add verification status rendering**

```javascript
    function renderVerificationStatus(history) {
        if (!history.verification_status || history.verification_status === 'pending') {
            return '';
        }

        const statusConfig = {
            'verified': { icon: '✓', label: 'Verified', class: 'verified' },
            'corrections': {
                icon: '⚠',
                label: `${history.corrections?.length || 0} corrections`,
                class: 'corrections'
            },
            'skipped': { icon: '⚡', label: 'Unverified', class: 'skipped' },
            'failed': { icon: '⚡', label: 'Unverified', class: 'skipped' },
        };

        const config = statusConfig[history.verification_status] || statusConfig.skipped;

        let html = `
            <div class="stash-copilot-verification-status ${config.class}">
                <span class="stash-copilot-verification-icon">${config.icon}</span>
                <span class="stash-copilot-verification-label">${config.label}</span>
            </div>
        `;

        // Add expandable corrections if present
        if (history.verification_status === 'corrections' && history.corrections?.length > 0) {
            html += `
                <div class="stash-copilot-corrections-toggle" data-expanded="false">
                    <span>Show corrections ▼</span>
                </div>
                <div class="stash-copilot-corrections-list" style="display: none;">
                    ${history.corrections.map(c => `
                        <div class="stash-copilot-correction">
                            <span class="stash-copilot-correction-claim">❌ "${c.claim}"</span>
                            <span class="stash-copilot-correction-fix">→ ${c.correction}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        return html;
    }
```

**Step 2: Add CSS styles**

```css
.stash-copilot-verification-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 8px;
}

.stash-copilot-verification-status.verified {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
}

.stash-copilot-verification-status.corrections {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    cursor: pointer;
}

.stash-copilot-verification-status.skipped {
    background: rgba(107, 114, 128, 0.15);
    color: #9ca3af;
}

.stash-copilot-corrections-toggle {
    font-size: 11px;
    color: #9ca3af;
    cursor: pointer;
    margin-bottom: 8px;
}

.stash-copilot-corrections-list {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 12px;
}

.stash-copilot-correction {
    padding: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.stash-copilot-correction:last-child {
    border-bottom: none;
}

.stash-copilot-correction-claim {
    display: block;
    color: #f87171;
    text-decoration: line-through;
    font-size: 12px;
}

.stash-copilot-correction-fix {
    display: block;
    color: #6ee7b7;
    font-size: 12px;
    margin-top: 2px;
}
```

**Step 3: Add click handler for corrections toggle**

**Step 4: Commit**

```bash
git add stash-copilot.js stash-copilot.css
git commit -m "feat(ui): add verification status and corrections display

Shows verified/corrections/unverified status badge with expandable
corrections list when hallucinations were detected.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 12: Add Stage Progress Indicator

**Files:**
- Modify: `stash-copilot.js` (update progress display)

**Step 1: Update progress status mapping**

Find where status messages are displayed and add stage-aware messages:

```javascript
    function getStatusMessage(status) {
        const messages = {
            'pending': 'Initializing...',
            'extracting': 'Extracting frames...',
            'classifying': 'Stage 1/3: Classifying scene...',
            'describing': 'Stage 2/3: Generating description...',
            'verifying': 'Stage 3/3: Verifying claims...',
            'tagging': 'Suggesting tags...',
            'complete': 'Analysis complete',
            'partial': 'Partial results (see below)',
            'error': 'Analysis failed',
        };
        return messages[status] || status;
    }
```

**Step 2: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(ui): add stage progress indicator

Shows current stage (1/3, 2/3, 3/3) during multi-stage analysis.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 13: Add Custom Prompts Persistence

**Files:**
- Modify: `stash-copilot.js` (add localStorage persistence for custom prompts)

**Step 1: Add prompt storage functions**

```javascript
    const CUSTOM_PROMPTS_KEY = 'stash-copilot-custom-prompts';

    function loadCustomPrompts() {
        try {
            const stored = localStorage.getItem(CUSTOM_PROMPTS_KEY);
            return stored ? JSON.parse(stored) : {};
        } catch (e) {
            console.error('Failed to load custom prompts:', e);
            return {};
        }
    }

    function saveCustomPrompts(prompts) {
        try {
            localStorage.setItem(CUSTOM_PROMPTS_KEY, JSON.stringify(prompts));
        } catch (e) {
            console.error('Failed to save custom prompts:', e);
        }
    }

    function resetCustomPrompts() {
        localStorage.removeItem(CUSTOM_PROMPTS_KEY);
    }
```

**Step 2: Wire up to prompt editor UI**

**Step 3: Pass custom prompts to backend when analyzing**

**Step 4: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(ui): add custom prompts persistence

Custom prompts stored in localStorage and persist across scenes.
Reset button clears to defaults.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 14: Delete Existing Vision History Files

**Files:**
- Create: `scripts/clear_vision_history.py` (one-time migration script)

**Step 1: Create migration script**

```python
#!/usr/bin/env python3
"""Clear all existing vision history files for schema migration."""

import os
import glob
from pathlib import Path

def main():
    # Find assets directory
    plugin_dir = Path(__file__).parent.parent
    assets_dir = plugin_dir / "assets"

    if not assets_dir.exists():
        print("Assets directory not found")
        return

    # Find all vision history files
    pattern = str(assets_dir / "vision_history_*.json")
    history_files = glob.glob(pattern)

    if not history_files:
        print("No vision history files found")
        return

    print(f"Found {len(history_files)} vision history files")

    # Confirm deletion
    response = input("Delete all vision history files? [y/N] ")
    if response.lower() != 'y':
        print("Aborted")
        return

    # Delete files
    deleted = 0
    for f in history_files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

    print(f"Deleted {deleted} files")

if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/clear_vision_history.py
git commit -m "chore: add vision history migration script

One-time script to clear existing vision history files before
schema migration to multi-stage format.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 15: Integration Testing

**Files:**
- Test via Playwright MCP

**Step 1: Navigate to a scene page**

Use Playwright to navigate to scene 13368 (the problematic scene from the design doc).

**Step 2: Open settings and verify options appear**

**Step 3: Run analysis and verify stages**

- Watch for Stage 1/3, 2/3, 3/3 progress messages
- Verify classification badges appear
- Verify verification status appears
- Check logs for any errors

**Step 4: Test quick mode**

- Enable quick mode checkbox
- Run analysis
- Verify single-pass behavior

**Step 5: Document results**

Take screenshots and save to `tests/screenshots/`.

**Step 6: Commit test artifacts**

```bash
git add tests/screenshots/
git commit -m "test: add multi-stage vision analysis test results

Screenshots showing classification badges, verification status,
and stage progress for scene 13368.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Est. Complexity |
|------|-------------|-----------------|
| 1 | VisionHistory fields | Low |
| 2 | AnalysisOptions dataclass | Low |
| 3 | Multi-stage prompts | Low |
| 4 | Classification stage | Medium |
| 5 | Constrained description | Medium |
| 6 | Verification stage | Medium |
| 7 | Orchestrator | Medium |
| 8 | Wire up as default | Low |
| 9 | Settings gear UI | Medium |
| 10 | Classification badges | Medium |
| 11 | Verification status | Medium |
| 12 | Stage progress | Low |
| 13 | Prompt persistence | Low |
| 14 | Migration script | Low |
| 15 | Integration testing | Medium |
