# Content Detection Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an embedding-based content detection tool that finds specific content types (starting with "creampie") using text-to-image similarity on existing 1fps frame embeddings.

**Architecture:** New tool in `stash_ai/tools/content_detection.py` using CLIP text embeddings to search frame embeddings. Tool is gated by classification results and integrated into `scene_vision.py`'s description stage. LLM verifies candidates visually and reports both embedding results and its own judgment.

**Tech Stack:** Python, OpenCLIP (text embedding), numpy (cosine similarity), existing EmbeddingStorage

---

## Task 1: Create Content Detection Data Structures

**Files:**
- Create: `stash_ai/tools/content_detection.py`

**Step 1: Create the module with data structures**

```python
"""Content detection tool using text-to-image similarity on frame embeddings."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class GatingConditions(TypedDict, total=False):
    """Optional conditions from classification that enable this detector."""

    scene_type: List[str]       # e.g., ["couple", "threesome"]
    genders_present: List[str]  # e.g., ["mixed"]
    activities: List[str]       # e.g., ["vaginal"] - boosts relevance


@dataclass
class ContentDetector:
    """Definition for a content type detector."""

    name: str                                    # Internal identifier
    phrases: List[str]                           # 2-3 visually descriptive phrases
    threshold: float                             # Min similarity (e.g., 0.30)
    suggested_tag: str                           # Tag to suggest if detected
    requires: Optional[GatingConditions] = None  # Optional gating conditions
    cluster_window: float = 30.0                 # Seconds to group nearby frames


# Content detector definitions
CONTENT_DETECTORS: Dict[str, ContentDetector] = {
    "creampie": ContentDetector(
        name="creampie",
        phrases=["creampie dripping out", "cum leaking from vagina"],
        threshold=0.30,
        suggested_tag="creampie",
        requires={
            "scene_type": ["couple", "threesome", "group"],
            "genders_present": ["mixed"],
            "activities": ["vaginal"],
        },
    ),
}
```

**Step 2: Verify module imports correctly**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.tools.content_detection import CONTENT_DETECTORS, ContentDetector; print(f'Loaded {len(CONTENT_DETECTORS)} detectors')"`

Expected: `Loaded 1 detectors`

**Step 3: Commit**

```bash
git add stash_ai/tools/content_detection.py
git commit -m "feat(tools): add content detection data structures"
```

---

## Task 2: Add Gating Logic Function

**Files:**
- Modify: `stash_ai/tools/content_detection.py`

**Step 1: Add the gating function**

Add after `CONTENT_DETECTORS`:

```python
def get_available_detectors(classification: Optional[Dict[str, Any]]) -> List[str]:
    """
    Filter content detectors based on classification results.

    Args:
        classification: Classification dict from stage 1, or None

    Returns:
        List of detector names that pass gating conditions
    """
    if not classification:
        # Return detectors with no gating requirements
        return [
            name for name, detector in CONTENT_DETECTORS.items()
            if detector.requires is None
        ]

    available: List[str] = []

    for name, detector in CONTENT_DETECTORS.items():
        if detector.requires is None:
            # No gating = always available
            available.append(name)
            continue

        reqs = detector.requires
        passes_gating = True

        # Check scene_type requirement
        if "scene_type" in reqs:
            if classification.get("scene_type") not in reqs["scene_type"]:
                passes_gating = False

        # Check genders_present requirement
        if "genders_present" in reqs:
            if classification.get("genders_present") not in reqs["genders_present"]:
                passes_gating = False

        if passes_gating:
            available.append(name)

    return available
```

**Step 2: Test the gating logic**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.tools.content_detection import get_available_detectors

# Test: No classification
assert get_available_detectors(None) == []
print('No classification: PASS')

# Test: Classification without required fields
assert get_available_detectors({}) == []
print('Empty classification: PASS')

# Test: Classification with wrong scene_type
assert get_available_detectors({'scene_type': 'solo_female', 'genders_present': 'female_only'}) == []
print('Wrong scene_type: PASS')

# Test: Classification matching requirements
result = get_available_detectors({'scene_type': 'couple', 'genders_present': 'mixed'})
assert 'creampie' in result
print('Matching classification: PASS')

print('All gating tests passed!')
"`

Expected: All tests pass

**Step 3: Commit**

```bash
git add stash_ai/tools/content_detection.py
git commit -m "feat(tools): add content detection gating logic"
```

---

## Task 3: Add Event Clustering Functions

**Files:**
- Modify: `stash_ai/tools/content_detection.py`

**Step 1: Add clustering helper functions**

Add after `get_available_detectors`:

```python
def format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    if seconds < 0:
        return "0:00"

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class ContentEvent(TypedDict):
    """A detected content event (cluster of frames)."""

    start_timestamp: float
    end_timestamp: float
    start_formatted: str
    end_formatted: str
    peak_timestamp: float
    peak_formatted: str
    peak_similarity: float
    matched_phrase: str
    frame_count: int


def cluster_events(
    candidates: List[Dict[str, Any]],
    window: float,
) -> List[ContentEvent]:
    """
    Group nearby candidate frames into events.

    Args:
        candidates: Frames above threshold with timestamp/similarity/matched_phrase
        window: Seconds to group (e.g., 30.0)

    Returns:
        List of content events
    """
    if not candidates:
        return []

    # Sort by timestamp
    sorted_candidates = sorted(candidates, key=lambda c: c["timestamp"])

    events: List[ContentEvent] = []
    current_cluster: List[Dict[str, Any]] = [sorted_candidates[0]]

    for candidate in sorted_candidates[1:]:
        # If within window of last frame in current cluster, add to it
        if candidate["timestamp"] - current_cluster[-1]["timestamp"] <= window:
            current_cluster.append(candidate)
        else:
            # Finalize current cluster, start new one
            events.append(_finalize_event(current_cluster))
            current_cluster = [candidate]

    # Don't forget last cluster
    events.append(_finalize_event(current_cluster))

    return events


def _finalize_event(frames: List[Dict[str, Any]]) -> ContentEvent:
    """Summarize a cluster of frames into an event."""
    peak_frame = max(frames, key=lambda f: f["similarity"])

    return {
        "start_timestamp": frames[0]["timestamp"],
        "end_timestamp": frames[-1]["timestamp"],
        "start_formatted": format_timestamp(frames[0]["timestamp"]),
        "end_formatted": format_timestamp(frames[-1]["timestamp"]),
        "peak_timestamp": peak_frame["timestamp"],
        "peak_formatted": format_timestamp(peak_frame["timestamp"]),
        "peak_similarity": round(peak_frame["similarity"], 3),
        "matched_phrase": peak_frame["matched_phrase"],
        "frame_count": len(frames),
    }
```

**Step 2: Test clustering logic**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.tools.content_detection import cluster_events, format_timestamp

# Test format_timestamp
assert format_timestamp(65) == '1:05'
assert format_timestamp(3661) == '1:01:01'
assert format_timestamp(0) == '0:00'
print('format_timestamp: PASS')

# Test empty candidates
assert cluster_events([], 30.0) == []
print('Empty candidates: PASS')

# Test single candidate
result = cluster_events([{'timestamp': 100.0, 'similarity': 0.35, 'matched_phrase': 'test'}], 30.0)
assert len(result) == 1
assert result[0]['frame_count'] == 1
print('Single candidate: PASS')

# Test clustering within window
candidates = [
    {'timestamp': 100.0, 'similarity': 0.30, 'matched_phrase': 'phrase1'},
    {'timestamp': 110.0, 'similarity': 0.35, 'matched_phrase': 'phrase2'},
    {'timestamp': 120.0, 'similarity': 0.32, 'matched_phrase': 'phrase1'},
]
result = cluster_events(candidates, 30.0)
assert len(result) == 1  # All within 30s window
assert result[0]['frame_count'] == 3
assert result[0]['peak_similarity'] == 0.35  # Highest similarity
print('Clustering within window: PASS')

# Test separate clusters
candidates = [
    {'timestamp': 100.0, 'similarity': 0.30, 'matched_phrase': 'phrase1'},
    {'timestamp': 200.0, 'similarity': 0.35, 'matched_phrase': 'phrase2'},
]
result = cluster_events(candidates, 30.0)
assert len(result) == 2  # 100s apart > 30s window
print('Separate clusters: PASS')

print('All clustering tests passed!')
"`

Expected: All tests pass

**Step 3: Commit**

```bash
git add stash_ai/tools/content_detection.py
git commit -m "feat(tools): add content detection event clustering"
```

---

## Task 4: Implement FindContentTool

**Files:**
- Modify: `stash_ai/tools/content_detection.py`

**Step 1: Add imports and tool class**

Add imports at top of file:

```python
from typing import TYPE_CHECKING

import numpy as np

from .base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from ..embeddings.storage import EmbeddingStorage
    from ..embeddings.base import BaseImageEmbeddingProvider
```

Add tool class after clustering functions:

```python
class FindContentTool(BaseTool):
    """
    Search scene frame embeddings to detect specific content types.

    Uses text-to-image similarity with CLIP-style embeddings.
    Returns candidate timestamps with similarity scores for LLM verification.
    """

    def __init__(
        self,
        stash: Any,  # StashInterface, but we don't use it
        scene_id: int,
        storage: "EmbeddingStorage",
        image_embedder: "BaseImageEmbeddingProvider",
        available_detectors: List[str],
    ):
        """
        Initialize the content detection tool.

        Args:
            stash: StashInterface (required by base class)
            scene_id: Scene being analyzed
            storage: EmbeddingStorage for loading frame embeddings
            image_embedder: Image embedding provider (must support text embedding)
            available_detectors: List of detector names available for this scene
        """
        super().__init__(stash)
        self._scene_id = scene_id
        self._storage = storage
        self._image_embedder = image_embedder
        self._available_detectors = available_detectors

    @property
    def name(self) -> str:
        return "find_content"

    @property
    def description(self) -> str:
        detector_list = ", ".join(self._available_detectors)
        return (
            f"Search for specific content types ({detector_list}) in the scene using "
            "frame embeddings. Returns candidate timestamps with similarity scores. "
            "You should verify candidates against the frames you can see and report "
            "both the embedding results AND your visual verification."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            {
                "name": "content_type",
                "type": "string",
                "description": "Type of content to search for",
                "required": True,
                "enum": self._available_detectors,
            },
            {
                "name": "custom_threshold",
                "type": "number",
                "description": "Override default similarity threshold (0.0-1.0). Optional.",
                "required": False,
                "enum": None,
            },
        ]

    def execute(
        self,
        content_type: str,
        custom_threshold: Optional[float] = None,
    ) -> ToolResult:
        """
        Search for content type in scene frame embeddings.

        Args:
            content_type: Type of content to search for (from available_detectors)
            custom_threshold: Optional override for similarity threshold

        Returns:
            ToolResult with detected events or error
        """
        # Validate content_type
        if content_type not in self._available_detectors:
            return {
                "success": False,
                "data": None,
                "error": f"Unknown content type: {content_type}. Available: {self._available_detectors}",
            }

        detector = CONTENT_DETECTORS[content_type]
        threshold = custom_threshold if custom_threshold is not None else detector.threshold

        try:
            # 1. Embed text phrases
            phrase_embeddings: List[np.ndarray] = []
            for phrase in detector.phrases:
                result = self._image_embedder.embed_text(phrase)
                phrase_embeddings.append(np.array(result["embedding"], dtype=np.float32))

            # 2. Load all frame embeddings for scene
            frames = self._storage._load_all_frames_for_scene(self._scene_id)

            if not frames:
                return {
                    "success": False,
                    "data": None,
                    "error": f"No frame embeddings found for scene {self._scene_id}",
                }

            # 3. Compute max similarity per frame across all phrases
            frame_scores: List[Dict[str, Any]] = []
            for frame in frames:
                frame_emb = np.array(frame["embedding"], dtype=np.float32)

                # Normalize for cosine similarity (if not already normalized)
                frame_norm = frame_emb / (np.linalg.norm(frame_emb) + 1e-8)

                similarities = []
                for phrase_emb in phrase_embeddings:
                    phrase_norm = phrase_emb / (np.linalg.norm(phrase_emb) + 1e-8)
                    similarity = float(np.dot(frame_norm, phrase_norm))
                    similarities.append(similarity)

                best_idx = int(np.argmax(similarities))
                frame_scores.append({
                    "timestamp": frame["timestamp"],
                    "frame_index": frame["frame_index"],
                    "similarity": similarities[best_idx],
                    "matched_phrase": detector.phrases[best_idx],
                })

            # 4. Filter by threshold
            candidates = [f for f in frame_scores if f["similarity"] >= threshold]

            # 5. Cluster into events
            events = cluster_events(candidates, detector.cluster_window)

            # 6. Return top 5 events sorted by peak similarity
            events = sorted(events, key=lambda e: e["peak_similarity"], reverse=True)[:5]

            return {
                "success": True,
                "data": {
                    "detected": len(events) > 0,
                    "content_type": content_type,
                    "threshold_used": threshold,
                    "events": events,
                    "suggested_tag": detector.suggested_tag if events else None,
                    "note": "Please verify these candidates against the frames you can see.",
                },
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Content detection failed: {str(e)}",
            }
```

**Step 2: Verify imports work**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.tools.content_detection import FindContentTool; print('Import successful')"`

Expected: `Import successful`

**Step 3: Commit**

```bash
git add stash_ai/tools/content_detection.py
git commit -m "feat(tools): implement FindContentTool"
```

---

## Task 5: Integrate Tool into Scene Vision - Build Tools

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py`

**Step 1: Add import**

Find the imports section (around line 30-60) and add:

```python
from ..tools.content_detection import FindContentTool, get_available_detectors
```

**Step 2: Modify `_build_vision_tools` method**

Find `_build_vision_tools` (around line 1894) and modify to add the content detection tool. After the `FindSimilarFramesTool` block, add:

```python
                # Add content detection tool if classification enables any detectors
                # and we have an image embedder that supports text embedding
                if hasattr(self, '_current_classification') and self._current_classification:
                    available_detectors = get_available_detectors(self._current_classification)
                    if available_detectors and self.image_embedder:
                        try:
                            # Verify the embedder supports text (CLIP-style models do)
                            if hasattr(self.image_embedder, 'embed_text'):
                                tools.append(FindContentTool(
                                    self.stash,
                                    int(scene_id),
                                    storage,
                                    self.image_embedder,
                                    available_detectors,
                                ))
                                self.log(f"Content detection tool enabled for: {available_detectors}", "debug")
                        except Exception as e:
                            self.log(f"Could not enable content detection tool: {e}", "warning")
```

The full modified method should look like:

```python
    def _build_vision_tools(
        self,
        scene_id: str,
        frame_timestamps: List[float],
    ) -> List[BaseTool]:
        """
        Build vision-specific tools for the current analysis.

        Args:
            scene_id: Scene being analyzed
            frame_timestamps: List of timestamps for displayed frames

        Returns:
            List of tool instances (may be empty if model doesn't support tools)
        """
        tools: List[BaseTool] = []

        # Always add the frame timestamp lookup tool
        tools.append(GetFrameTimestampTool(self.stash, frame_timestamps))

        # Add similar frame search if we have frame embeddings
        if self.image_embedding_config:
            model_key = self.image_embedding_config.model_key
            storage = EmbeddingStorage(model_key=model_key)

            # Only add if this scene has frame embeddings
            if storage.has_frame_embeddings(int(scene_id)):
                tools.append(FindSimilarFramesTool(
                    self.stash,
                    int(scene_id),
                    frame_timestamps,
                    storage,
                ))

                # Add content detection tool if classification enables any detectors
                # and we have an image embedder that supports text embedding
                if hasattr(self, '_current_classification') and self._current_classification:
                    available_detectors = get_available_detectors(self._current_classification)
                    if available_detectors and self.image_embedder:
                        try:
                            # Verify the embedder supports text (CLIP-style models do)
                            if hasattr(self.image_embedder, 'embed_text'):
                                tools.append(FindContentTool(
                                    self.stash,
                                    int(scene_id),
                                    storage,
                                    self.image_embedder,
                                    available_detectors,
                                ))
                                self.log(f"Content detection tool enabled for: {available_detectors}", "debug")
                        except Exception as e:
                            self.log(f"Could not enable content detection tool: {e}", "warning")

                self.log("Vision tools enabled: get_frame_timestamp, find_similar_frames", "debug")
            else:
                self.log("Vision tools enabled: get_frame_timestamp (no frame embeddings for similarity search)", "debug")
        else:
            self.log("Vision tools enabled: get_frame_timestamp", "debug")

        return tools
```

**Step 3: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('Import successful')"`

Expected: `Import successful`

**Step 4: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): integrate content detection tool into build_vision_tools"
```

---

## Task 6: Store Classification for Tool Access

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py`

**Step 1: Initialize `_current_classification` in `__init__`**

Find the `__init__` method (around line 785) and add after `self._image_embedder: Optional[Any] = None`:

```python
        self._current_classification: Optional[Dict[str, Any]] = None
```

**Step 2: Set classification in `_run_classification_stage`**

Find `_run_classification_stage` (around line 2345) and after `history.classification = {...}` (around line 2424), add:

```python
            # Store classification for tool gating
            self._current_classification = history.classification
```

**Step 3: Set classification in `_run_constrained_description_stage`**

Find `_run_constrained_description_stage` (around line 2470). At the start of the method body (after the docstring), add:

```python
        # Store classification for tool gating during description stage
        self._current_classification = classification
```

**Step 4: Clear classification after analysis**

Find `_run_multi_stage_analysis` (around line 3564). Before the final `return history` (around line 3666), add:

```python
        # Clear classification after analysis completes
        self._current_classification = None
```

**Step 5: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('Import successful')"`

Expected: `Import successful`

**Step 6: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): store classification for content detection tool gating"
```

---

## Task 7: Add Tool Instructions to Prompt

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py`

**Step 1: Modify `_get_tool_instructions`**

Find `_get_tool_instructions` (around line 2065) and add content detection instructions. After the existing tool instructions loop, add a section for content detection:

```python
        # Add content detection specific instructions if tool is available
        has_content_tool = any(t.name == "find_content" for t in tools)
        if has_content_tool:
            lines.extend([
                "",
                "**Content Detection:** The `find_content` tool can search for specific content "
                "types using frame embeddings. When you use it:",
                "1. Call the tool with the content type to search for",
                "2. Review the returned candidate timestamps and similarity scores",
                "3. **Verify the candidates** by examining the frames you can see",
                "4. In your response, report BOTH:",
                "   - The embedding results (similarity scores, timestamps)",
                "   - Your own visual verification (agree/disagree, what you observe)",
                "5. Your visual judgment takes precedence if you disagree with embeddings",
            ])
```

The full modified method should look like:

```python
    def _get_tool_instructions(self, tools: List[BaseTool]) -> str:
        """
        Generate instructions for using available tools.

        Args:
            tools: List of available tools

        Returns:
            Formatted instruction string to append to prompt
        """
        if not tools:
            return ""

        lines = [
            "",
            "---",
            "**Timestamp Tools Available:**",
            "You have access to tools for accurate timestamps:",
        ]

        for tool in tools:
            lines.append(f"- `{tool.name}`: {tool.description}")

        lines.extend([
            "",
            "**IMPORTANT:** Before mentioning any timestamp in your description, use `get_frame_timestamp(frame_index)` "
            "to get the exact time. Frame 1 is the first frame shown.",
        ])

        # Add content detection specific instructions if tool is available
        has_content_tool = any(t.name == "find_content" for t in tools)
        if has_content_tool:
            lines.extend([
                "",
                "**Content Detection:** The `find_content` tool can search for specific content "
                "types using frame embeddings. When you use it:",
                "1. Call the tool with the content type to search for",
                "2. Review the returned candidate timestamps and similarity scores",
                "3. **Verify the candidates** by examining the frames you can see",
                "4. In your response, report BOTH:",
                "   - The embedding results (similarity scores, timestamps)",
                "   - Your own visual verification (agree/disagree, what you observe)",
                "5. Your visual judgment takes precedence if you disagree with embeddings",
            ])

        lines.extend([
            "---",
            "",
        ])

        return "\n".join(lines)
```

**Step 2: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('Import successful')"`

Expected: `Import successful`

**Step 3: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): add content detection instructions to tool prompt"
```

---

## Task 8: Add Content Detection Results to Tagging Context

**Files:**
- Modify: `stash_ai/tasks/scene_vision.py`

**Step 1: Find the tag suggestion method**

Search for where tags are suggested. Look for `_suggest_tags` or similar method that builds context for the tagging LLM.

Run: `grep -n "def _suggest_tags\|def _run_tag\|tag.*stage" ~/.stash/plugins/stash-copilot/stash_ai/tasks/scene_vision.py | head -20`

**Step 2: Add content detection context to tagging**

Find where the tagging prompt is built (likely in `_suggest_tags` or similar). Add after the description is included in context:

```python
        # Add content detection results from tool calls
        content_detections = [
            tc for tc in history.tool_calls
            if tc.tool_name == "find_content" and tc.success
        ]

        if content_detections:
            context_parts.append("\n**Content Detection Results:**")
            for tc in content_detections:
                result = tc.result.get("data", {}) if isinstance(tc.result, dict) else {}
                if result.get("detected"):
                    events = result.get("events", [])
                    if events:
                        context_parts.append(
                            f"- {result.get('content_type')}: Detected at {events[0]['peak_formatted']} "
                            f"(similarity: {events[0]['peak_similarity']}), "
                            f"suggested tag: {result.get('suggested_tag')}"
                        )
```

Note: The exact location depends on how the tagging context is built. Look at the actual code structure.

**Step 3: Verify syntax**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.tasks.scene_vision import SceneVisionTask; print('Import successful')"`

Expected: `Import successful`

**Step 4: Commit**

```bash
git add stash_ai/tasks/scene_vision.py
git commit -m "feat(vision): include content detection results in tagging context"
```

---

## Task 9: Add Module Exports

**Files:**
- Modify: `stash_ai/tools/__init__.py`

**Step 1: Add exports**

Add to the `__init__.py`:

```python
from .content_detection import (
    ContentDetector,
    CONTENT_DETECTORS,
    FindContentTool,
    get_available_detectors,
)
```

**Step 2: Verify imports**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "from stash_ai.tools import FindContentTool, get_available_detectors; print('Exports working')"`

Expected: `Exports working`

**Step 3: Commit**

```bash
git add stash_ai/tools/__init__.py
git commit -m "feat(tools): export content detection module"
```

---

## Task 10: Manual Testing

**Files:** None (testing only)

**Step 1: Run the full import chain**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.tools.content_detection import (
    CONTENT_DETECTORS,
    ContentDetector,
    FindContentTool,
    get_available_detectors,
    cluster_events,
    format_timestamp,
)
from stash_ai.tasks.scene_vision import SceneVisionTask

print('All imports successful')
print(f'Detectors available: {list(CONTENT_DETECTORS.keys())}')
print(f'Creampie detector phrases: {CONTENT_DETECTORS[\"creampie\"].phrases}')
"`

Expected: All imports successful, shows detector info

**Step 2: Test gating with real classification structure**

Run: `cd ~/.stash/plugins/stash-copilot && uv run python -c "
from stash_ai.tools.content_detection import get_available_detectors

# Simulated multi-stage classification result
classification = {
    'content_type': 'live_action',
    'scene_type': 'couple',
    'performer_count': 2,
    'genders_present': 'mixed',
    'setting_progression': 'indoor_only',
    'activities': ['oral', 'vaginal'],
    'has_intro_segments': False,
}

available = get_available_detectors(classification)
print(f'Available detectors for couple scene: {available}')
assert 'creampie' in available, 'Creampie should be available for couple with mixed genders'

# Test solo scene (should not have creampie)
solo_classification = {
    'scene_type': 'solo_female',
    'genders_present': 'female_only',
}
solo_available = get_available_detectors(solo_classification)
print(f'Available detectors for solo scene: {solo_available}')
assert 'creampie' not in solo_available, 'Creampie should NOT be available for solo scenes'

print('Gating tests passed!')
"`

Expected: Tests pass

**Step 3: Document in commit**

```bash
git add -A
git commit -m "test: verify content detection integration" --allow-empty
```

---

## Summary

After completing all tasks, the content detection tool will be:

1. **Defined** in `stash_ai/tools/content_detection.py` with:
   - `ContentDetector` dataclass for detector definitions
   - `CONTENT_DETECTORS` dict with "creampie" detector
   - `get_available_detectors()` for gating logic
   - `cluster_events()` for grouping nearby frames
   - `FindContentTool` implementing `BaseTool`

2. **Integrated** into `stash_ai/tasks/scene_vision.py`:
   - Tool added in `_build_vision_tools()` when classification passes gating
   - Classification stored in `_current_classification` for tool access
   - Tool instructions added in `_get_tool_instructions()`
   - Detection results passed to tagging stage

3. **Behavior**:
   - Tool only appears when classification shows couple/threesome/group with mixed genders
   - LLM can call `find_content("creampie")` during description stage
   - Tool returns timestamps, similarity scores, and suggested tag
   - LLM verifies candidates visually and reports both results
   - Tagging stage sees detection results for informed tag suggestions
