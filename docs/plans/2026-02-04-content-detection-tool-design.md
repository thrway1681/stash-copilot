# Content Detection Tool Design

**Date:** 2026-02-04
**Status:** Approved
**Feature:** Embedding-based content detection for scene vision analysis

## Overview

Add a new tool (`find_content`) to the scene vision analysis task that uses text-to-image similarity on existing 1fps frame embeddings to detect specific content types (starting with "creampie"). The tool returns candidate timestamps with similarity scores, and the LLM verifies candidates visually, reporting both embedding results and its own classification.

## Goals

1. **Timestamp extraction** - Find exact timestamps where content occurs
2. **Binary detection** - Determine if content is present with confidence scores
3. **Tag suggestion** - Suggest relevant tags (no auto-apply)
4. **Generalization** - Structure supports adding new content types easily

## Non-Goals

- Automatic tag application (user decides)
- Training custom models (uses existing CLIP text-image alignment)
- Real-time detection (batch analysis only)

## Technical Approach

### Detection Method

1. **Text embeddings** - Embed 2-3 visually descriptive phrases per content type
2. **Max similarity** - For each frame, compute similarity to all phrases, take max
3. **Threshold filtering** - Keep frames above threshold (~0.30 for text-to-image)
4. **Event clustering** - Group nearby frames (30s window) into events
5. **LLM verification** - LLM examines candidates, reports both embedding and visual results

### Why This Approach

- CLIP-style models align text and images in same embedding space
- Fewer, more visually-descriptive phrases work better than many synonyms
- Max similarity catches content if any visual aspect matches
- LLM verification catches false positives/negatives
- Showing both results builds trust and aids debugging

## Data Structures

### Content Detector Configuration

```python
# stash_ai/tools/content_detection.py

from dataclasses import dataclass
from typing import TypedDict

class GatingConditions(TypedDict, total=False):
    """Optional conditions from classification that enable this detector."""
    scene_type: list[str]      # e.g., ["couple", "threesome"]
    genders_present: list[str] # e.g., ["mixed"]
    activities: list[str]      # e.g., ["vaginal"] - boosts relevance

@dataclass
class ContentDetector:
    """Definition for a content type detector."""
    name: str                           # Internal identifier
    phrases: list[str]                  # 2-3 visually descriptive phrases
    threshold: float                    # Min similarity (e.g., 0.30)
    suggested_tag: str                  # Tag to suggest if detected
    requires: GatingConditions | None   # Optional gating conditions
    cluster_window: float = 30.0        # Seconds to group nearby frames

# Initial detector
CONTENT_DETECTORS: dict[str, ContentDetector] = {
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

### Tool Output Format

```python
{
    "detected": True,
    "content_type": "creampie",
    "threshold_used": 0.30,
    "events": [
        {
            "start_timestamp": 842.0,
            "end_timestamp": 858.0,
            "start_formatted": "14:02",
            "end_formatted": "14:18",
            "peak_timestamp": 847.0,
            "peak_formatted": "14:07",
            "peak_similarity": 0.34,
            "matched_phrase": "creampie dripping out",
            "frame_count": 12
        }
    ],
    "suggested_tag": "creampie",
    "note": "Please verify these candidates against the frames you can see."
}
```

## Tool Implementation

### FindContentTool Class

```python
class FindContentTool(BaseTool):
    """
    Searches scene frame embeddings to detect specific content types.
    Uses text-to-image similarity with CLIP-style embeddings.
    """

    name = "find_content"
    description = (
        "Search for specific content types in the scene using frame embeddings. "
        "Returns candidate timestamps with similarity scores. "
        "You should verify candidates against the frames you can see."
    )

    def __init__(self, scene_id: int, storage: EmbeddingStorage,
                 image_embedder: ImageEmbedder, available_detectors: list[str]):
        self.scene_id = scene_id
        self.storage = storage
        self.image_embedder = image_embedder
        self.available_detectors = available_detectors

    @property
    def parameters(self) -> dict:
        return {
            "content_type": {
                "type": "string",
                "enum": self.available_detectors,
                "description": "Type of content to search for",
                "required": True,
            },
            "custom_threshold": {
                "type": "number",
                "description": "Override default similarity threshold (0.0-1.0)",
                "required": False,
            },
        }

    def execute(self, content_type: str, custom_threshold: float | None = None) -> dict:
        detector = CONTENT_DETECTORS[content_type]
        threshold = custom_threshold or detector.threshold

        # 1. Embed text phrases
        phrase_embeddings = [
            self.image_embedder.embed_text(phrase)
            for phrase in detector.phrases
        ]

        # 2. Load all frame embeddings for scene
        frames = self.storage.get_all_frame_embeddings(self.scene_id)

        # 3. Compute max similarity per frame across all phrases
        frame_scores = []
        for frame in frames:
            similarities = [
                cosine_similarity(frame["embedding"], phrase_emb)
                for phrase_emb in phrase_embeddings
            ]
            best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            frame_scores.append({
                "timestamp": frame["timestamp"],
                "frame_index": frame["frame_index"],
                "similarity": similarities[best_idx],
                "matched_phrase": detector.phrases[best_idx],
            })

        # 4. Filter by threshold
        candidates = [f for f in frame_scores if f["similarity"] >= threshold]

        # 5. Cluster into events
        events = self._cluster_events(candidates, detector.cluster_window)

        # 6. Return top 5 events
        events = sorted(events, key=lambda e: e["peak_similarity"], reverse=True)[:5]

        return {
            "detected": len(events) > 0,
            "content_type": content_type,
            "threshold_used": threshold,
            "events": events,
            "suggested_tag": detector.suggested_tag if events else None,
            "note": "Please verify these candidates against the frames you can see.",
        }
```

### Event Clustering

```python
def _cluster_events(self, candidates: list[dict], window: float) -> list[dict]:
    """Group nearby candidate frames into events (30s window)."""
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["timestamp"])

    events = []
    current_event = [candidates[0]]

    for candidate in candidates[1:]:
        if candidate["timestamp"] - current_event[-1]["timestamp"] <= window:
            current_event.append(candidate)
        else:
            events.append(self._finalize_event(current_event))
            current_event = [candidate]

    events.append(self._finalize_event(current_event))
    return events

def _finalize_event(self, frames: list[dict]) -> dict:
    """Summarize a cluster of frames into an event."""
    peak_frame = max(frames, key=lambda f: f["similarity"])

    return {
        "start_timestamp": frames[0]["timestamp"],
        "end_timestamp": frames[-1]["timestamp"],
        "start_formatted": self._format_timestamp(frames[0]["timestamp"]),
        "end_formatted": self._format_timestamp(frames[-1]["timestamp"]),
        "peak_timestamp": peak_frame["timestamp"],
        "peak_formatted": self._format_timestamp(peak_frame["timestamp"]),
        "peak_similarity": round(peak_frame["similarity"], 3),
        "matched_phrase": peak_frame["matched_phrase"],
        "frame_count": len(frames),
    }
```

## Integration with Scene Vision

### Tool Availability (Gating)

Tool is only available when classification meets detector requirements:

```python
def _get_available_detectors(self) -> list[str]:
    """Filter content detectors based on classification results."""
    if not hasattr(self, '_classification') or not self._classification:
        return []

    classification = self._classification
    available = []

    for name, detector in CONTENT_DETECTORS.items():
        if detector.requires is None:
            available.append(name)
            continue

        reqs = detector.requires

        if "scene_type" in reqs:
            if classification.get("scene_type") not in reqs["scene_type"]:
                continue

        if "genders_present" in reqs:
            if classification.get("genders_present") not in reqs["genders_present"]:
                continue

        available.append(name)

    return available
```

### Prompt Instructions

When the tool is available, add these instructions to the LLM prompt:

```
**Content Detection:** The `find_content` tool can search for specific content
types using frame embeddings. When you use it:
1. Call the tool with the content type to search for
2. Review the returned candidate timestamps and similarity scores
3. **Verify the candidates** by examining the frames you can see
4. In your response, report BOTH:
   - The embedding results (similarity scores, timestamps)
   - Your own visual verification (agree/disagree, what you observe)
5. Your visual judgment takes precedence if you disagree with embeddings
```

### Tag Suggestion Integration

Tool results are passed to the tagging stage as additional context:

```python
def _build_tagging_context(self, history: VisionHistory) -> str:
    """Build context for tag suggestion stage."""
    context = [f"Scene Description:\n{history.description}"]

    content_detections = [
        tc for tc in history.tool_calls
        if tc.tool_name == "find_content" and tc.success
    ]

    if content_detections:
        context.append("\nContent Detection Results:")
        for tc in content_detections:
            result = tc.result
            if result.get("detected"):
                context.append(
                    f"- {result['content_type']}: Detected at "
                    f"{result['events'][0]['peak_formatted']} "
                    f"(similarity: {result['events'][0]['peak_similarity']}), "
                    f"suggested tag: {result.get('suggested_tag')}"
                )

    return "\n".join(context)
```

## Analysis Flow

```
Stage 0: Classification
    ↓
    Classification result: {scene_type: "couple", genders_present: "mixed", ...}
    ↓
_get_available_detectors() → ["creampie"]  (gating passes)
    ↓
Stage 1: Description (tools available)
    ↓
    LLM calls find_content("creampie")
    ↓
    Tool returns: {detected: true, events: [...], suggested_tag: "creampie"}
    ↓
    LLM verifies visually, includes both results in description
    ↓
Stage 3: Tagging
    ↓
    Tagging LLM sees description + tool results → suggests "creampie" tag
```

## File Changes

### New File
- `stash_ai/tools/content_detection.py` - Tool implementation, detectors, gating

### Modified Files
- `stash_ai/tasks/scene_vision.py`:
  - `_build_vision_tools()` - Add FindContentTool when detectors available
  - `_get_available_detectors()` - New method for gating logic
  - `_get_tool_instructions()` - Add content detection usage instructions
  - `_build_tagging_context()` - Include detection results for tag stage

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Similarity threshold | 0.30 | Text-to-image baseline, tunable per detector |
| Cluster window | 30 seconds | Groups action + aftermath |
| Max events returned | 5 | Top events by similarity |
| Phrases per detector | 2-3 | Visually descriptive, not synonyms |

## Future Extensibility

Adding new content types requires only adding to `CONTENT_DETECTORS`:

```python
CONTENT_DETECTORS["facial"] = ContentDetector(
    name="facial",
    phrases=["cum on face", "facial cumshot"],
    threshold=0.30,
    suggested_tag="facial",
    requires={
        "genders_present": ["mixed"],
    },
)
```

No code changes needed - gating and tool availability handled automatically.

## Testing Plan

1. **Unit tests** - Clustering logic, gating logic, tool parameter validation
2. **Integration tests** - Tool execution with mock embeddings
3. **Manual testing** - Run on known scenes with/without target content
4. **Threshold tuning** - Evaluate precision/recall at different thresholds
