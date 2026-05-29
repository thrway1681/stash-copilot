# RFC 001: O-Moment Embeddings for Scene Recommendations

## Summary

Create embeddings from video frames around O events to power a new recommendation mode that finds scenes with visually similar "peak moments" - the parts of scenes that triggered engagement.

## Motivation

The current recommendation system builds user profiles from whole-scene embeddings weighted by engagement metrics (O count, play count, duration). While effective, this approach treats each scene as a single unit.

A scene with multiple O events likely has specific moments that resonate with the user. By extracting frames from the video around these peak moments and creating embeddings specifically from those frames, we can:

1. **Find scenes with similar climactic content** - Not just similar overall themes, but similar peak moments
2. **Better capture user preferences** - A user might O to a specific act/position/performer action, not the whole scene
3. **Enable "More moments like this"** - Recommend scenes based on what specifically triggered engagement

## Technical Challenge: Missing Playback Position

### Current Data Model

The Stash `scenes_o_dates` table stores:
```sql
CREATE TABLE scenes_o_dates (
    scene_id INTEGER NOT NULL,
    o_date TEXT NOT NULL,  -- Wall-clock timestamp when O button pressed
    FOREIGN KEY (scene_id) REFERENCES scenes(id)
);
```

**Critical limitation**: We have the wall-clock time when the user pressed O, but NOT the playback position in the video at that moment.

### Approaches to Estimate Video Position

#### Approach A: Correlation Heuristic (Recommended for MVP)

Estimate video position by correlating O event timestamps with viewing session data:

```
O_timestamp = when O button was pressed (wall clock)
view_start = scene start time (from scenes_view_dates)
estimated_position = O_timestamp - view_start
```

**Caveats**:
- User may have paused, seeked, or rewound
- Multiple viewing sessions blur the data
- Position could exceed scene duration (user paused for extended time)

**Mitigation**: Validate that `estimated_position` is within scene duration bounds, discard outliers.

#### Approach B: Statistical Fallback

When correlation fails, use statistical assumptions:
- O events typically occur in the latter portion of scenes
- Extract frames from the final 30-50% of the scene as a fallback

#### Approach C: Future Stash Enhancement (Ideal)

Propose a Stash enhancement to track playback position with O events:
```sql
-- Proposed schema addition
ALTER TABLE scenes_o_dates ADD COLUMN playback_position REAL;  -- seconds
```

This would require upstream Stash changes but would enable precise frame extraction.

## Proposed Implementation

### Phase 1: Data Collection & Storage

#### New Types (`stash_ai/recommendations/types.py`)

```python
class OEventData(TypedDict):
    """O event with estimated video position."""
    scene_id: int
    o_date: str  # ISO timestamp
    estimated_position: Optional[float]  # seconds into video, None if unknown
    confidence: float  # 0-1, how confident we are in position estimate


class OMomentEmbedding(TypedDict):
    """Embedding derived from frames around an O event."""
    scene_id: int
    o_event_index: int  # Which O event for this scene (0-indexed)
    center_timestamp: float  # Center of extraction window (seconds)
    window_seconds: float  # Total window size (e.g., 120 for +/- 60s)
    embedding: List[float]
    frame_count: int  # How many frames were averaged
    created_at: str
```

#### New Storage Table

Add to `EmbeddingStorage` schema:
```sql
CREATE TABLE o_moment_embeddings (
    scene_id INTEGER NOT NULL,
    o_event_index INTEGER NOT NULL,
    center_timestamp REAL NOT NULL,
    window_seconds REAL NOT NULL,
    embedding BLOB NOT NULL,
    frame_count INTEGER NOT NULL,
    model_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (scene_id, o_event_index, model_key)
);
```

### Phase 2: Frame Extraction Around O Events

#### New Module: `stash_ai/tasks/o_moment_extractor.py`

```python
class OMomentExtractor:
    """Extract and embed frames around estimated O event positions."""

    def __init__(
        self,
        frame_extractor: FrameExtractor,
        embedder: BaseEmbeddingProvider,
        window_seconds: float = 120.0,  # +/- 60 seconds
        frames_per_window: int = 12,  # 1 frame per 10 seconds in window
    ): ...

    def estimate_o_positions(
        self,
        scene_id: int,
    ) -> List[OEventData]:
        """
        Estimate video positions for O events using correlation heuristic.

        1. Get all O events for scene from scenes_o_dates
        2. Get all view events from scenes_view_dates
        3. For each O event, find closest preceding view event
        4. Calculate: position = o_timestamp - view_timestamp
        5. Validate position is within scene duration
        6. Return list with confidence scores
        """
        ...

    def extract_o_moment_frames(
        self,
        scene_id: int,
        video_path: str,
        center_position: float,
        duration: float,
    ) -> List[str]:
        """
        Extract frames from window around estimated O position.

        Returns list of base64-encoded frame images.
        """
        ...

    def create_o_moment_embedding(
        self,
        scene_id: int,
        o_event_index: int,
        frames_base64: List[str],
    ) -> OMomentEmbedding:
        """Create averaged embedding from O-moment frames."""
        ...
```

### Phase 3: New Recommendation Mode

#### Add to `RecommendationMode` enum:

```python
class RecommendationMode(Enum):
    DISCOVER_NEW = "discover_new"
    REWATCH_FAVORITES = "rewatch"
    O_MOMENTS = "o_moments"  # NEW: Find scenes with similar peak moments
```

#### New Engine Method: `_recommend_by_o_moments()`

```python
def _recommend_by_o_moments(
    self,
    config: RecommendationConfig,
) -> List[RecommendationResult]:
    """
    Find scenes with similar O-moment embeddings.

    1. Get user's O-moment embeddings (from scenes they've O'd to)
    2. Build profile by averaging top O-moment embeddings
    3. Find scenes with O-moment embeddings similar to profile
    4. Optionally: Also search whole-scene embeddings for diversity
    """
    ...
```

### Phase 4: Task & UI Integration

#### New Stash Task: `embed_o_moments`

```python
class EmbedOMomentsTask:
    """Generate embeddings for O-moment frames across library."""

    def embed_scene_o_moments(self, scene_id: int) -> Dict[str, Any]:
        """Extract and embed O moments for a single scene."""
        ...

    def embed_all_o_moments(self) -> Dict[str, Any]:
        """Process all scenes with O events."""
        ...
```

#### Plugin Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `o_moment_window` | Seconds before/after O position | 60 |
| `o_moment_frames` | Frames per window | 12 |
| `o_moment_min_confidence` | Minimum position confidence | 0.5 |

#### UI Tab/Button

Add to Recommendations tab:
- New mode toggle: "Discover" / "Re-watch" / "**Peak Moments**"
- Peak Moments mode uses O-moment embeddings to find similar climactic content

## Implementation Plan

### Step 1: Schema & Types (1-2 days)
- [ ] Add `OEventData`, `OMomentEmbedding` types to `types.py`
- [ ] Add `o_moment_embeddings` table to `EmbeddingStorage`
- [ ] Write migration for existing databases

### Step 2: Position Estimation (2-3 days)
- [ ] Implement `estimate_o_positions()` correlation logic
- [ ] Add confidence scoring based on session timing
- [ ] Handle edge cases (position > duration, no matching view event)
- [ ] Add statistical fallback for low-confidence estimates

### Step 3: Frame Extraction (2-3 days)
- [ ] Extend `FrameExtractor` for position-based extraction (not just interval)
- [ ] Implement `OMomentExtractor` class
- [ ] Add caching for O-moment frames (separate from scene frames)

### Step 4: Embedding Generation (1-2 days)
- [ ] Implement `create_o_moment_embedding()`
- [ ] Create `EmbedOMomentsTask` with progress reporting
- [ ] Add "Embed O Moments" task to Stash plugin tasks

### Step 5: Recommendation Engine (2-3 days)
- [ ] Add `O_MOMENTS` mode to `RecommendationMode`
- [ ] Implement `_recommend_by_o_moments()` in engine
- [ ] Add O-moment profile building logic
- [ ] Create combined scoring (O-moment similarity + whole-scene similarity)

### Step 6: UI Integration (2-3 days)
- [ ] Add "Peak Moments" mode toggle in Recs tab
- [ ] Update API endpoints for new mode
- [ ] Add visual indicators for O-moment-based recommendations

### Step 7: Testing & Documentation (2 days)
- [ ] Unit tests for position estimation
- [ ] Integration tests for full pipeline
- [ ] Update CLAUDE.md with new feature documentation

## Open Questions

1. **Window size**: Is +/- 60 seconds optimal? Should it be configurable per-scene based on duration?

2. **Multiple O events**: How to weight multiple O events in the same scene?
   - Average all O-moment embeddings?
   - Weight by recency?
   - Keep separate and match against any?

3. **Low-confidence handling**: When position estimate confidence is low:
   - Skip the O event?
   - Use fallback (final 30% of scene)?
   - Extract multiple windows and average?

4. **Scene coverage**: What percentage of O'd scenes will have usable position estimates?
   - Need to analyze real user data to understand accuracy

5. **Stash upstream**: Should we propose the `playback_position` column addition to Stash?

## Alternatives Considered

### Alternative 1: Manual O-Moment Markers

Let users manually mark "peak moments" in scenes via a UI overlay.
- **Pro**: Accurate positions
- **Con**: Requires user effort, low adoption

### Alternative 2: VLM-Based Peak Detection

Use a VLM to analyze scene frames and identify "climactic" moments automatically.
- **Pro**: No position estimation needed
- **Con**: Subjective, may not match user preferences, expensive

### Alternative 3: Audio Analysis

Detect peaks in audio (moans, intensity changes) to identify climactic moments.
- **Pro**: Could be accurate for certain content types
- **Con**: Requires audio processing, doesn't generalize well

## Success Metrics

1. **Coverage**: % of O'd scenes with usable position estimates
2. **Accuracy**: User feedback on recommendation relevance
3. **Engagement**: Click-through rate on O-moment recommendations vs standard recommendations

## References

- Current recommendation system: `stash_ai/recommendations/`
- Frame extraction: `stash_ai/tasks/frame_extractor.py`
- Embedding storage: `stash_ai/embeddings/storage.py`
- Stash schema: `tests/fixtures/schema.py`
