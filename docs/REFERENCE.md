# Technical Reference

Detailed technical documentation for stash-copilot internals. For coding guidance, see `CLAUDE.md`.

## Recommendation System

### O-Moment Embeddings

**Location**: `stash_ai/tasks/o_moment_extractor.py`, `stash_ai/tasks/embed_o_moments.py`

O-moment embeddings enable the "Peak Moments" recommendation mode by extracting frames around O markers.

**Data Source**:
```sql
-- O markers from scene_markers table
SELECT sm.id, sm.scene_id, sm.seconds, sm.end_seconds
FROM scene_markers sm
JOIN tags t ON sm.primary_tag_id = t.id
WHERE t.name = 'O'
```

**Processing Flow**:
1. Query O markers for exact playback positions
2. Extract frames in window around each marker (+/- 60s by default)
3. Generate image embeddings for extracted frames
4. Average frame embeddings into single O-moment embedding
5. Store in `o_moment_embeddings` table

**Storage Schema**:
```sql
CREATE TABLE o_moment_embeddings (
    scene_id INTEGER NOT NULL,
    o_event_index INTEGER NOT NULL,
    marker_id INTEGER NOT NULL,
    center_timestamp REAL NOT NULL,
    window_seconds REAL NOT NULL,
    embedding BLOB NOT NULL,
    frame_count INTEGER NOT NULL,
    model_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (scene_id, o_event_index, model_key)
);
```

**Plugin Settings**:
| Setting | Description | Default |
|---------|-------------|---------|
| `o_moment_window` | Total window size in seconds | 120 |
| `o_moment_frames` | Frames per window | 12 |
| `o_tag_name` | O marker tag name | "O" |

### User Profile Building

```python
from stash_ai.recommendations import UserProfileBuilder, RecommendationConfig

config = RecommendationConfig(
    top_scenes_for_profile=20,  # Use top 20 engaged scenes
    weights={"o_count": 20.0, "view_count": 2.0, "play_duration": 1.0},
)

builder = UserProfileBuilder(storage=embedding_storage)
profile = builder.build_profile(config)

# profile.profile_embedding is a weighted average of scene embeddings
# Each scene's weight is proportional to its engagement score
```

### Task Integration

Three tasks are available in Stash UI:
- **Get Recommendations (Discover)**: Unwatched scenes similar to preferences
- **Get Recommendations (Re-watch)**: Watched scenes ranked by engagement + similarity
- **Get Recommendations (Time Decay)**: Discovery mode with recency weighting

### Output Format

Results are saved to `assets/recommendations_{request_id}.json`:

```json
{
  "status": "complete",
  "mode": "discover_new",
  "scoring_method": "base_weighted",
  "profile": {
    "contributing_scenes": [123, 456],
    "total_engagement_score": 45.5,
    "scene_count": 20
  },
  "results": [
    {
      "scene_id": 789,
      "similarity_score": 0.85,
      "engagement_score": 0.0,
      "combined_score": 0.85,
      "scene": { "id": 789, "title": "...", "performers": [], "tags": [] }
    }
  ]
}
```

### Plugin Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `rec_top_scenes` | Scenes for profile building | 20 |
| `rec_o_weight` | O-count weight | 20.0 |
| `rec_view_weight` | Replay count weight (per replay, views beyond first) | 2.0 |
| `rec_duration_weight` | Play duration weight (per hour) | 1.0 |
| `rec_rating_weight` | Rating weight (per star, only if rated) | 1.5 |
| `rec_time_decay_days` | Half-life for time decay | 30 |
