# Tag Deduplication & Merging — Design

**Date:** 2026-02-17
**Status:** Approved

## Problem

Libraries accumulate duplicate tags that describe the same thing with different wording (e.g., "tit fuck" vs "Titty Fucking"). Manual cleanup is tedious without tooling to surface candidates and execute merges.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Detection method | OpenCLIP embedding cosine similarity | Already cached in SQLite, no new dependencies |
| Similarity threshold | 0.75 | Moderate — catches synonyms with manageable false positives |
| UI pattern | Pair-by-pair review (versus cards) | Low cognitive load per decision, interactive |
| Merge behavior | Reassign scenes, delete empty tag | Clean, standard merge |
| Keep target | Auto-suggest tag with more scenes | Saves time, user can override |
| Scope | Scenes only | Covers primary use case, extensible later |

## Architecture

### Backend

**New file:** `stash_ai/tasks/tag_dedup.py`

#### FindDuplicateTagsTask

1. Load all Stash tag embeddings from SQLite via `TagVocabulary`
2. Compute all-pairs cosine similarity (vectorized NumPy, upper triangle only)
3. Filter pairs with similarity >= 0.75
4. Exclude previously dismissed pairs (from `dismissed_tag_merges` table)
5. Fetch scene counts per candidate tag from Stash API
6. Sort by descending similarity
7. Pre-select higher-scene-count tag as "keep" target
8. Return sorted candidate list

**Return type:**
```python
class TagInfo(TypedDict):
    id: str
    name: str
    scene_count: int

class TagDedupCandidate(TypedDict):
    tag_a: TagInfo
    tag_b: TagInfo
    similarity: float
    suggested_keep: str  # "a" or "b"
```

#### MergeTagsTask

1. Query all scenes with the "remove" tag
2. For each scene: add "keep" tag (if not present), remove "remove" tag
3. Delete the now-empty tag via `TagDestroy`
4. Remove tag embedding from SQLite storage
5. Return `{success: bool, scenes_updated: int}`

#### DismissTagMergeTask

Store dismissed pair in `dismissed_tag_merges` table.

### SQLite Schema Addition

```sql
CREATE TABLE IF NOT EXISTS dismissed_tag_merges (
    tag_a_name TEXT NOT NULL,
    tag_b_name TEXT NOT NULL,
    dismissed_at TEXT NOT NULL,
    PRIMARY KEY (tag_a_name, tag_b_name)
)
```

Uses tag names (not IDs) as primary key — IDs can change if tags are deleted/recreated.

### Frontend

**UI:** Injected into Stash task page when "Find Duplicate Tags" task completes.

**Layout:**
```
┌─────────────────────────────────────────────────┐
│          Tag Deduplication Review                │
│          Pair 3 of 47  ·  87% similar           │
│                                                  │
│   ┌──────────────┐    VS    ┌──────────────┐    │
│   │  tit fuck    │          │ Titty Fucking │    │
│   │              │          │               │    │
│   │  23 scenes   │          │  5 scenes     │    │
│   │  * KEEP      │          │               │    │
│   └──────────────┘          └──────────────┘    │
│                                                  │
│     [<- Keep Left]  [Skip]  [Keep Right ->]      │
│                                                  │
│   Progress: ========--------  3/47               │
└─────────────────────────────────────────────────┘
```

**Interactions:**
- Keep Left / Keep Right: merge and advance
- Skip: dismiss pair, advance
- Keyboard: Left arrow = keep left, Right arrow = keep right, Down/S = skip

**Animations:** Chosen card stays, unchosen fades/slides out, next pair slides in.

**End screen:** Summary — "Merged X tags, skipped Y pairs, Z scenes updated."

### Data Flow

```
Stash task trigger
  -> FindDuplicateTagsTask.run()
  -> Returns candidate list as JSON
  -> JS renders pair review UI
  -> User action (keep/skip)
     -> Keep: MergeTagsTask.run(keep_id, remove_id)
     -> Skip: DismissTagMergeTask.run(tag_a_name, tag_b_name)
  -> Advance to next pair
  -> End screen with summary
```

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Tag already deleted | Skip pair silently, log warning |
| Scene update fails mid-merge | Stop, report partial state, don't delete tag |
| No embeddings cached | Prompt user to run "Build Tag Vocabulary" first |
| Zero candidates | Show "No duplicate tags found" message |
| Network/API timeout | Toast error, keep current pair for retry |

## Performance

- ~2000 tags: similarity matrix computed in < 2s (vectorized NumPy)
- Each merge: < 1s per scene reassignment
- Memory: < 1 GB (embeddings are 512-dim float32)
- Well within task performance budgets

## Testing

Manual Playwright MCP testing:
1. Trigger task, verify candidates render
2. Keep Left/Right, verify merge executes
3. Skip, verify dismissal persists across re-runs
4. Keyboard shortcuts match button actions
5. Edge: no embeddings -> helpful error
6. Edge: no duplicates -> "none found" message
