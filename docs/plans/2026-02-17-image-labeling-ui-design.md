# Image Labeling UI — Design Document

**Date**: 2026-02-17
**Status**: Approved
**Goal**: Build a human-in-the-loop annotation tool to create training datasets for fine-tuning OpenCLIP or similar embedding models.

## Overview

A dedicated full-width page in Stash for labeling images with tags. The system uses **uncertainty sampling** (active learning) to prioritize images where the model is least confident, maximizing annotation value per label. Labels are stored as structured tags and auto-converted to captions at export time in **WebDataset** format for OpenCLIP training.

## Architecture: Frontend-Heavy Preload (Approach B)

Backend prepares a large batch (200 frames ranked by uncertainty). Frontend loads the full batch and handles navigation/labeling locally, syncing labels back periodically. Sampling adapts between sessions, not mid-session.

```
User opens page
  → JS triggers "Prepare Labeling Session" task
  → Backend: sync tag vocabulary, compute uncertainty, select 200 frames
  → Backend writes session JSON
  → JS loads JSON, user labels locally
  → JS syncs labels back periodically (every 30 labels or on page unload)
  → User clicks "Export" → backend generates WebDataset tar
```

## Data Model

All annotation data lives in `assets/stash_copilot.sqlite`.

### New Tables

#### `labeling_sessions`

Tracks annotation sessions.

```sql
CREATE TABLE labeling_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',   -- active | completed | exported
    sampling_method TEXT NOT NULL,            -- uncertainty | random | clustered
    batch_size INTEGER NOT NULL,
    total_frames INTEGER NOT NULL,
    labeled_count INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    config_json TEXT                          -- session-specific settings
);
```

#### `frame_annotations`

Per-frame tag labels — the core training data.

```sql
CREATE TABLE frame_annotations (
    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES labeling_sessions(session_id),
    scene_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    image_source TEXT NOT NULL DEFAULT 'extracted_frame',  -- 'extracted_frame' | 'stash_image'
    tag_text TEXT NOT NULL,
    tag_source TEXT NOT NULL,              -- 'suggested' | 'manual' | 'existing'
    label TEXT NOT NULL,                   -- 'confirmed' | 'rejected' | 'skipped'
    similarity_score REAL,                 -- original model similarity (NULL for manual)
    labeled_at TEXT NOT NULL,
    UNIQUE(session_id, scene_id, frame_index, tag_text)
);
```

#### `labeling_progress`

Tracks which frames have been seen across sessions.

```sql
CREATE TABLE labeling_progress (
    scene_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    image_source TEXT NOT NULL DEFAULT 'extracted_frame',
    session_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | labeled | skipped
    PRIMARY KEY (scene_id, frame_index, session_id)
);
```

### Design Notes

- `image_source` field future-proofs for Stash image library integration (v2)
- `tag_source` tracks provenance: model-suggested vs human-added vs existing Stash tags
- `similarity_score` preserved for analyzing model improvement over time

## Backend Tasks

### 1. `PrepareLabelingSession`

**Location**: `stash_ai/tasks/labeling.py`
**Mode**: `prepare_labeling_session`

#### Algorithm (Uncertainty Sampling)

1. **Sync tag vocabulary**: Pull current Stash tags + unembedded manual tags from previous sessions. Generate CLIP text embeddings for any new tags.
2. Load all frame embeddings from `frame_embeddings` table
3. Load all tag embeddings from `tag_embeddings` table
4. Compute cosine similarity matrix (frames × tags)
5. For each frame, calculate **uncertainty score**: count tags with similarity in the confusion zone (0.25–0.35)
6. Exclude frames already labeled in previous sessions (from `labeling_progress`)
7. Select top `batch_size` frames by uncertainty score
8. For each selected frame, include:
   - Frame file path
   - Top ~10 suggested tags with similarity scores
   - Existing Stash tags for the parent scene
   - Scene metadata (title, ID)
9. Create session record in `labeling_sessions`
10. Write batch to `assets/labeling_session_{session_id}.json`

#### Tag Vocabulary Sync

Runs at the start of every session to catch tags from all sources:

- **Stash UI tags**: Query Stash database for new tags not yet in `tag_embeddings`
- **Manual labeling tags**: Query `frame_annotations` for `tag_source='manual'` without embeddings
- **Curated phrases**: Already embedded (Tier 2/3 from tag_vocabulary.py)

### 2. `SyncAnnotations`

**Mode**: `sync_annotations`
**Triggered by**: JS frontend periodically (every 30 labels or on page unload)

- Input: JSON blob of annotations from frontend
- Bulk insert into `frame_annotations`
- Update `labeling_progress` status
- Update `labeling_sessions.labeled_count` and `skipped_count`

### 3. `ExportDataset`

**Mode**: `export_dataset`
**Triggered by**: User clicks "Export" button

1. Query all frames with at least one `confirmed` annotation
2. For each frame:
   - Copy image file to tar
   - Generate caption from confirmed tags using template
   - Optionally generate negative caption from rejected tags
3. Package as WebDataset `.tar`
4. Include `metadata.json` sidecar
5. Save to `assets/exports/dataset_{timestamp}.tar`

#### Caption Generation

Template (v1):
```
"a scene featuring {tag1}, {tag2}, and {tag3}"
```

#### Negative Labels (Optional)

Rejected tags exported as `_neg.txt` files for contrastive training with hard negatives.

#### Export Metadata

```json
{
  "created_at": "2026-02-17T14:30:00",
  "total_images": 1847,
  "total_tags": 45,
  "sessions_included": ["session_abc", "session_def"],
  "caption_template": "a scene featuring {tags}",
  "tag_stats": { "tag_name": count, ... }
}
```

## UI Design

### Page Location

Dedicated full-width page: `/plugin/stash-copilot/label`
Injected via Stash's plugin page system.

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER BAR                                                 │
│  [← Back]  "Image Labeling"   Session: 47/200  [⚙ Settings]│
│  [Single View] [Grid View]    Progress: ████████░░ 24%      │
├───────────────────────────────────┬─────────────────────────┤
│                                   │  TAG PANEL              │
│                                   │                         │
│                                   │  Suggested Tags         │
│         IMAGE AREA                │  ┌─────────────────────┐│
│                                   │  │ ✓ blowjob    0.32  ││
│    (large image in single view    │  │ ✗ brunette    0.29  ││
│     or 2x3 grid in grid view)    │  │ ? POV         0.27  ││
│                                   │  │ ? bedroom     0.26  ││
│                                   │  └─────────────────────┘│
│                                   │                         │
│                                   │  Existing Scene Tags    │
│                                   │  [oral] [amateur] [HD]  │
│                                   │                         │
│                                   │  Add Tag                │
│                                   │  [type to search...   ] │
│                                   │                         │
├───────────────────────────────────┤  ────────────────────── │
│  NAVIGATION                       │  ACTIONS                │
│  [← Prev]  12 / 200  [Next →]   │  [Skip] [Save & Next]   │
│  Keyboard: ← →  or  A D          │  [S]     [Enter]        │
└───────────────────────────────────┴─────────────────────────┘
```

### View Modes

#### Single View (Default)

- Large image: ~65% width
- Tag panel: ~35% width
- Suggested tags with similarity scores and confirm/reject toggles
- Existing scene tags as read-only pills (context)
- Autocomplete tag input at bottom

#### Grid View

- 2×3 grid (6 images)
- Click image to select → tag panel updates
- Visual indicators: green border (labeled), yellow (in progress), gray (pending)
- Bulk action: "Apply tag X to all visible"

### Tag Interaction

**Suggested tags** — three states per tag:
- ✓ **Confirmed** (green): tag applies to this image
- ✗ **Rejected** (red): tag does NOT apply
- ? **Undecided** (gray, default): not yet labeled

**Autocomplete input**:
- Filters against full tag vocabulary (all tiers + user tags)
- Fuzzy matching (e.g., "cowg" → "reverse cowgirl")
- Selected tags appear as confirmed
- Unmatched text → "Create new tag: [text]" option

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` / `D` | Next image |
| `←` / `A` | Previous image |
| `Enter` | Save labels & next |
| `S` | Skip image |
| `1`–`9` | Toggle suggested tag (undecided → confirmed → rejected) |
| `/` | Focus autocomplete input |
| `Escape` | Unfocus input |
| `G` | Toggle grid/single view |

### Progress & Stats

Header bar: position (47/200), progress bar, session stats on hover (confirmed/rejected/skipped/manual counts).

## Image Sources

### v1: Extracted Frames

Frames from `assets/embedded_frames/scene_<ID>/frame_<INDEX>.jpg`. Already have CLIP embeddings in `frame_embeddings` table — tag suggestions work immediately.

### v2: Stash Image Library

Images from Stash's image library (galleries, covers). Will require:
- New queries to Stash database for image metadata
- Embedding generation for Stash images
- Extended `image_source` field to distinguish sources

## Active Learning Loop

```
Session N:
  1. Sync tag vocabulary (Stash tags + manual tags from previous sessions)
  2. Compute uncertainty scores for all unlabeled frames
  3. Present top 200 uncertain frames
  4. User labels: confirm/reject suggested tags, add manual tags
  5. Labels synced to SQLite

Session N+1:
  1. New manual tags get CLIP embeddings
  2. Previously labeled frames excluded
  3. Uncertainty recalculated with updated vocabulary
  4. Model suggests new manual tags for other frames
  → Fewer manual additions needed over time

Export:
  1. All confirmed tags → auto-generated captions
  2. All rejected tags → optional negative labels
  3. WebDataset .tar for OpenCLIP fine-tuning
```

## CSS Theming

Following the plugin's design language:
- **Theme color**: To be determined (distinct from existing cyan/green/purple/orange themes)
- **Class prefix**: `.stash-copilot-label-*`
- **Styling**: Modern AI aesthetic with glow, gradients, animations per CLAUDE.md spec

## Plugin Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `label_batch_size` | Frames per labeling session | 200 |
| `label_uncertainty_low` | Lower bound of confusion zone | 0.25 |
| `label_uncertainty_high` | Upper bound of confusion zone | 0.35 |
| `label_suggested_tags` | Max suggested tags per frame | 10 |
| `label_caption_template` | Caption generation template | `"a scene featuring {tags}"` |

## File Structure (New Files)

```
stash_ai/tasks/labeling.py           # PrepareSession, SyncAnnotations, ExportDataset
stash-copilot.js                      # Extended: labeling page injection + rendering
stash-copilot.css                     # Extended: .stash-copilot-label-* classes
stash-copilot.yml                     # Extended: new task registrations
assets/exports/                       # WebDataset export output directory
assets/labeling_session_*.json        # Session batch files
```
