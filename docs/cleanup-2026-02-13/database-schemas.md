# Database Schemas

**Date:** 2026-02-13
**Source:** Exploration agent analysis

## Overview

The plugin maintains **3 SQLite databases** in `assets/`:

| Database | Size | Schema Ver | Status | Purpose |
|----------|------|------------|--------|---------|
| `stash_copilot.sqlite` | 19 GB | 10 | **ACTIVE** | Main embeddings, preferences, tags |
| `stash_copilot_ViT-bigG-14-dense.sqlite` | 204 MB | 4 | STALE | Experimental dense frames |
| `stash_copilot_ViT-H-14-sparse.sqlite` | 432 MB | 4 | STALE | Multi-model experiments |

**Empty/Unused:**
- `embeddings.db` (root) - 0 bytes
- `assets/embeddings.db` - 0 bytes
- `stash_ai/embeddings/embeddings.db` - 0 bytes

---

## Primary Database: stash_copilot.sqlite

### Table: scene_embeddings
**Purpose:** Scene-level embeddings for similarity search

| Column | Type | Description |
|--------|------|-------------|
| scene_id | INTEGER | PK with model_key |
| model_key | TEXT | e.g., "openclip:ViT-H-14" |
| composite_embedding | BLOB | Packed float32 array |
| visual_embedding | BLOB | Vision model output |
| metadata_embedding | BLOB | Text model output |
| visual_description | TEXT | Scene description |
| dimensions | INTEGER | Embedding dimensions |
| created_at / updated_at | TEXT | Timestamps |

**Stats:** 12,812 rows

### Table: frame_embeddings
**Purpose:** Per-frame embeddings at 1 FPS

| Column | Type | Description |
|--------|------|-------------|
| scene_id | INTEGER | PK with model_key, frame_index |
| frame_index | INTEGER | 0-based index |
| timestamp | REAL | Seconds into video |
| embedding | BLOB | Packed float32 array |

**Stats:** 4,073,927 frames from 12,760 scenes (~319 frames/scene avg)

### Table: performer_embeddings
**Purpose:** Aggregated performer visual embeddings

| Column | Type | Description |
|--------|------|-------------|
| performer_id | INTEGER | PK with model_key |
| embedding | BLOB | Average of scene embeddings |
| contributing_scenes | INTEGER | Scenes used for average |
| total_engagement_score | REAL | Weighted engagement |
| visual_description | TEXT | Performer description |

**Stats:** 313 performers

### Table: frame_tag_coverage
**Purpose:** Tag gap detection - which frames match which tags

| Column | Type | Description |
|--------|------|-------------|
| scene_id | INTEGER | PK with frame_index, model_key |
| frame_index | INTEGER | Frame index |
| best_tag | TEXT | Most similar tag |
| best_similarity | REAL | 0-1 similarity score |
| is_covered | INTEGER | Boolean coverage flag |

**Stats:** 4,073,278 rows

### Table: preference_comparisons
**Purpose:** User preference learning data

| Column | Type | Description |
|--------|------|-------------|
| scene_a_id | INTEGER | First comparison scene |
| scene_b_id | INTEGER | Second comparison scene |
| winner_id | INTEGER | User's preferred scene |
| phase | TEXT | Learning phase |
| response_time_ms | INTEGER | User response time |
| session_id | TEXT | Preference session |

**Stats:** 1,296 comparisons

### Table: taste_clusters
**Purpose:** User preference clusters for taste profiling

| Column | Type | Description |
|--------|------|-------------|
| cluster_id | INTEGER | PK with model_key |
| centroid | BLOB | Cluster center embedding |
| scene_ids | TEXT | Comma-separated IDs |
| auto_label | TEXT | Generated cluster name |
| engagement_share | REAL | Share of total engagement |

**Stats:** 15 clusters

### Table: tag_embeddings
**Purpose:** Tag text embeddings for similarity matching

| Column | Type | Description |
|--------|------|-------------|
| text | TEXT | Tag name (PK with model_key) |
| embedding | BLOB | Packed float32 array |
| source | TEXT | "user_library" or "llm_generated" |

**Stats:** 507 tags

### Other Tables

| Table | Purpose | Rows |
|-------|---------|------|
| scene_umap_coords | 2D/3D visualization coordinates | 12,756 |
| frame_embedding_metadata | Scene-level frame extraction metadata | 12,362 |
| preference_model_state | Bradley-Terry model state | 1 |
| preference_sessions | Preference learning sessions | 73 |
| schema_info | Schema version and config | 2 |

---

## Data Statistics

| Metric | Value |
|--------|-------|
| Total scenes embedded | 12,812 |
| Total frames embedded | 4,073,927 |
| Performers embedded | 313 |
| Tags embedded | 507 |
| Preference comparisons | 1,296 |
| Taste clusters | 15 |
| Primary model | openclip:ViT-H-14 |
| Total database size | ~19 GB |

---

## Recommendations

1. **Consolidate databases** - Migrate any needed data from stale DBs to primary
2. **Delete empty DBs** - Remove 0-byte `embeddings.db` files
3. **Consider data cleanup** - Stale experimental DBs may be deletable
4. **Index optimization** - Review indexes on large tables (frame_embeddings)
