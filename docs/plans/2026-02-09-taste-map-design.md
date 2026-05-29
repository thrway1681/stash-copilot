# Taste Map: Multi-Cluster Recommendation Engine + Interactive Visualization

**Date:** 2026-02-09
**Status:** Approved

## Overview

Replace the single-profile recommendation engine with a multi-cluster taste profiling system. Users' engaged scenes are clustered by embedding similarity, auto-labeled via CLIP text embeddings, and visualized as an interactive 2D scatter plot. Recommendations are queried per-cluster with proportional sampling for diversity.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Full stack at once | Backend + UI + recs integration shipped together |
| Integration | Replace single-profile | One system, no toggle. Clusters become THE recommendation engine |
| Visualization | ECharts (Apache) | Best balance of aesthetics, features, and ease of integration. CDN, MIT licensed |
| Aesthetic | Neon/Cyberpunk | Glowing points, dark backgrounds, saturated neon colors. Matches existing purple gradient theme |
| Label source | Tags + curated phrases | Existing tags plus ~150 descriptive phrases. Inline custom phrase testing from UI |
| Custom vocab | Inline from Taste Map | Users type phrases in the tag match panel to test similarity against clusters |
| Sidebar recs | Untouched | Focus on AI Insights modal only |

## Architecture

Four layers:

1. **Clustering Engine** (`stash_ai/recommendations/clusters.py`) — K-Means + silhouette score, replaces single-profile
2. **Tag Embedding Vocabulary** (`stash_ai/embeddings/tag_vocabulary.py`) — OpenCLIP text encoder for tags + curated phrases
3. **Cluster Recommendation Engine** (`stash_ai/recommendations/cluster_engine.py`) — Per-cluster querying with proportional sampling
4. **Taste Map UI** (new AI Insights tab) — ECharts scatter plot + cluster sidebar + tag match panel

## Design System: Neon/Cyberpunk

Shared visual language for all data visualizations across the plugin.

### Color Palette

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Primary | Electric Purple | `#8b5cf6` | Recs theme, default accent, cluster borders |
| Secondary | Neon Cyan | `#06b6d4` | Hover states, secondary data series |
| Success | Neon Green | `#10b981` | "New" badges, positive signals, discover theme |
| Warning | Neon Orange | `#f59e0b` | Scene-based recs, engagement highlights |
| Danger | Neon Pink | `#ec4899` | High engagement (O-count), hot clusters |
| Info | Electric Blue | `#3b82f6` | Re-watch badges, cool clusters |

### Cluster Colors (auto-assigned, up to 8)

```
Cluster 1: #8b5cf6 (purple)
Cluster 2: #06b6d4 (cyan)
Cluster 3: #10b981 (green)
Cluster 4: #f59e0b (orange)
Cluster 5: #ec4899 (pink)
Cluster 6: #3b82f6 (blue)
Cluster 7: #f43f5e (rose)
Cluster 8: #a855f7 (violet)
```

### Surface Colors

- Background: `#0a0a0f` (near-black with blue tint)
- Panel: `rgba(15, 15, 25, 0.9)` (dark with translucency)
- Border: `rgba(139, 92, 246, 0.2)` (faint purple)
- Hover border: `rgba(139, 92, 246, 0.5)` (brighter purple)

### Effects

- Glow on hover: `box-shadow: 0 0 12px rgba(139, 92, 246, 0.4)`
- Data points: radial gradient from bright center to transparent edge (neon orbs)
- Active cluster: pulsing glow animation (subtle, 2s cycle)
- Transitions: 200ms ease for all interactive state changes
- Chart grid lines: `rgba(255, 255, 255, 0.05)` (barely visible)

### ECharts Theme

Registered as a custom theme so every chart uses it automatically:

```javascript
const COPILOT_ECHARTS_THEME = {
    backgroundColor: 'transparent',
    color: ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#3b82f6', '#f43f5e', '#a855f7'],
    textStyle: { color: 'rgba(255,255,255,0.7)', fontFamily: 'system-ui, monospace' },
    // axis, tooltip, legend styles all following the neon aesthetic
};
echarts.registerTheme('stash-copilot', COPILOT_ECHARTS_THEME);
```

Every chart instantiation uses `echarts.init(container, 'stash-copilot')`.

## Clustering Engine

### Algorithm

1. Load visual embeddings for top 200 engaged scenes (sorted by engagement score)
2. Run K-Means for k=2 through k=8
3. For each k, compute silhouette score
4. Pick the k with highest silhouette score (fall back to k=3 if scores are flat)
5. For each cluster: compute centroid (weighted average by engagement), collect member scenes, sum engagement stats

### Output — TasteProfile

```python
@dataclass
class TasteCluster:
    cluster_id: int
    centroid: NDArray[np.float32]
    scene_ids: list[int]
    engagement_total: float
    engagement_share: float
    auto_label: str
    tag_similarities: list[TagMatch]
    coords_2d: list[tuple[float, float]]

@dataclass
class TasteProfile:
    clusters: list[TasteCluster]
    umap_coords: dict[int, tuple[float, float]]
    optimal_k: int
    silhouette_score: float
```

Cluster centroids are weighted by engagement within the cluster.

## Tag Embedding Vocabulary

### Three tiers

**Tier 1: Existing Stash tags** — pulled from database. User's own taxonomy, tried first.

**Tier 2: Content descriptors (~100 curated phrases)** — grouped by category:

| Category | Example phrases |
|----------|----------------|
| Act/Position | "oral sex", "doggy style", "missionary", "cowgirl riding", "solo masturbation" |
| Setting | "bedroom scene", "bathroom shower", "outdoor nature", "hotel room" |
| Style | "POV perspective", "close-up intimate", "wide shot", "professional studio", "amateur homemade" |
| Aesthetic | "high energy music video", "slow romantic", "fast cuts montage", "softcore artistic" |
| Body type | "petite slim woman", "curvy voluptuous woman", "athletic fit body" |
| Features | "blonde hair", "brunette", "redhead", "tattoos and piercings", "lingerie" |
| Group | "solo performer", "couple", "threesome", "multiple performers" |
| Interactive | "funscript interactive", "VR virtual reality" |

**Tier 3: Compound phrases (~50 combinations)** — more specific niche descriptors.

### Labeling process

1. Compute cosine similarity between cluster centroid and ALL vocabulary embeddings
2. Top 2 matches joined with " / " become the auto-label
3. Top 8 shown in tag match panel
4. Users can type custom phrases for instant similarity testing (embedded on the fly, <100ms)

### Caching

Vocabulary embeddings computed once, stored in `tag_embeddings` SQLite table. Recomputed only if vocabulary or embedding model changes.

## Recommendation Engine — Cluster-Based Querying

### Replaces single-profile approach

**Old:** 1 profile embedding → cosine similarity vs all scenes → ranked results

**New:** K cluster centroids → K similarity queries → merge with proportional sampling → ranked results

### Proportional merge

If Cluster A has 50% engagement share, Cluster B 30%, Cluster C 20%, then out of every 10 recommendations, ~5 come from A, ~3 from B, ~2 from C.

Implementation: round-robin sampling weighted by engagement share (or user-adjusted weight), deduplicating as we go. A scene appearing in multiple cluster results gets its best score.

### User-adjustable weights

Cluster sidebar has a weight slider per cluster (default = engagement share). Users can override to bias recommendations.

### Exclusion

Toggling "Exclude" on a cluster removes it from querying. Its engagement share redistributes proportionally.

### Integration with existing modes

- **All/New/Re-watch filter:** Per-cluster query against appropriate candidate set
- **Time decay:** Applied to engagement scores before clustering

## Taste Map UI

### Location

New tab in AI Insights modal, between existing tabs.

### Layout

Two-panel split: scatter plot (70% width) + cluster sidebar (30% width). Tag match panel slides up from bottom of scatter plot when a cluster is selected.

### Scatter Plot (ECharts)

- Each scene is a neon orb point (radial gradient, bright core, soft glow)
- Size scales with engagement score (min 6px, max 20px)
- Color matches assigned cluster
- Unwatched non-profile scenes: dim gray dots (2px, 10% opacity) for full landscape
- Hover: dark frosted tooltip with thumbnail, title, play count, O count
- Click: navigate to scene page
- Lasso select: draw freeform region, shows aggregate stats
- Zoom/pan: mouse wheel and drag
- Click cluster label: highlights cluster, dims others

### Cluster Sidebar

Vertical stack of cluster cards sorted by engagement share:

```
+---------------------------+
| * PMV / High Energy       |  <- cluster color dot + auto-label (editable)
| 18 scenes . 42%           |  <- count + engagement share
| > 234  O 45               |  <- aggregate play + O count
| [thumb][thumb][thumb]      |  <- 3 representative scenes
| ================== 100%   |  <- weight slider
| [Exclude]                  |  <- toggle
+---------------------------+
```

Active cluster card has glowing border in cluster color.

### Tag Match Panel

Slides up when cluster selected. Shows top 8 tag/phrase matches as horizontal bars:

```
TAG MATCHES FOR: * PMV / High Energy
"music video compilation" ......... 0.82
"fast editing montage" ............ 0.79
"multiple performers" ............. 0.74
```

Bar width = similarity score, colored with cluster color.

Includes text input for custom phrase testing — type a phrase, see instant similarity score.

### Build Button

"Build Taste Map" button triggers backend computation. Results polled via JSON file (same pattern as recommendations).

## Backend Pipeline

**Task:** `Build Taste Map` — single task, ~10-15 seconds

```
Step 1: Load engaged scenes + embeddings (reuse EngagementCalculator)
Step 2: K-Means clustering + silhouette scoring
Step 3: UMAP projection -> 2D coordinates (all embedded scenes)
Step 4: Tag embedding matching -> auto-labels per cluster
Step 5: Save results to assets/taste_map_{request_id}.json
```

## Data Flow

```
User clicks "Build Taste Map"
    |
Backend task fires:
    1. Load top 200 engaged scene embeddings
    2. Run UMAP -> 2D coordinates
    3. Run K-Means -> cluster assignments
    4. Embed all tags via CLIP text encoder
    5. Match cluster centroids to tag embeddings -> auto-labels
    6. Save results to JSON file
    |
Frontend polls for results
    |
Renders scatter plot + cluster sidebar + tag panel
    |
User interacts (rename, test phrases, adjust weights, exclude)
    |
Modified clusters saved to backend
    |
"Generate" on Recs tab uses cluster-based profiles
```

## Storage

| Data | Location | Persistence |
|------|----------|-------------|
| Cluster assignments | `taste_clusters` SQLite table | Until rebuild |
| Cluster weights/labels | `taste_clusters` SQLite table | Survives rebuild via centroid matching |
| UMAP 2D coordinates | `scene_umap_coords` SQLite table | Until rebuild |
| Tag vocabulary embeddings | `tag_embeddings` SQLite table | Until model change |
| Custom user phrases | `tag_embeddings` table (source='user') | Permanent |
| Taste map JSON (display) | `assets/taste_map_{id}.json` | Ephemeral (polling) |

User-edited labels survive rebuilds: if user renamed "Cluster 2" to "My Solo Favs", on rebuild the system matches new clusters to old clusters by centroid similarity and inherits custom labels.

## SQLite Schema

```sql
CREATE TABLE taste_clusters (
    cluster_id INTEGER NOT NULL,
    model_key TEXT NOT NULL,
    centroid BLOB NOT NULL,
    scene_ids TEXT NOT NULL,
    engagement_total REAL NOT NULL,
    engagement_share REAL NOT NULL,
    auto_label TEXT NOT NULL,
    user_label TEXT,
    weight_override REAL,
    excluded INTEGER DEFAULT 0,
    tag_matches TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (cluster_id, model_key)
);

CREATE TABLE scene_umap_coords (
    scene_id INTEGER NOT NULL,
    model_key TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    cluster_id INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (scene_id, model_key)
);

CREATE TABLE tag_embeddings (
    text TEXT NOT NULL,
    model_key TEXT NOT NULL,
    embedding BLOB NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (text, model_key)
);
```

## File Map

### New files

| File | Purpose | ~Lines |
|------|---------|--------|
| `stash_ai/recommendations/clusters.py` | K-Means clustering, silhouette scoring, centroid computation, TasteProfile | ~250 |
| `stash_ai/embeddings/tag_vocabulary.py` | Curated phrase list, OpenCLIP text encoding, vocabulary caching | ~200 |
| `stash_ai/tasks/taste_map.py` | Task entry point — orchestrates pipeline, saves JSON | ~150 |
| `stash_ai/recommendations/cluster_engine.py` | Per-cluster querying, proportional merge, weight-adjusted sampling | ~200 |

### Modified files

| File | Change | ~Lines changed |
|------|--------|----------------|
| `stash_ai/recommendations/types.py` | Add TasteCluster, TasteProfile, TagMatch types | ~60 |
| `stash_ai/recommendations/engine.py` | Replace single-profile with cluster-based querying | ~80 |
| `stash_ai/embeddings/storage.py` | Add 3 new tables + CRUD methods | ~150 |
| `stash_ai/embeddings/providers/openclip.py` | Expose embed_text() as public method | ~10 |
| `stash-copilot.py` | Register Build Taste Map task | ~5 |
| `stash-copilot.js` | New Taste Map tab — ECharts scatter plot, cluster sidebar, tag panel | ~600 |
| `stash-copilot.css` | Taste Map layout, cluster cards, tag match bars, neon theme | ~200 |

### Not modified

- `profile.py` — kept for backward compat, no longer called by engine
- Sidebar recs code — left as-is
- `frame_analysis.py` — UMAP/K-Means reused but not modified

**Total estimate:** ~800 new lines, ~1100 modified lines. ~1900 lines total.
