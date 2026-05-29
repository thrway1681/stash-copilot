# Feature Dependency Table

**Date:** 2026-02-13
**Source:** Code exploration and CLAUDE.md

## Feature Categories

### 1. LLM Integration

| Feature | Dependencies | Entry Point | UI Location |
|---------|--------------|-------------|-------------|
| **Text LLM** | ollama/openai/anthropic providers | `LLMConfig` | Plugin Settings |
| **Vision LLM** | Text LLM + vision-capable model | `SceneVisionTask` | Scene page |
| **Chat** | Text LLM + Tools | `run_chat()` | Modal → Chat tab |
| **Ask** | Text LLM + Tools | `run_ask()` | Task |
| **Stats Summary** | Text LLM + LibraryStatsAggregator | `run_stats_summary()` | Modal → Summary |

### 2. Embedding System

| Feature | Dependencies | Entry Point | UI Location |
|---------|--------------|-------------|-------------|
| **Image Embeddings** | OpenCLIP/SigLIP/CLIP providers | `EmbeddingConfig` | Plugin Settings |
| **Scene Embeddings** | Image Embeddings | `run_embed_scenes()` | Task |
| **Frame Embeddings** | Scene Embeddings + FFmpeg | `run_embed_cached_frames()` | Task |
| **Performer Embeddings** | Scene Embeddings | `run_embed_performers()` | Task |
| **O-Moment Embeddings** | Frame Extraction + O markers | `run_embed_o_moments()` | Task |
| **Tag Embeddings** | Text embedding model | `tag_vocabulary.py` | Internal |

### 3. Scene Discovery

| Feature | Dependencies | Entry Point | UI Location |
|---------|--------------|-------------|-------------|
| **Similar Scenes** | Scene Embeddings | `run_find_similar()` | Sidebar → Similar |
| **Text Search** | Scene Embeddings | `run_search_by_text()` | Search page |
| **Frame Search** | Frame Embeddings + FAISS | `run_build_frame_index()` | Search page |

### 4. Recommendations

| Feature | Dependencies | Entry Point | UI Location |
|---------|--------------|-------------|-------------|
| **Discover New** | Scene Embeddings + Engagement | `run_recommendations()` | Sidebar → Recs |
| **Re-watch** | Scene Embeddings + Engagement | `run_recommendations()` | Sidebar → Recs |
| **Peak Moments** | O-Moment Embeddings | `run_recommendations()` | Modal → Peak |
| **Performer-Based** | Performer Embeddings | `run_recommendations()` | Modal → Recs |

### 5. Preference Learning

| Feature | Dependencies | Entry Point | UI Location |
|---------|--------------|-------------|-------------|
| **Training Session** | Scene Embeddings | `run_preference_trainer()` | Modal → Train |
| **Swipe UI** | Training Session | JS: `preferenceState` | Modal → Train |
| **Preference Recs** | Trained Model | `run_preference_recs()` | Modal → Train |

### 6. Taste Profiling

| Feature | Dependencies | Entry Point | UI Location |
|---------|--------------|-------------|-------------|
| **Taste Map** | Scene Embeddings + Clustering | `run_build_taste_map()` | Modal → Taste Map |
| **3D Visualization** | Taste Map + Plotly.js | JS: `renderTasteMapChart()` | Modal → Taste Map |

### 7. Tag Management

| Feature | Dependencies | Entry Point | UI Location |
|---------|--------------|-------------|-------------|
| **Tag Suggestions** | Vision LLM + Tags | `SceneVisionTask` | Sidebar → Analyze |
| **Tag Gap Detection** | Frame Embeddings + Tag Embeddings | `run_detect_tag_gaps()` | Sidebar → Gaps |

### 8. Vision Analysis

| Feature | Dependencies | Entry Point | UI Location |
|---------|--------------|-------------|-------------|
| **Scene Vision** | Vision LLM + Frame Extraction | `run_scene_vision()` | Sidebar → Analyze |
| **Frame Extraction** | FFmpeg | `FrameExtractor` | Internal |
| **Smart Frame Selection** | Frame Embeddings | `SmartFrameSelector` | Internal |

---

## Dependency Graph

```
Plugin Settings
├── Text LLM Config
│   ├── Chat → Tools (47 database query tools)
│   ├── Ask → Tools
│   └── Stats Summary → LibraryStatsAggregator
│
├── Vision LLM Config (extends Text LLM)
│   ├── Scene Vision → Frame Extraction
│   │   ├── Tag Suggestions
│   │   └── Scene Description
│   └── Performer Description
│
└── Embedding Config
    ├── Scene Embeddings ←─────────────────────────────┐
    │   ├── Similar Scenes                             │
    │   ├── Text Search                                │
    │   ├── Discover/Re-watch Recs ← Engagement Data   │
    │   ├── Taste Map ← Clustering                     │
    │   └── Preference Learning ← Swipe UI             │
    │                                                  │
    ├── Frame Embeddings ← Scene Embeddings            │
    │   ├── Frame Search ← FAISS Index                 │
    │   ├── Tag Gap Detection ← Tag Embeddings         │
    │   └── Smart Frame Selection                      │
    │                                                  │
    ├── Performer Embeddings ← Scene Embeddings ───────┤
    │   ├── Similar Performers                         │
    │   └── Performer-Based Recs                       │
    │                                                  │
    └── O-Moment Embeddings ← Frame Extraction ────────┤
        └── Peak Moments Recs                          │
                                                       │
    Tag Embeddings ─────────────────────────────────────┘
```

---

## Feature Dependency Chains

### Critical Path: Recommendations

```
1. Plugin Settings → Embedding Config
2. Embedding Config → Scene Embeddings
3. Scene Embeddings → Similar Scenes / Recommendations
4. Engagement Data (Stash DB) → Weighted Recommendations
```

### Critical Path: Vision Analysis

```
1. Plugin Settings → Vision LLM Config
2. Scene → FFmpeg Frame Extraction
3. Frames → Vision LLM Analysis
4. Analysis → Tag Suggestions + Description
```

### Critical Path: Preference Learning

```
1. Scene Embeddings (must exist)
2. Start Training Session
3. Swipe Comparisons → Bradley-Terry Update
4. Trained Model → Preference Recommendations
```

---

## Feature Categories for Interview

### Core Features (Likely Keep)
- Scene Embeddings
- Similar Scenes
- Text Search
- Vision Analysis
- Tag Suggestions
- Discover/Re-watch Recommendations

### Advanced Features (Review)
- Preference Learning (swipe training)
- Taste Map (3D visualization)
- Peak Moments (O-marker based)
- Tag Gap Detection
- Frame Search

### Support Features (Internal)
- Frame Extraction
- Smart Frame Selection
- Engagement Scoring
- Clustering
