# Taste Map 3D Visualization Redesign

**Date:** 2026-02-12
**Scope:** Frontend rewrite of taste map chart, tooltips, and sidebar. No backend changes.

## Overview

Complete rewrite of the taste map 3D visualization. Replace current rendering with polished glowing-orb aesthetics, unified card tooltips with thumbnails, and a simplified sidebar with bidirectional hover interaction.

## Chart — Point Rendering

Every embedded scene appears as a glowing orb in 3D space. No distinction between profile and non-profile scenes — all scenes are equal citizens, differentiated only by cluster color and engagement intensity.

**Point sizing:** Engagement score drives marker size on a logarithmic scale. Scenes with zero engagement get the minimum size (3px). High-engagement scenes scale up to ~12px. Formula: `3 + log(1 + engagement) * 2`.

**Glow effect:** Each point gets a soft radial glow matching its cluster color. Achieved through Plotly's marker properties — a semi-transparent outer ring (`marker.line` with cluster color at ~30% opacity and width scaled by engagement) layered with the solid inner point. Higher engagement = wider, brighter glow halo.

**Opacity:** All points start at 0.85 opacity. Zero-engagement scenes drop to 0.4 — visible but quieter. Creates a natural nebula where favorites burn bright and unexplored scenes form a dim cloud.

**Colors:** 8-color cluster rotation: purple (#8b5cf6), cyan (#06b6d4), green (#10b981), amber (#f59e0b), pink (#ec4899), blue (#3b82f6), rose (#f43f5e), alt-purple (#a855f7). Background: #0a0a0f.

**Camera:** Plotly's default orbital controls only — drag to rotate, scroll to zoom, right-drag to pan. No custom click-to-center behavior.

## Tooltips — Unified Card Tooltip on Hover

Custom HTML tooltip following the cursor, using the existing `.stash-copilot-card-tooltip` component with the `taste-map` theme (purple accent).

**Tooltip content:**
- **Thumbnail** — Scene screenshot at 16:9 aspect ratio, `object-fit: contain` (letterboxed)
- **Title** — Scene title, truncated with ellipsis if needed
- **Cluster label** — Colored pill showing cluster membership, using cluster color
- **Stats row** — Play count, O count, engagement score. Values highlighted in accent color when non-zero

**Positioning:** CSS `position: fixed`, continuously updated via global `mousemove` listener to follow cursor. Offset 12px from cursor. Repositions to avoid viewport edges (flip left if overflowing right, clamp top/bottom).

**Show/hide:** Plotly's `plotly_hover` and `plotly_unhover` events trigger tooltip visibility. Same fade+scale animation as existing card tooltips (opacity 0→1, scale 0.96→1, ~150ms transition).

**Mouse tracking:** Global `mousemove` listener on chart container updates cursor position (WebGL canvas swallows mouse events). Tooltip repositions on each frame while visible.

## Cluster Sidebar — Simplified & Connected

300px fixed-width panel on the right side of the taste map. Clean, informational, no sliders or buttons.

**Cluster cards:** Each cluster shows:
- **Header row** — Small colored dot (cluster color) + auto-generated label + collapse toggle (▼/▶)
- **Stats line** — Scene count and engagement share: "42 scenes · 35%"
- **Representative thumbnails** — 3 small thumbnails (48px) of scenes closest to cluster centroid

Cards are collapsible — clicking header toggles stats and thumbnails. All start expanded.

**Hover interaction (sidebar → chart):** Hovering a cluster card desaturates all other clusters in the chart. The hovered cluster keeps full color and glow, while others go grayscale and drop to ~40% opacity. Leaving the card restores all clusters to normal.

**Hover interaction (chart → sidebar):** Hovering a point in the chart adds a subtle border glow on the corresponding cluster card in the sidebar, in the cluster's color. Bidirectional connection.

**Scroll:** Sidebar scrolls independently from chart if many clusters exist (up to 8).

## Removed Features

- Profile vs non-profile scene distinction (all scenes treated equally)
- Click-to-center camera interaction (broken, removed)
- Weight sliders on cluster cards
- Exclude buttons on cluster cards
- Tag match panel (bottom overlay)
- `highlightClusterInChart` / `resetClusterHighlight` functions
- `tasteMapSelectedCluster` state

## Data Flow

Backend unchanged. The `TasteMapResponse` from `taste_map.py` provides all needed data:
- `scenes[]` — x, y, z coordinates, cluster_id, engagement_score, title, thumbnail, play_count, o_counter
- `clusters[]` — auto_label, scene_ids, engagement_share, representative_scenes

The `is_profile` field is ignored.

## Files Changed

| File | Change |
|------|--------|
| `stash-copilot.js` | Rewrite `renderTasteMap`, `renderTasteMapChart`, `renderClusterSidebar`. Remove `highlightClusterInChart`, `resetClusterHighlight`, `showTagMatches`. Simplify state variables. |
| `stash-copilot.css` | Update taste map styles: chart point aesthetics, sidebar card simplification, remove tag match panel styles, remove weight slider / exclude button styles. |

No backend or Python changes.
