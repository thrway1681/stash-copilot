# Taste Map 3D Visualization Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the taste map 3D visualization with glowing-orb aesthetics, unified card tooltips, and a simplified sidebar with bidirectional hover interaction.

**Architecture:** Frontend-only rewrite. Replace `renderTasteMapChart`, `renderClusterSidebar`, and related functions in `stash-copilot.js`. Update CSS in `stash-copilot.css`. Remove dead code (tag match panel, weight sliders, exclude buttons, click-to-center). No backend changes.

**Tech Stack:** Plotly.js GL3D (already loaded), vanilla JS, CSS custom properties

**Design doc:** `docs/plans/2026-02-12-taste-map-3d-redesign.md`

---

### Task 1: Clean Up State and Remove Dead Code

**Files:**
- Modify: `stash-copilot.js:56-60` (state object)
- Modify: `stash-copilot.js:5237-5392` (remove `setupClusterCardEvents`, `highlightClusterInChart`, `resetClusterHighlight`, `showTagMatches`, `testCustomPhrase`)
- Modify: `stash-copilot.js:5336-5377` (simplify `setupTasteMapEvents` — remove tag panel close and phrase input handlers)

**Step 1: Remove `tasteMapSelectedCluster` from state**

In the state object at line 56-60, remove the `tasteMapSelectedCluster` property:

```javascript
// Before (lines 56-60):
tasteMapData: null,
tasteMapRequestId: null,
tasteMapLoading: false,
tasteMapSelectedCluster: null,
tasteMapChart: null,

// After:
tasteMapData: null,
tasteMapRequestId: null,
tasteMapLoading: false,
tasteMapChart: null,
```

**Step 2: Delete dead functions**

Delete these entire functions from `stash-copilot.js`:
- `setupClusterCardEvents` (lines 5237-5277) — replaced in Task 4
- `highlightClusterInChart` (lines 5279-5299)
- `resetClusterHighlight` (lines 5301-5310)
- `showTagMatches` (lines 5312-5334)
- `testCustomPhrase` (lines 5379-5391)

**Step 3: Simplify `setupTasteMapEvents`**

Rewrite `setupTasteMapEvents` (lines 5336-5377) to only keep the build button and sidebar collapse toggle. Remove tag panel close and phrase input handlers:

```javascript
function setupTasteMapEvents(modal) {
    const buildBtn = modal.querySelector('.stash-copilot-taste-map-build-btn');
    if (buildBtn) {
        buildBtn.addEventListener('click', () => buildTasteMap(modal));
    }

    const sidebarHeader = modal.querySelector('.stash-copilot-taste-map-sidebar-header');
    if (sidebarHeader) {
        sidebarHeader.addEventListener('click', () => {
            const sidebar = modal.querySelector('.stash-copilot-taste-map-sidebar');
            sidebar.classList.toggle('collapsed');
            const toggle = sidebarHeader.querySelector('.stash-copilot-taste-map-sidebar-toggle');
            if (toggle) toggle.textContent = sidebar.classList.contains('collapsed') ? '▶' : '◀';
            if (state.tasteMapChart) {
                setTimeout(() => Plotly.Plots.resize(state.tasteMapChart), 300);
            }
        });
    }
}
```

**Step 4: Commit**

```bash
git add stash-copilot.js
git commit -m "refactor(taste-map): remove dead code and simplify state"
```

---

### Task 2: Simplify HTML Template

**Files:**
- Modify: `stash-copilot.js:3873-3908` (taste map HTML template)

**Step 1: Remove tag match panel from HTML**

The HTML template is inline in `stash-copilot.js`. Remove the entire tag match panel (`stash-copilot-taste-map-tags` div and its children) from lines 3884-3894. The simplified template should be:

```html
<div class="stash-copilot-taste-map-container">
    <div class="stash-copilot-taste-map-toolbar">
        <button class="btn btn-primary stash-copilot-taste-map-build-btn">Build Taste Map</button>
        <label class="stash-copilot-taste-map-k-label" title="Number of clusters (leave empty for auto-detect)">
            k: <input type="number" class="stash-copilot-taste-map-k-input" min="2" max="20" placeholder="auto" />
        </label>
        <span class="stash-copilot-taste-map-status"></span>
    </div>
    <div class="stash-copilot-taste-map-content" style="display:none">
        <div class="stash-copilot-taste-map-main">
            <div class="stash-copilot-taste-map-chart" id="taste-map-chart"></div>
        </div>
        <div class="stash-copilot-taste-map-sidebar">
            <div class="stash-copilot-taste-map-sidebar-header" title="Toggle sidebar">
                <span>CLUSTERS</span>
                <span class="stash-copilot-taste-map-sidebar-toggle">◀</span>
            </div>
            <div class="stash-copilot-taste-map-clusters"></div>
        </div>
    </div>
    <div class="stash-copilot-taste-map-empty">
        <p>Build a Taste Map to visualize your preference clusters.</p>
        <p class="stash-copilot-taste-map-empty-sub">Your most engaged scenes are grouped by visual similarity, auto-labeled, and plotted in 3D space.</p>
    </div>
</div>
```

Key changes:
- Removed `stash-copilot-taste-map-tags` div and all children (tag match panel, phrase input)
- Kept chart, sidebar, toolbar, and empty state as-is

**Step 2: Commit**

```bash
git add stash-copilot.js
git commit -m "refactor(taste-map): remove tag match panel from HTML template"
```

---

### Task 3: Rewrite `renderTasteMapChart`

**Files:**
- Modify: `stash-copilot.js:4954-5185` (replace entire `renderTasteMapChart` function)

**Step 1: Replace `renderTasteMapChart` with new implementation**

Delete the existing function (lines 4954-5185) and replace with:

```javascript
function renderTasteMapChart(modal, data) {
    const chartContainer = modal.querySelector('#taste-map-chart');
    if (!chartContainer) return;

    // Purge old chart if exists
    if (state.tasteMapChart) {
        if (state.tasteMapChart._tmMouseCleanup) state.tasteMapChart._tmMouseCleanup();
        Plotly.purge(state.tasteMapChart);
    }

    state.tasteMapChart = chartContainer;

    // Build cluster label lookup
    const clusterLabels = {};
    for (const c of data.clusters) {
        clusterLabels[c.cluster_id] = c.auto_label;
    }

    // One trace per cluster — all scenes, no profile/non-profile distinction
    const traces = [];
    for (const cluster of data.clusters) {
        const clusterScenes = data.scenes.filter(s => s.cluster_id === cluster.cluster_id);
        if (clusterScenes.length === 0) continue;

        const color = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];
        const colorRgb = hexToRgb(color);

        traces.push({
            name: cluster.auto_label,
            type: 'scatter3d',
            mode: 'markers',
            x: clusterScenes.map(s => s.x),
            y: clusterScenes.map(s => s.y),
            z: clusterScenes.map(s => s.z || 0),
            customdata: clusterScenes.map(s => ({
                id: s.scene_id,
                title: s.title || `Scene ${s.scene_id}`,
                plays: s.play_count || 0,
                oCnt: s.o_counter || 0,
                eng: s.engagement_score || 0,
                clusterId: cluster.cluster_id,
            })),
            marker: {
                size: clusterScenes.map(s => {
                    const eng = s.engagement_score || 0;
                    return Math.max(3, Math.min(12, 3 + Math.log(1 + eng) * 2));
                }),
                color: color,
                opacity: clusterScenes.map(s => {
                    const eng = s.engagement_score || 0;
                    return eng > 0 ? 0.85 : 0.4;
                }),
                line: {
                    color: clusterScenes.map(s => {
                        const eng = s.engagement_score || 0;
                        const glowAlpha = Math.min(0.3, 0.05 + Math.log(1 + eng) * 0.04);
                        return `rgba(${colorRgb}, ${glowAlpha})`;
                    }),
                    width: clusterScenes.map(s => {
                        const eng = s.engagement_score || 0;
                        return Math.max(0, Math.min(4, Math.log(1 + eng) * 0.8));
                    }),
                },
            },
            hoverinfo: 'none',
            hovertemplate: null,
        });
    }

    const layout = {
        paper_bgcolor: 'transparent',
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
            bgcolor: 'rgba(10, 10, 15, 0.5)',
            xaxis: { visible: false },
            yaxis: { visible: false },
            zaxis: { visible: false },
            camera: {
                eye: { x: 1.5, y: 1.5, z: 1.2 },
                up: { x: 0, y: 0, z: 1 },
            },
            dragmode: 'orbit',
        },
        showlegend: false,
        hovermode: 'closest',
    };

    const config = {
        responsive: true,
        scrollZoom: 'gl3d',
        displayModeBar: true,
        modeBarButtonsToRemove: [
            'toImage', 'orbitRotation', 'tableRotation',
            'zoom3d', 'pan3d', 'resetCameraLastSave3d',
        ],
        displaylogo: false,
    };

    Plotly.newPlot(chartContainer, traces, layout, config);

    // --- Custom tooltip (unified card tooltip, cursor-following) ---
    let tooltip = document.querySelector('.stash-copilot-card-tooltip[data-theme="taste-map"]');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'stash-copilot-card-tooltip';
        tooltip.dataset.theme = 'taste-map';
        document.body.appendChild(tooltip);
    }

    // Track mouse globally (WebGL canvas swallows events)
    let mouseX = 0, mouseY = 0;
    function trackMouse(e) { mouseX = e.clientX; mouseY = e.clientY; }
    document.addEventListener('mousemove', trackMouse);
    chartContainer._tmMouseCleanup = () => {
        document.removeEventListener('mousemove', trackMouse);
    };

    chartContainer.on('plotly_hover', function(eventData) {
        if (!eventData.points || !eventData.points.length) return;
        const pt = eventData.points[0];
        const d = pt.customdata;
        if (!d) return;

        const clusterLabel = clusterLabels[d.clusterId] || '';
        const clusterColor = CLUSTER_COLORS[d.clusterId % CLUSTER_COLORS.length];

        tooltip.innerHTML = `
            <img class="stash-copilot-card-tooltip-thumb" src="/scene/${d.id}/screenshot" alt="">
            <div class="stash-copilot-card-tooltip-header">
                <div class="stash-copilot-card-tooltip-title">${escapeHtml(d.title)}</div>
                ${clusterLabel ? `<div class="stash-copilot-card-tooltip-studio" style="color: ${clusterColor}">${escapeHtml(clusterLabel)}</div>` : ''}
            </div>
            <div class="stash-copilot-card-tooltip-stats">
                <span class="${d.plays > 0 ? 'has-value' : ''}">▶ ${d.plays} plays</span>
                <span class="${d.oCnt > 0 ? 'has-value' : ''}">💦 ${d.oCnt}</span>
                <span class="stash-copilot-card-tooltip-score">Eng: ${d.eng.toFixed(1)}</span>
            </div>
        `;

        const tooltipWidth = 260;
        const margin = 12;
        let left = mouseX + margin;
        let top = mouseY - margin;

        if (left + tooltipWidth > window.innerWidth) {
            left = mouseX - tooltipWidth - margin;
        }
        if (top < margin) top = margin;
        if (top + 120 > window.innerHeight) {
            top = window.innerHeight - 120 - margin;
        }

        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
        tooltip.classList.add('visible');

        // Chart → sidebar: highlight corresponding cluster card
        const cards = modal.querySelectorAll('.stash-copilot-taste-map-cluster-card');
        cards.forEach(card => {
            if (parseInt(card.dataset.clusterId) === d.clusterId) {
                card.classList.add('chart-hover');
            } else {
                card.classList.remove('chart-hover');
            }
        });
    });

    chartContainer.on('plotly_unhover', function() {
        tooltip.classList.remove('visible');

        // Remove chart-hover from all sidebar cards
        const cards = modal.querySelectorAll('.stash-copilot-taste-map-cluster-card');
        cards.forEach(card => card.classList.remove('chart-hover'));
    });

    // Resize observer
    if (chartContainer._resizeObserver) {
        chartContainer._resizeObserver.disconnect();
    }
    const resizeObserver = new ResizeObserver(() => {
        Plotly.Plots.resize(chartContainer);
    });
    resizeObserver.observe(chartContainer);
    chartContainer._resizeObserver = resizeObserver;
}
```

Key changes from old version:
- **No background trace** — all scenes in per-cluster traces
- **No `is_profile` filter** — every scene included
- **Glow via `marker.line`** — engagement-scaled border width/opacity
- **Per-point opacity** — 0.85 for engaged, 0.4 for zero-engagement
- **No `plotly_click` handler** — removed broken click-to-center
- **Chart→sidebar hover** — adds `chart-hover` class to matching cluster card

**Step 2: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(taste-map): rewrite chart with glowing orbs and unified tooltips"
```

---

### Task 4: Rewrite `renderClusterSidebar` with Bidirectional Hover

**Files:**
- Modify: `stash-copilot.js:5187-5235` (replace entire `renderClusterSidebar` function)

**Step 1: Replace `renderClusterSidebar`**

Delete the existing function (lines 5187-5235) and replace with the simplified version. No sliders, no exclude buttons, no tag matches. Adds hover-to-desaturate interaction:

```javascript
function renderClusterSidebar(modal, data) {
    const container = modal.querySelector('.stash-copilot-taste-map-clusters');
    if (!container) return;

    container.innerHTML = data.clusters.map(cluster => {
        const color = CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length];
        const thumbs = (cluster.representative_scenes || []).map(sid => {
            const scene = data.scenes.find(s => s.scene_id === sid);
            const thumb = scene?.thumbnail || `/scene/${sid}/screenshot`;
            return `<img src="${thumb}" class="stash-copilot-taste-map-cluster-thumb" alt="" loading="lazy">`;
        }).join('');

        return `
            <div class="stash-copilot-taste-map-cluster-card"
                 data-cluster-id="${cluster.cluster_id}"
                 style="--cluster-color: ${color}; --cluster-color-rgb: ${hexToRgb(color)}">
                <div class="stash-copilot-taste-map-cluster-header">
                    <span class="stash-copilot-taste-map-cluster-dot" style="background: ${color}"></span>
                    <span class="stash-copilot-taste-map-cluster-label">${escapeHtml(cluster.auto_label)}</span>
                    <span class="stash-copilot-taste-map-cluster-toggle">▼</span>
                </div>
                <div class="stash-copilot-taste-map-cluster-body">
                    <div class="stash-copilot-taste-map-cluster-stats">
                        <span>${cluster.scene_ids.length} scenes &middot; ${(cluster.engagement_share * 100).toFixed(0)}%</span>
                    </div>
                    <div class="stash-copilot-taste-map-cluster-thumbs">${thumbs}</div>
                </div>
            </div>
        `;
    }).join('');

    // Setup sidebar event handlers
    setupClusterCardEvents(modal, data);
}

function setupClusterCardEvents(modal, data) {
    const cards = modal.querySelectorAll('.stash-copilot-taste-map-cluster-card');

    cards.forEach(card => {
        const clusterId = parseInt(card.dataset.clusterId);
        const header = card.querySelector('.stash-copilot-taste-map-cluster-header');

        // Click toggle to collapse/expand
        header.addEventListener('click', () => {
            card.classList.toggle('collapsed');
        });

        // Hover card → desaturate other clusters in chart
        card.addEventListener('mouseenter', () => {
            if (!state.tasteMapChart) return;
            const el = state.tasteMapChart;
            const traceCount = el.data ? el.data.length : 0;
            if (traceCount === 0) return;

            // Build restyle arrays: desaturate all except hovered cluster
            const updates = { opacity: [], 'marker.color': [] };
            for (let i = 0; i < traceCount; i++) {
                const traceClusterId = data.clusters[i]?.cluster_id;
                if (traceClusterId === clusterId) {
                    // Keep this cluster vibrant
                    updates.opacity.push(0.9);
                    updates['marker.color'].push(CLUSTER_COLORS[traceClusterId % CLUSTER_COLORS.length]);
                } else {
                    // Desaturate: grey out + low opacity
                    updates.opacity.push(0.15);
                    updates['marker.color'].push('rgba(150, 150, 150, 0.4)');
                }
            }
            Plotly.restyle(el, {
                opacity: updates.opacity.map(o => [o]),
                'marker.color': updates['marker.color'].map(c => [c]),
            }, Array.from({ length: traceCount }, (_, i) => i));
        });

        // Mouse leave → restore all clusters
        card.addEventListener('mouseleave', () => {
            if (!state.tasteMapChart) return;
            const el = state.tasteMapChart;
            const traceCount = el.data ? el.data.length : 0;
            if (traceCount === 0) return;

            const restoreUpdates = {};
            for (let i = 0; i < traceCount; i++) {
                const traceClusterId = data.clusters[i]?.cluster_id;
                const color = CLUSTER_COLORS[(traceClusterId ?? i) % CLUSTER_COLORS.length];
                if (!restoreUpdates.opacity) restoreUpdates.opacity = [];
                if (!restoreUpdates['marker.color']) restoreUpdates['marker.color'] = [];
                // Restore per-point opacity from original data
                const clusterScenes = data.scenes.filter(s => s.cluster_id === traceClusterId);
                restoreUpdates.opacity.push([clusterScenes.map(s => {
                    const eng = s.engagement_score || 0;
                    return eng > 0 ? 0.85 : 0.4;
                })]);
                restoreUpdates['marker.color'].push([color]);
            }
            Plotly.restyle(el, restoreUpdates, Array.from({ length: traceCount }, (_, i) => i));
        });
    });
}
```

Key changes:
- **Removed** weight slider, exclude button, total plays/O stats from card body
- **Removed** click-to-select/highlight behavior (was `card.classList.add('active')`)
- **Added** `mouseenter`/`mouseleave` on cards for desaturation effect
- **Collapse** is now on header click (full header, not just toggle icon)

**Step 2: Commit**

```bash
git add stash-copilot.js
git commit -m "feat(taste-map): rewrite sidebar with bidirectional hover interaction"
```

---

### Task 5: Clean Up CSS — Remove Dead Styles

**Files:**
- Modify: `stash-copilot.css:9486-9596` (remove tag match panel and phrase input styles)
- Modify: `stash-copilot.css:9697-9714` (remove `.active` and `.excluded` card states)
- Modify: `stash-copilot.css:9793-9843` (remove weight slider and exclude button styles)

**Step 1: Remove tag match panel CSS**

Delete lines 9486-9596 (from `.stash-copilot-taste-map-tags` through `.stash-copilot-taste-map-phrase-result`). This includes:
- `.stash-copilot-taste-map-tags` and its animation
- `.stash-copilot-taste-map-tags-header`, `-title`, `-close`
- `.stash-copilot-taste-map-tag-row`, `-text`, `-bar`, `-fill`, `-score`
- `.stash-copilot-taste-map-tags-custom`
- `.stash-copilot-taste-map-phrase-input`, `-result`

**Step 2: Remove `.active` and `.excluded` card states**

Delete lines 9697-9714:
- `.stash-copilot-taste-map-cluster-card.active` (and `clusterPulse` keyframes)
- `.stash-copilot-taste-map-cluster-card.excluded`
- `.stash-copilot-taste-map-cluster-card.excluded .stash-copilot-taste-map-cluster-label`

**Step 3: Remove weight slider and exclude button CSS**

Delete lines 9793-9843:
- `.stash-copilot-taste-map-cluster-weight`
- `.stash-copilot-taste-map-weight-slider` (and its `::-webkit-slider-thumb`)
- `.stash-copilot-taste-map-weight-value`
- `.stash-copilot-taste-map-exclude-btn` (and its `:hover`)

**Step 4: Commit**

```bash
git add stash-copilot.css
git commit -m "refactor(taste-map): remove dead CSS for tag panel, sliders, exclude buttons"
```

---

### Task 6: Add Chart-Hover Sidebar Highlight CSS

**Files:**
- Modify: `stash-copilot.css` (add new `.chart-hover` state after cluster card styles)

**Step 1: Add `.chart-hover` class for sidebar cards**

After the existing `.stash-copilot-taste-map-cluster-card:hover` rule, add a new rule for when the chart hovers a point and highlights the corresponding sidebar card:

```css
/* Chart → sidebar hover feedback */
.stash-copilot-taste-map-cluster-card.chart-hover {
    border-color: rgba(var(--cluster-color-rgb), 0.6);
    box-shadow: 0 0 12px rgba(var(--cluster-color-rgb), 0.25);
    transition: all 0.15s ease;
}
```

This creates a subtle glow on the sidebar card matching the cluster's color when hovering a point in the chart.

**Step 2: Commit**

```bash
git add stash-copilot.css
git commit -m "feat(taste-map): add chart-hover highlight style for sidebar cards"
```

---

### Task 7: Verify and Test

**Step 1: Check for JS references to removed functions**

Search `stash-copilot.js` for any remaining references to:
- `highlightClusterInChart`
- `resetClusterHighlight`
- `showTagMatches`
- `testCustomPhrase`
- `tasteMapSelectedCluster`
- `is_profile`

All should return zero matches. If any remain, update/remove them.

**Step 2: Check for CSS references to removed classes**

Search both files for:
- `taste-map-tags` (should only appear in removed code)
- `weight-slider`
- `exclude-btn`
- `phrase-input`

All should return zero matches.

**Step 3: Visual verification**

1. Open Stash → Insights modal → Taste Map tab
2. Click "Build Taste Map" (or verify cached data loads)
3. Verify:
   - All scenes render as colored orbs (no faint white background dots)
   - Engagement drives size and opacity (bright large vs dim small)
   - Glow halos visible on high-engagement points
   - Hover shows thumbnail tooltip following cursor
   - Tooltip shows: thumbnail, title, cluster label (colored), stats
   - Sidebar shows simplified cards (no sliders, no exclude buttons)
   - Hovering sidebar card desaturates other clusters (grayscale + low opacity)
   - Leaving sidebar card restores all colors
   - Hovering chart point highlights corresponding sidebar card (border glow)
   - Collapse toggle works on sidebar cards
   - Sidebar collapse/expand toggle works
   - Plotly default controls work (orbit, zoom, pan)

**Step 4: Check logs for errors**

```bash
tail -50 ~/.stash/stash.log | grep -i "error\|warn\|exception"
```

**Step 5: Final commit if any fixes were needed**

```bash
git add stash-copilot.js stash-copilot.css
git commit -m "fix(taste-map): address issues found during testing"
```
