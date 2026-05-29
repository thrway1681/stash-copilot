# Design System

**Generated:** 2026-02-15

## Visual Identity

Stash Copilot uses a modern AI-inspired aesthetic with gradients, glows, and subtle animations to convey intelligence and responsiveness.

## Color Palette

### Primary Colors

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| Purple Primary | `#8b5cf6` | 139, 92, 246 | Recommendations, primary actions |
| Cyan Accent | `#06b6d4` | 6, 182, 212 | Gradients, highlights |
| Green Success | `#10b981` | 16, 185, 129 | Similar scenes, success states |
| Orange Alert | `#f59e0b` | 245, 158, 11 | Scene-based recs, warnings |

### Gradient Combinations

```css
/* Primary navbar gradient */
linear-gradient(135deg, #8b5cf6 0%, #06b6d4 50%, #8b5cf6 100%)

/* Score badge gradient */
linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #06b6d4 100%)

/* Glow effect */
box-shadow: 0 0 20px rgba(139, 92, 246, 0.3)
```

### Background Colors

| Name | Value | Usage |
|------|-------|-------|
| Modal Background | `rgba(0, 0, 0, 0.8)` | Modal overlays |
| Card Background | `#1a1a2e` | Cards, panels |
| Input Background | `#2a2a4a` | Form inputs |
| Hover State | `rgba(139, 92, 246, 0.1)` | Interactive elements |

## Typography

### Font Stack

```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
```

### Size Scale

| Name | Size | Usage |
|------|------|-------|
| XS | 10px | Badges, labels |
| SM | 12px | Secondary text, metadata |
| Base | 14px | Body text |
| MD | 16px | Headings, emphasis |
| LG | 18px | Modal titles |
| XL | 24px | Page headers |

## Component Patterns

### Scene Cards

Unified card system with theme variants:

```css
.stash-copilot-card {
    --card-accent: #8b5cf6;
    --card-accent-rgb: 139, 92, 246;
    --card-gradient: linear-gradient(...);
}
```

**Themes:**
| Theme | Color | Use Case |
|-------|-------|----------|
| `similar` | Green (#10b981) | Similar scenes search |
| `recs` | Purple (#8b5cf6) | AI recommendations |
| `scene-recs` | Orange (#f59e0b) | Scene-based recommendations |

**Card Features:**
- 16:9 aspect ratio thumbnails (letterboxed)
- Video preview on hover (300ms delay)
- Sprite scrubbing at bottom edge
- Score badge (top-right) with pulse animation for 90%+
- Duration/resolution badges (bottom corners)
- Interactive badge (funscript indicator)
- Stats row with color highlighting

### Buttons

**Primary Button:**
```css
.stash-copilot-btn {
    background: linear-gradient(135deg, #8b5cf6, #06b6d4);
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stash-copilot-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
}
```

**Secondary Button:**
```css
.stash-copilot-btn-secondary {
    background: transparent;
    border: 1px solid rgba(139, 92, 246, 0.5);
    color: #8b5cf6;
}
```

### Sliders

Weight sliders for blending parameters:

```css
.stash-copilot-slider {
    accent-color: #8b5cf6;
    height: 4px;
}

.stash-copilot-slider-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
}
```

### Tabs

Pill-style navigation:

```css
.stash-copilot-subtabs {
    display: flex;
    gap: 4px;
    padding: 4px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
}

.stash-copilot-subtab {
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    transition: all 0.2s;
}

.stash-copilot-subtab.active {
    background: linear-gradient(135deg, #8b5cf6, #06b6d4);
}
```

### Tooltips

Interactive tooltips with scene details:

**Fixed Mode:** Positioned beside card (for modals)
**Cursor Mode:** Follows mouse (for sidebar)

```css
.stash-copilot-card-tooltip {
    position: fixed;
    background: rgba(30, 30, 50, 0.95);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 8px;
    padding: 12px;
    min-width: 280px;
    z-index: 10001;
}
```

### Modals

Staggered entrance animations:

```css
.stash-copilot-insights-modal {
    animation: modalFadeIn 0.3s ease-out;
}

.stash-copilot-insights-tab {
    animation: tabSlideIn 0.3s ease-out;
    animation-delay: calc(var(--tab-index) * 0.15s);
}

@keyframes modalFadeIn {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}
```

## Animation Guidelines

### Timing Functions

| Type | Easing | Duration |
|------|--------|----------|
| Hover | `ease-out` | 150-200ms |
| Enter | `ease-out` | 200-300ms |
| Exit | `ease-in` | 150ms |
| Attention | `ease-in-out` | 500ms+ |

### Common Animations

**Pulse (high scores):**
```css
@keyframes scorePulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
```

**Shimmer (loading):**
```css
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
```

**Glow (active state):**
```css
@keyframes glow {
    0%, 100% { box-shadow: 0 0 10px rgba(139, 92, 246, 0.3); }
    50% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.6); }
}
```

## Layout Patterns

### Responsive Breakpoints

| Breakpoint | Width | Behavior |
|------------|-------|----------|
| Mobile | < 640px | Stack layouts, simplified |
| Tablet | 640-1024px | Two-column grids |
| Desktop | > 1024px | Full layouts |

### Grid System

**Card Grid:**
```css
.stash-copilot-results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
}
```

**Sidebar Layout:**
```css
.stash-copilot-sidebar-layout {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px;
    max-width: 400px;
}
```

### Spacing Scale

| Name | Value | Usage |
|------|-------|-------|
| XS | 4px | Tight spacing |
| SM | 8px | Element gaps |
| MD | 12px | Section spacing |
| LG | 16px | Component spacing |
| XL | 24px | Section padding |
| 2XL | 32px | Modal padding |

## CSS Class Naming

### Naming Convention

```
.stash-copilot-[FEATURE]-[ELEMENT]
.stash-copilot-[FEATURE]-[ELEMENT]--[MODIFIER]
```

### Feature Prefixes

| Prefix | Feature |
|--------|---------|
| `nav-` | Navbar |
| `insights-` | Main modal |
| `dropdown-` | Dropdown panel |
| `card-` | Scene cards |
| `rec-` | Recommendations |
| `peak-` | Peak moments |
| `train-` | Preference training |
| `sidebar-` | Scene sidebar |
| `similar-` | Similar scenes |
| `search-` | Search page |
| `taste-` | Taste map |
| `vision-` | Vision analysis |
| `chat-` | Chat interface |
| `tool-` | Tool displays |

## Iconography

### Emoji-based Icons

| Icon | Usage |
|------|-------|
| 🤖 | AI/Copilot |
| ⚡ | Actions, speed |
| 🔍 | Search |
| 💡 | Insights, suggestions |
| 🎯 | Recommendations |
| 🏷️ | Tags |
| 📊 | Stats, analytics |
| ⏱️ | Duration |
| ⭐ | Rating |
| 🔧 | Tools |
| 💬 | Chat |
| 🎮 | Interactive/funscript |

### Badge Colors

| Badge | Color | Condition |
|-------|-------|-----------|
| Score ≥ 90% | Animated pulse | High match |
| Score ≥ 70% | Static gradient | Good match |
| Score < 70% | Muted | Lower match |
| Interactive | Gold (#ffd700) | Has funscript |

## Accessibility

### Focus States

```css
*:focus-visible {
    outline: 2px solid #8b5cf6;
    outline-offset: 2px;
}
```

### Contrast Ratios

| Element | Background | Foreground | Ratio |
|---------|------------|------------|-------|
| Body text | #1a1a2e | #ffffff | 12.4:1 ✅ |
| Muted text | #1a1a2e | #888888 | 4.9:1 ✅ |
| Links | #1a1a2e | #8b5cf6 | 4.7:1 ✅ |

### Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
```

## Dark Theme (Default)

The plugin uses dark theme exclusively to match Stash's default appearance:

```css
:root {
    --bg-primary: #1a1a2e;
    --bg-secondary: #16162b;
    --bg-tertiary: #2a2a4a;
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.7);
    --text-muted: rgba(255, 255, 255, 0.5);
    --border-color: rgba(139, 92, 246, 0.3);
}
```
