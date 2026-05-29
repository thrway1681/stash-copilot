# Taste Map 3D: Click-to-Center Camera

## Overview

Add click-to-center orbit camera behavior to the taste map 3D visualization. Clicking a point makes it the new orbit center and zooms in. Double-clicking empty space resets to the default overview.

## Approach

Use Plotly's `camera.center` and `camera.eye` properties via `Plotly.relayout()`. No coordinate transformation of data points needed — Plotly's orbit controls natively support changing the pivot point.

## Click-to-Center

- Listen for `plotly_click` events on the chart
- Extract clicked point's `{x, y, z}` coordinates
- Animate `camera.center` to the clicked point
- Animate `camera.eye` closer to the point (45% of current distance), preserving viewing angle
- Animation: 500ms with cubic ease-out (`1 - (1-t)^3`)

### Animation Function

Custom `requestAnimationFrame` loop that lerps both `center` and `eye`:

1. Read current camera state from Plotly's internal scene object
2. Compute end state: new center = clicked point, new eye = closer along same viewing vector
3. Each frame: interpolate with ease-out, call `Plotly.relayout()`
4. Stop when t >= 1

### Zoom Calculation

```
dx = currentEye.x - target.x
dy = currentEye.y - target.y
dz = currentEye.z - target.z
newEye = target + (dx, dy, dz) * 0.45
```

This preserves the viewing direction while bringing the camera to 45% of the original distance from the clicked point.

## Double-Click Reset

- Listen for `plotly_doubleclick` on the chart
- Suppress Plotly's default axis-reset behavior
- Animate back to default camera: `eye: {1.5, 1.5, 1.2}`, `center: {0, 0, 0}`, `up: {0, 0, 1}`
- Animation: 600ms (longer since it covers more distance)

## Files Modified

- `stash-copilot.js`: Add click handler, animation function, and double-click reset in `renderTasteMapChart()`

## No Backend Changes

This is purely a frontend camera control feature. No Python changes needed.
