# ADR-0004: The Engagement Score formula

**Status:** accepted

The **Engagement Score** for a Scene is `O-count·20 + replays·2 + stars·1.5`, where
`replays = max(view_count − 1, 0)` and `stars = rating100 / 20` (added only if the
scene is rated; unrated scenes get no bonus and no penalty). A time-decayed variant
multiplies this by an exponential recency factor (30-day half-life). The score is
computed in exactly **one** module (`EngagementCalculator`); no other code may
reimplement it.

## Why these choices

- **Play time is excluded.** Raw watch-hours bias the score toward long videos;
  O-count and replays are cleaner intent signals. Several call sites had already
  dropped play time by hand ("to avoid duration bias") — this makes that the rule.
- **Single source of truth.** The formula previously lived in ≥4 places with drifting
  weights, an inconsistent rating scale (one site treated `rating` as 0–5 stars,
  another as `rating100/20` — a ~20× discrepancy), and a docstring (`o·3`) that
  disagreed with its code (`o·20`). One owning module removes the drift.

## Consequences

- **Removed metrics.** The `completion` (watch-fraction) and `intensity` ranking modes
  in `RankScenesByEngagementTool` are dropped: `intensity` (`o_rate × view_count`)
  reduces algebraically to O-count, and `completion` reintroduces the play-time bias
  this ADR excludes.
- **Weights are part of the taste spec.** Changing them later is a deliberate,
  one-place edit — and a ranking-behaviour change across Discover, Rewatch, the Taste
  Map, and the agent tools.
