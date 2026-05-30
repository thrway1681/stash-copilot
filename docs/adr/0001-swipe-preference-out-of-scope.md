# ADR-0001: Explicit swipe-based preference learning is out of scope

**Status:** accepted

Stash Copilot infers user taste **implicitly** from engagement (views, O-count, play
time, rating). It will *not* ask users to train their taste through explicit A/B
"swipe" comparisons. The explicit-preference subsystem — the `preferences/` package
(a Bayesian Bradley-Terry model), its seven plugin tasks, the swipe-trainer UI, the
`preference_*` database tables, and the optional `_apply_preference_model` blend in
the recommendations engine — is therefore slated for removal, tracked as a separate
refactor.

## Considered options

1. **Keep building it** — continue investing in the swipe trainer and Bayesian model.
2. **Freeze it in place** — stop developing it but leave the code in the tree.
3. **Remove it** *(chosen)* — the maintainer considers explicit swiping out of scope
   and prefers a purely engagement-driven taste model. Carrying the subsystem costs
   ~5 modules, a large JS UI surface, three DB tables, and a coupling into the core
   engine for a capability that won't be used.

## Consequences

- The recommendations engine's preference blend is removed. It already degrades
  gracefully (returns the engagement profile unchanged when no trained model exists),
  so **Discover and Rewatch are unaffected**.
- "Preference" is retired from the taste vocabulary. The implicit engagement centroid
  formerly called `UserPreferenceProfile` is renamed **Engagement Profile**
  (see `CONTEXT.md`).
- The unrelated **`performer_preference`** recommendation mode keeps its name — it is
  not part of this subsystem.

## Removal touchpoints (for the tracking refactor)

`stash_ai/preferences/` · `stash_ai/tasks/preference_recs.py` ·
`stash-copilot.py` (7 task handlers + dispatch) · `stash-copilot.js` (swipe-trainer UI) ·
`stash_ai/embeddings/storage.py` (`preference_comparisons`, `preference_model_state`,
`preference_sessions`) · `stash_ai/recommendations/engine.py` (`_apply_preference_model`
+ call site) · `tests/test_preferences.py` · `stash-copilot.yml` (7 task buttons).
