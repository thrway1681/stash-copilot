# ADR-0005: Taste Map produces clusters; recommendations consume them

**Status:** accepted

Taste Clusters are computed in exactly one place — the **Build Taste Map** task
(`TasteMapTask` → `build_taste_profile`) — which persists them to the local store
(`taste_clusters` + `scene_umap_coords`). Cluster-based **Discover** does not cluster;
it *reads* the stored clusters via `ClusterRecommendationEngine`. This is a deliberate
producer/consumer split, not duplication.

## Why record this

A whole-repo architecture sweep flagged "Discover and the Taste Map cluster engaged
scenes twice." Reading the code shows otherwise: only the Taste Map task clusters, and
it already scores via the shared `EngagementCalculator` (see ADR-0004) — Discover only
consumes the persisted result. Recording the design here stops a future review (human
or agent) from re-proposing a merge that doesn't apply.

## Consequences

- Discover's cluster mode depends on the Taste Map having been built (it warns "run
  Build Taste Map first" when no clusters exist) — an accepted dependency.
- The remaining friction at this seam is **storage-shaped**, not clustering-shaped: the
  consumer reads clusters as untyped dicts, and the Taste Map task loads embeddings
  one-by-one. Those are tracked under the storage-interface refactor (architecture
  review candidate #1), not here.
