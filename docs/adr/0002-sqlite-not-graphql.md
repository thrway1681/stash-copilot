# ADR-0002: Read Stash data from its SQLite database, not the GraphQL API

**Status:** accepted

Stash Copilot reads the user's library data directly from Stash's SQLite database
(through the `StashInterface` helper from the `stashapi` library) rather than through
Stash's GraphQL API. GraphQL is reserved for the few needs the database can't serve.

## Why

- The plugin's core work is **bulk aggregation** over the whole library — engagement
  scoring, profile building, taste maps — which needs granular, per-event data
  (`scenes_view_dates`, `scenes_o_dates`) and would otherwise require thousands of
  GraphQL round-trips.
- Direct reads are far faster and avoid N+1 query patterns against the API.

## Consequences

- The plugin is **coupled to Stash's internal schema**; a Stash schema change can
  break queries. Mitigated by going through `stashapi`'s `StashInterface` rather than
  hand-rolled raw SQL where possible.
- Reads bypass GraphQL's auth/validation layer, so the DB is treated as **read-mostly**;
  the plugin must never corrupt Stash's own data.
- **Writes** back to Stash (e.g. applying tags) must still go through the official API,
  never direct SQL.
