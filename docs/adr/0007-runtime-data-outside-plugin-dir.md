# 7. Runtime data lives outside the plugin directory

## Status

Accepted

## Context

The plugin writes all generated data inside its own directory: the embeddings
database at `assets/stash_copilot.sqlite`, extracted frames under
`assets/embedded_frames/`, plus per-model vector files, caches, vision history,
and exports — every path computed from `__file__`. On a real library this is
tens of gigabytes (the production library is a ~21 GB embeddings DB and 13k+
frame directories).

Under the installed-plugin model ([[0006-distribute-as-installed-plugin]]), Stash
owns the plugin directory and an update replaces it. Data stored there would be
destroyed on every update. ADR-0003 said embeddings are stored "alongside the
plugin"; that location is what this decision revises (the *local-only* principle
of 0003 is unchanged).

## Decision

We will store all generated runtime data outside the plugin directory, under a
stable per-user root: `$STASH_CONFIG_DIR/stash-copilot/` (falling back to
`~/.stash/stash-copilot/` when the variable is unset). A single helper resolves
this root and every module that reads or writes generated data goes through it.
The existing production data is copied into this root once during migration.

## Consequences

Easier: data survives plugin updates and clean reinstalls; the released package
stays free of runtime data; the data root is co-located with Stash's own config
and DB, so it moves with a Stash relocation.

Harder: ~15 modules that hardcode `plugin_dir/assets/...` must be routed through
the helper; a one-time multi-gigabyte copy is required at migration; the data
root is now an external dependency the plugin must create and validate at
startup, and back up separately. Anchoring to `STASH_CONFIG_DIR` ties us to Stash
setting that variable (true today for plugin subprocesses).

## Related

- [[0006-distribute-as-installed-plugin]]
- [[0003-embeddings-stored-locally]] — this revises *where* "local" is
- [[CONTEXT.md]] — Embedding, Frame, O-moment
