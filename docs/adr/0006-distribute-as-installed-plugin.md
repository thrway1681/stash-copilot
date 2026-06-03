# 6. Distribute as an installed plugin, not a git checkout

## Status

Accepted

## Context

The plugin grew up as a git working tree living directly inside Stash's plugin
directory (`<stash>/plugins/stash-copilot`), edited in place and run from there.
That blurs development and production: the running copy carries dev tooling,
uncommitted edits, secrets, and ~50 GB of generated data all in one tree, and
there is no clean notion of a released version.

Stash also supports a first-class distribution path: a plugin-source index
(`index.yml` + a versioned zip) that users add in-app and update from. The
repository already carries the machinery for it — `scripts/package_plugin.sh`,
`release.yml`, and a `gh-pages` channel.

## Decision

We will treat the GitHub repository as a pure source/development checkout and
ship the plugin as a packaged, installable artifact. Production is a clean
install of that artifact via the Stash plugin source (or a release zip), never a
git working tree run in place. Releases are cut by tagging; the package excludes
dev tooling, tests, docs, and runtime data.

## Consequences

Easier: dev and prod are cleanly separated; releases are versioned and
reproducible; end users get one-click install and update notifications; the
shipped surface is small and auditable.

Harder: anything the running plugin needs that is *not* in the package must live
elsewhere — most notably generated data and secrets, which can no longer sit in
the plugin tree because an update replaces it (see [[0007-runtime-data-outside-plugin-dir]]
and [[0008-secrets-via-stash-settings]]). Quick "edit the live copy" hotfixes are
no longer the workflow; a fix means a new release.

## Related

- [[0007-runtime-data-outside-plugin-dir]]
- [[0008-secrets-via-stash-settings]]
