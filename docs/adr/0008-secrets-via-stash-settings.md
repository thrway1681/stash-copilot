# 8. API keys come from Stash plugin settings, not .env files

## Status

Accepted

## Context

Cloud LLM/embedding providers and the EroScripts integration need credentials.
Today they are read from files in the plugin directory: `.env`
(`GEMINI_API_KEY`, `OPENROUTER_API_KEY`) loaded via `__file__`, plus
`.eroscripts_auth.json`. Like all in-tree state, these are wiped when an
installed plugin updates ([[0006-distribute-as-installed-plugin]]), and shipping
or copying a tree risks leaking them.

Stash already exposes typed plugin settings (the manifest defines
`text_llm_api_key`, `vision_llm_api_key`, …) stored in Stash's own database,
which is update-safe and the normal way users configure a plugin.

## Decision

We will source all credentials from Stash plugin settings (and, where a setting
does not yet exist, add one) rather than from `.env`/JSON files in the plugin
tree. The `.env` loader is retained only as a dev convenience and is never relied
on in a packaged install.

## Consequences

Easier: secrets are update-safe and never travel with the source tree or the
release zip; configuration is all in one place (the Stash settings UI); there is
no secret-file handling in the migration.

Harder: credentials must be re-entered once in the Stash UI after install; any
code path that read a key from `.env` must be rerouted to read the setting;
secret rotation is now a UI action rather than a file edit.

## Related

- [[0006-distribute-as-installed-plugin]]
- [[0007-runtime-data-outside-plugin-dir]]
