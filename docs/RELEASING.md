# CI/CD

## Continuous integration (`.github/workflows/ci.yml`)

Runs on pushes to `main`/`dev` and PRs into either.

| Job | Gate | What it does |
|---|---|---|
| **Test** | **blocking** | `uv sync --extra dev` + `pytest` (deselects the one known-failing `test_caption_dashboard` test so it can't mask regressions). |
| Lint & Format | advisory | `ruff check` + `ruff format --check`, scoped to `stash-copilot.py stash_ai/`. |
| Type Check | advisory | strict `mypy .`. |
| **Integration** | **blocking** (needs Test) | Boots a real Dockerized Stash, runs the `docs/TESTING.md` §6 flow (setup → scan → OpenCLIP embed → recommendations) and asserts real recs come back. |

Lint and type checks are advisory because the package carries a known baseline
(~97 ruff findings, ~124 mypy errors). The intent is "no **new** errors." To make
them blocking, first run a cleanup pass:

```bash
uv run --extra dev ruff check --fix stash-copilot.py stash_ai/
uv run --extra dev ruff format stash-copilot.py stash_ai/
```

then drop the `continue-on-error: true` lines.

## Releases (`.github/workflows/release.yml`)

Cut a release by pushing a version tag:

```bash
git tag v1.2.3
git push origin v1.2.3
```

That triggers three jobs:

1. **package** — `scripts/package_plugin.sh` runs `npm ci` (vendors marked/dompurify/plotly
   into `assets/`), stages the plugin (source + manifest + Python dep manifests +
   vendored frontend assets, **excluding** runtime data/tests/docs), stamps the
   manifest version, and produces `dist/stash-copilot.zip` + `dist/index.yml`.
2. **github-release** — attaches the zip to a GitHub Release for direct download.
3. **plugin-index** — publishes `index.yml` + the zip to the `gh-pages` branch so
   Stash users can install/update in-app.

You can also build a bundle locally without tagging:

```bash
VERSION=v1.2.3 bash scripts/package_plugin.sh   # -> dist/
```

### One-time setup for the in-app plugin source

Enable GitHub Pages once: **Settings → Pages → Source: `gh-pages` branch**. After the
first release, users add the Pages URL (e.g. `https://<owner>.github.io/<repo>/index.yml`)
under **Stash → Settings → Plugins → Add Source**, then install **Stash Copilot**
and get update notifications on future releases.

> The index is a single-channel "latest" source: each release overwrites
> `stash-copilot.zip` and `index.yml`. To offer multiple selectable versions,
> version the zip filename in `package_plugin.sh` and append (rather than replace)
> the index entry.
