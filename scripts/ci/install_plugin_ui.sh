#!/usr/bin/env bash
# Install the plugin's UI into the dev Stash's plugins dir so a browser
# (Playwright) loads it. Used by the `ui` CI job and for local UI development.
#
# Copies ONLY the manifest + UI surface (the ML/LLM backend can't run in the
# Alpine Stash container) plus a stub exec, into
#   docker/stash-config/plugins/stash-copilot/
# which docker-compose.dev.yml bind-mounts to the container's
#   /root/.stash/plugins/stash-copilot/
# Run this BEFORE `docker compose up` (so the plugin is present at startup), or
# after, followed by the `reloadPlugins` GraphQL mutation.
#
# Prereq: `npm ci` (or npm install) has run so assets/*.min.js are vendored.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/docker/stash-config/plugins/stash-copilot"

mkdir -p "$DEST/assets"

# Manifest + UI files (the contract is name/result-key in the yml + the JS).
cp "$ROOT/stash-copilot.yml" "$DEST/"
cp "$ROOT/stash-copilot.js" "$DEST/"
cp "$ROOT/stash-copilot.css" "$DEST/"

# Vendored UI libraries the JS loads at runtime (npm postinstall produces these).
missing_vendor=0
for lib in marked.min.js purify.min.js plotly-gl3d.min.js; do
  if [ -f "$ROOT/assets/$lib" ]; then
    cp "$ROOT/assets/$lib" "$DEST/assets/"
  else
    echo "[install_plugin_ui] WARNING: vendored lib missing: assets/$lib (run npm ci)" >&2
    missing_vendor=1
  fi
done

# Stub backend exec (see scripts/ci/stub-run-plugin.sh). The manifest's exec is
# [./run-plugin.sh, ./stash-copilot.py]; provide both so the reference resolves.
cp "$ROOT/scripts/ci/stub-run-plugin.sh" "$DEST/run-plugin.sh"
chmod +x "$DEST/run-plugin.sh"
printf '# CI UI-test stub — the real backend runs on the host, not in-container.\n' > "$DEST/stash-copilot.py"

echo "[install_plugin_ui] installed plugin UI into $DEST"
ls -1 "$DEST"
exit "$missing_vendor"
