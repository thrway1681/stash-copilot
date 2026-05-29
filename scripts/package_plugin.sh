#!/usr/bin/env bash
# Build a distributable Stash Copilot plugin bundle:
#   dist/stash-copilot.zip   — the plugin (source + manifest + vendored frontend assets)
#   dist/index.yml           — a Stash plugin-source index pointing at the zip
#
# The bundle deliberately EXCLUDES runtime/dev data (embeddings DB, extracted
# frames, recommendation JSON, tests, docs, the dev Docker setup, venv, etc.).
# Frontend libs (marked/dompurify/plotly) live under assets/ and are produced by
# `npm ci` (its postinstall step) — this script runs it if they're missing.
#
# Env:
#   VERSION   Release version string (default: `git describe --tags --always`)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PLUGIN_ID="stash-copilot"
VERSION="${VERSION:-$(git describe --tags --always 2>/dev/null || echo 0.0.0-dev)}"
DIST="$ROOT/dist"
STAGE="$DIST/$PLUGIN_ID"
ZIP="$DIST/$PLUGIN_ID.zip"

# Vendored frontend assets the manifest serves and the JS loads at runtime.
VENDORED_ASSETS=(
  assets/marked.min.js
  assets/purify.min.js
  assets/plotly-gl3d.min.js
)

# Ensure vendored assets exist (npm postinstall copies them into assets/).
need_build=0
for f in "${VENDORED_ASSETS[@]}"; do [ -f "$f" ] || need_build=1; done
if [ "$need_build" = 1 ]; then
  echo "[package] vendored assets missing — running npm ci"
  npm ci
fi

rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE/assets"

# --- top-level plugin files (manifest, entry point, frontend, runner) ---
cp stash-copilot.yml stash-copilot.py stash-copilot.js stash-copilot.css run-plugin.sh "$STAGE/"
# --- Python dependency manifests so `uv sync` works post-install ---
cp pyproject.toml uv.lock requirements.txt README.md "$STAGE/"
# --- Python package + prompt templates (sources only, no caches) ---
cp -R stash_ai "$STAGE/"
cp -R prompts "$STAGE/"
find "$STAGE" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type f -name '*.pyc' -delete
# --- vendored frontend assets only (NOT runtime data under assets/) ---
for f in "${VENDORED_ASSETS[@]}"; do cp "$f" "$STAGE/assets/"; done

# Stamp the manifest with the release version (replace the `version:` line).
python3 - "$STAGE/stash-copilot.yml" "$VERSION" <<'PY'
import re, sys
path, version = sys.argv[1], sys.argv[2]
text = open(path).read()
text = re.sub(r'(?m)^version:.*$', f'version: {version}', text, count=1)
open(path, 'w').write(text)
PY

# Zip with the plugin id as the top-level folder (Stash installs into plugins/<id>/).
( cd "$DIST" && zip -rq "$ZIP" "$PLUGIN_ID" )
rm -rf "$STAGE"

SHA=$(shasum -a 256 "$ZIP" | awk '{print $1}')
SIZE=$(wc -c < "$ZIP" | tr -d ' ')

# Stash plugin-source index (community convention). `path` is resolved relative
# to the URL the index is served from, so the zip must sit beside index.yml.
DATE="${SOURCE_DATE:-$(date -u +"%Y-%m-%d %H:%M:%S")}"
cat > "$DIST/index.yml" <<EOF
- id: $PLUGIN_ID
  name: Stash Copilot
  metadata:
    description: AI-powered assistant for your Stash library - chat, insights, and vision analysis.
  version: $VERSION
  date: "$DATE"
  path: $PLUGIN_ID.zip
  sha256: $SHA
  requires: []
EOF

echo "[package] built $ZIP"
echo "[package]   version: $VERSION"
echo "[package]   sha256:  $SHA"
echo "[package]   size:    $SIZE bytes"
echo "[package]   index:   $DIST/index.yml"
