#!/usr/bin/env bash
# Headlessly bring a fresh Dockerized dev Stash to a usable state:
#   1. complete first-run setup (library = $STASH_LIBRARY)
#   2. scan media (covers + sprites + phashes)
#   3. configure the plugin for OpenCLIP visual embedding (no VLM/LLM needed)
#
# Idempotent: safe to re-run. Talks to Stash over HTTP only, so it works the
# same on a dev laptop and on a CI runner. Used by docs/TESTING.md §6 and the
# integration-test CI job.
#
# Env:
#   STASH_URL       Base URL of the dev Stash      (default http://localhost:3000)
#   STASH_LIBRARY   Library path INSIDE the container. With the media path-parity
#                   mount it equals the absolute host media dir. (required)
#   EMBED_MODEL     OpenCLIP model to configure    (default ViT-B-32)
#   EMBED_DEVICE    Embedding device               (default cpu; use mps on macOS)
set -euo pipefail

STASH_URL="${STASH_URL:-http://localhost:3000}"
EMBED_MODEL="${EMBED_MODEL:-ViT-B-32}"
EMBED_DEVICE="${EMBED_DEVICE:-cpu}"
: "${STASH_LIBRARY:?set STASH_LIBRARY to the in-container library path (= host media dir under path parity)}"

GQL="$STASH_URL/graphql"

gql() { # $1 = JSON body -> prints response body
  curl -fsS -X POST "$GQL" -H 'Content-Type: application/json' -d "$1"
}

status() {
  gql '{"query":"{ systemStatus { status } }"}' \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['systemStatus']['status'])"
}

echo "[bootstrap] waiting for Stash at $STASH_URL ..."
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null "$GQL" -X POST -H 'Content-Type: application/json' \
       -d '{"query":"{ systemStatus { status } }"}' 2>/dev/null; then break; fi
  sleep 2
done

# Note: payloads are built into a variable first (not inlined as gql "$(...)").
# Nested double quotes inside a quoted command substitution mis-parse under
# bash 3.2 (the default /bin/bash on macOS), corrupting the JSON.
if [ "$(status)" = "SETUP" ]; then
  echo "[bootstrap] running first-run setup (library=$STASH_LIBRARY)"
  setup_payload=$(STASH_LIBRARY="$STASH_LIBRARY" python3 <<'PY'
import json, os
lib = os.environ["STASH_LIBRARY"]
print(json.dumps({
  "query": "mutation($i:SetupInput!){ setup(input:$i) }",
  "variables": {"i": {
    "configLocation": "/root/.stash/config.yml",
    "stashes": [{"path": lib, "excludeVideo": False, "excludeImage": False}],
    "databaseFile": "/root/.stash/stash-go.sqlite",
    "generatedLocation": "/generated",
    "cacheLocation": "/cache",
    "storeBlobsInDatabase": True,
    "blobsLocation": "/root/.stash/blobs",
  }},
}))
PY
)
  gql "$setup_payload" >/dev/null
  for _ in $(seq 1 30); do [ "$(status)" = "OK" ] && break; sleep 1; done
else
  echo "[bootstrap] Stash already set up (status=$(status)); skipping setup"
fi

echo "[bootstrap] scanning media"
gql '{"query":"mutation($i:ScanMetadataInput!){ metadataScan(input:$i) }","variables":{"i":{"scanGenerateCovers":true,"scanGenerateSprites":true,"scanGeneratePhashes":true}}}' >/dev/null
for _ in $(seq 1 120); do
  n=$(gql '{"query":"{ jobQueue { id } }"}' | python3 -c "import sys,json;print(len(json.load(sys.stdin)['data']['jobQueue'] or []))")
  [ "$n" = "0" ] && break
  sleep 2
done

echo "[bootstrap] configuring plugin for OpenCLIP ($EMBED_MODEL on $EMBED_DEVICE)"
config_payload=$(EMBED_MODEL="$EMBED_MODEL" EMBED_DEVICE="$EMBED_DEVICE" python3 <<'PY'
import json, os
print(json.dumps({
  "query": "mutation($id:ID!,$input:Map!){ configurePlugin(plugin_id:$id, input:$input) }",
  "variables": {"id": "stash-copilot", "input": {
    "image_embedding_provider": "openclip",
    "image_embedding_model": os.environ["EMBED_MODEL"],
    "image_embedding_device": os.environ["EMBED_DEVICE"],
  }},
}))
PY
)
gql "$config_payload" >/dev/null

count=$(gql '{"query":"{ findScenes { count } }"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['findScenes']['count'])")
echo "[bootstrap] done — $count scene(s) in library"
