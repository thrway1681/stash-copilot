#!/usr/bin/env bash
# Seed the dev/CI Stash with synthetic, NON-ADULT structure for the Playwright
# UI tests: a few performers (with a tiny committable placeholder image), a few
# tags, and scene associations. The UI/dispatchTask tests are stub-backed — they
# verify the frontend CONTRACT, not AI output — so they need DB *structure*, not
# real media. No adult content, no secrets; runs the same locally and on fork PRs.
#
# Run AFTER the library is scanned (scripts/dev_stash_bootstrap.sh), e.g.:
#   STASH_URL=http://localhost:3000 bash scripts/ci/seed_stash_ui.sh
#
# Idempotent: the dev Stash is mutated (not reset) between runs, so entities are
# upserted by exact name and scene associations use replace/ADD semantics.
set -euo pipefail

STASH_URL="${STASH_URL:-http://localhost:3000}"
GQL="$STASH_URL/graphql"

# A 1x1 PNG (70 bytes) — committable, non-adult, just proves a performer image
# renders. The actual pixels are irrelevant (nothing inspects them).
PLACEHOLDER_PNG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

# POST a JSON body, fail loudly on a GraphQL "errors" payload (curl -fsS only
# catches HTTP errors; GraphQL errors come back as HTTP 200 with an errors key).
gql() {
  resp="$(curl -fsS -X POST "$GQL" -H 'Content-Type: application/json' -d "$1")"
  if printf '%s' "$resp" | jq -e '.errors' >/dev/null 2>&1; then
    echo "[seed] GraphQL error: $(printf '%s' "$resp" | jq -c '.errors')" >&2
    return 1
  fi
  printf '%s' "$resp"
}

# ensure_performer NAME GENDER IMAGE_B64(optional) -> echoes the performer id
ensure_performer() {
  name="$1"; gender="$2"; img="${3:-}"
  found="$(gql "$(jq -n --arg n "$name" '{query:"query($n:String!){findPerformers(performer_filter:{name:{value:$n,modifier:EQUALS}}){performers{id}}}",variables:{n:$n}}')")"
  id="$(printf '%s' "$found" | jq -r '.data.findPerformers.performers[0].id // empty')"
  if [ -z "$id" ]; then
    body="$(jq -n --arg n "$name" --arg g "$gender" --arg img "$img" \
      '{query:"mutation($i:PerformerCreateInput!){performerCreate(input:$i){id}}",
        variables:{i:({name:$n,gender:$g} + (if $img=="" then {} else {image:("data:image/png;base64,"+$img)} end))}}')"
    id="$(gql "$body" | jq -r '.data.performerCreate.id')"
  fi
  printf '%s' "$id"
}

# ensure_tag NAME -> echoes the tag id
ensure_tag() {
  name="$1"
  found="$(gql "$(jq -n --arg n "$name" '{query:"query($n:String!){findTags(tag_filter:{name:{value:$n,modifier:EQUALS}}){tags{id}}}",variables:{n:$n}}')")"
  id="$(printf '%s' "$found" | jq -r '.data.findTags.tags[0].id // empty')"
  if [ -z "$id" ]; then
    id="$(gql "$(jq -n --arg n "$name" '{query:"mutation($i:TagCreateInput!){tagCreate(input:$i){id}}",variables:{i:{name:$n}}}')" | jq -r '.data.tagCreate.id')"
  fi
  printf '%s' "$id"
}

# Resolve the 3 oldest scenes by id (don't hardcode — survive scan-order changes).
scenes_json="$(gql '{"query":"{findScenes(filter:{per_page:3,sort:\"id\",direction:ASC}){scenes{id}}}"}')"
S1="$(printf '%s' "$scenes_json" | jq -r '.data.findScenes.scenes[0].id // empty')"
S2="$(printf '%s' "$scenes_json" | jq -r '.data.findScenes.scenes[1].id // empty')"
S3="$(printf '%s' "$scenes_json" | jq -r '.data.findScenes.scenes[2].id // empty')"
if [ -z "$S3" ]; then
  echo "[seed] need >=3 scanned scenes; found: ${S1:-_}/${S2:-_}/${S3:-_}" >&2
  exit 1
fi

# Performers: two with images, one without (exercises the default-avatar path).
PA="$(ensure_performer 'CI Performer Alpha' FEMALE "$PLACEHOLDER_PNG")"
PB="$(ensure_performer 'CI Performer Bravo' MALE "$PLACEHOLDER_PNG")"
PC="$(ensure_performer 'CI Performer Charlie' FEMALE '')"

# Tags.
TG="$(ensure_tag 'ci-tag')"
THD="$(ensure_tag 'hd')"
TFAV="$(ensure_tag 'ci-favorite')"

# Associations. sceneUpdate REPLACES the arrays (idempotent). Aim for a mix:
#   scene 1 = 2 performers + 2 tags, scene 2 = 1 + 1, scene 3 = 0 performers + 1 tag.
gql "$(jq -n --arg s "$S1" --arg a "$PA" --arg b "$PB" --arg t "$TG" --arg h "$THD" \
  '{query:"mutation($i:SceneUpdateInput!){sceneUpdate(input:$i){id}}",variables:{i:{id:$s,performer_ids:[$a,$b],tag_ids:[$t,$h]}}}')" >/dev/null
gql "$(jq -n --arg s "$S2" --arg a "$PA" --arg t "$TG" \
  '{query:"mutation($i:SceneUpdateInput!){sceneUpdate(input:$i){id}}",variables:{i:{id:$s,performer_ids:[$a],tag_ids:[$t]}}}')" >/dev/null
gql "$(jq -n --arg s "$S3" --arg t "$TFAV" \
  '{query:"mutation($i:SceneUpdateInput!){sceneUpdate(input:$i){id}}",variables:{i:{id:$s,performer_ids:[],tag_ids:[$t]}}}')" >/dev/null
# Additively spread 'hd' across all three (BulkUpdateIds {ids,mode}); ADD is a no-op if present.
gql "$(jq -n --arg s1 "$S1" --arg s2 "$S2" --arg s3 "$S3" --arg h "$THD" \
  '{query:"mutation($i:BulkSceneUpdateInput!){bulkSceneUpdate(input:$i){id}}",variables:{i:{ids:[$s1,$s2,$s3],tag_ids:{ids:[$h],mode:"ADD"}}}}')" >/dev/null

# Verify-back: assert the seed converged, fail loudly otherwise.
pc="$(gql '{"query":"{findPerformers{count}}"}' | jq -r '.data.findPerformers.count')"
tc="$(gql '{"query":"{findTags{count}}"}' | jq -r '.data.findTags.count')"
s1p="$(gql "$(jq -n --arg s "$S1" '{query:"query($s:ID!){findScene(id:$s){performers{id} tags{id}}}",variables:{s:$s}}')" | jq -r '.data.findScene.performers|length')"
if [ "${pc:-0}" -lt 3 ] || [ "${tc:-0}" -lt 3 ] || [ "${s1p:-0}" -lt 2 ]; then
  echo "[seed] verify failed: performers=$pc tags=$tc scene1_performers=$s1p" >&2
  exit 1
fi
echo "[seed_stash_ui] ok: performers=$pc, tags=$tc, scenes $S1/$S2/$S3 associated (scene1 performers=$s1p)"
