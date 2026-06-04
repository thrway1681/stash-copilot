#!/bin/sh
# CI UI-test stub for the Stash plugin backend exec.
#
# The real plugin backend (torch / faiss / LLM via uv) cannot run inside the
# Alpine/musl `stashapp/stash` container, so for UI tests Stash exec's THIS stub
# instead of run-plugin.sh. Stash passes the task payload as JSON on stdin
# (server_connection + args, exactly as stash-copilot.py's _read_input reads it).
#
# To make the frontend's dispatchTask flow resolve without a real backend, the
# stub parses the task `mode` (+ request_id / scene_id), maps it to the
# frontend's polled result file (the same {result_key}_{id}.json contract issue
# #4 / #5 define), and writes a canned fixture from ./fixtures/<mode>.json into
# ./assets/ (relative to the plugin dir, which Stash serves at
# /plugin/stash-copilot/assets/). Modes without a fixture are a no-op, so
# coverage grows by adding fixture files (one per feature, per-migration).
#
# Pure POSIX/busybox sh — no python/jq in the Stash image.
set -eu

# Stash exec's plugins with an arbitrary working directory (observed: "/"), so
# resolve ./fixtures and ./assets relative to THIS script's directory (the
# plugin dir) — mirroring the real run-plugin.sh, which cd's to its own dir.
cd "$(dirname "$0")" || exit 0

payload="$(cat)"

# Extract a top-level-ish string field's first occurrence: field "x":"y" -> y.
field() {
  printf '%s' "$payload" \
    | sed -n 's/.*"'"$1"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -n1
}

mode="$(field mode)"
request_id="$(field request_id)"
scene_id="$(field scene_id)"

[ -n "$mode" ] || exit 0

# Map mode -> result_key + how its result file is keyed (mirrors the JS TASKS
# registry's resultKey/keying; see stash-copilot.js).
result_key=""
keyed_by="request_id"
case "$mode" in
  recommendations)          result_key="recommendations" ;;
  build_taste_map)          result_key="taste_map" ;;
  detect_tag_gaps)          result_key="tag_gaps" ;;
  get_scene_tag_gaps)       result_key="tag_gaps_scene" ;;
  preview_tag_impact)       result_key="tag_preview" ;;
  get_tag_suggestions)      result_key="tag_suggestions" ;;
  find_duplicate_tags)      result_key="tag_dedup" ;;
  merge_tags)               result_key="tag_merge" ;;
  dismiss_tag_merge)        result_key="tag_dismiss" ;;
  search_by_text)           result_key="search_results" ;;
  get_embedding_models)     result_key="embedding_models" ;;
  find_similar_by_frame)    result_key="frame_search" ;;
  find_similar_performers)  result_key="similar_performers" ;;
  prepare_labeling_session) result_key="labeling_session" ;;
  sync_labeling_annotations) result_key="labeling_sync" ;;
  export_labeling_dataset)  result_key="labeling_export" ;;
  get_labeling_sessions)    result_key="labeling_sessions" ;;
  find_similar)             result_key="similar_results"; keyed_by="scene_id" ;;
  chat)                     result_key="chat_history";    keyed_by="fixed" ;;
  stats_summary)            result_key="last_summary";    keyed_by="fixed" ;;
  *) exit 0 ;;  # fire-and-forget / no polled result
esac

# A fixture is required to write a result; absent -> no-op (feature not yet
# under UI test).
fixture="./fixtures/${mode}.json"
[ -f "$fixture" ] || exit 0

case "$keyed_by" in
  scene_id)   stem="${result_key}_${scene_id}" ;;
  fixed)      stem="${result_key}" ;;
  *)          stem="${result_key}_${request_id}" ;;
esac

mkdir -p ./assets
# Substitute the live ids into the fixture (templates may reference them).
sed -e "s/__REQUEST_ID__/${request_id}/g" -e "s/__SCENE_ID__/${scene_id}/g" \
  "$fixture" > "./assets/${stem}.json"
exit 0
