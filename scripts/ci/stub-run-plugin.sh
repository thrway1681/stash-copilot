#!/bin/sh
# CI UI-test stub for the Stash plugin backend exec.
#
# The real plugin backend (torch / faiss / LLM via uv) cannot run inside the
# Alpine/musl `stashapp/stash` container, so for UI tests Stash exec's THIS stub
# instead of run-plugin.sh. Stash passes the task payload as JSON on stdin
# (server_connection + args, exactly as stash-copilot.py's _read_input reads it).
#
# P0: no-op. The smoke tests fire no backend tasks (auto-analyze is disabled and
# they only open client-side tabs), so the stub just drains stdin and exits 0 so
# Stash records the task as succeeded without a broken pipe.
#
# P1 will extend this to parse `mode` + `request_id` from stdin and write a
# fixture result JSON into ./assets/{result_key}_{request_id}.json so the UI's
# dispatchTask poll resolves end-to-end without a real backend.
cat > /dev/null 2>&1 || true
exit 0
