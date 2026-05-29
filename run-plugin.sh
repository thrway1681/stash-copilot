#!/bin/bash
# Wrapper script for Stash plugin execution using UV
set -euo pipefail

BASE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/homebrew/bin"
export PATH="${PATH:-}:$BASE_PATH"

# Navigate to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USER_HOME="${HOME:-}"
if [[ -z "$USER_HOME" ]]; then
  USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6 || true)"
fi

SCRIPT_OWNER_HOME=""
SCRIPT_OWNER="$(stat -c '%U' "$SCRIPT_DIR" 2>/dev/null || true)"
if [[ -n "$SCRIPT_OWNER" && "$SCRIPT_OWNER" != "UNKNOWN" ]]; then
  SCRIPT_OWNER_HOME="$(getent passwd "$SCRIPT_OWNER" | cut -d: -f6 || true)"
fi

for UV_HOME in "$USER_HOME" "$SCRIPT_OWNER_HOME"; do
  if [[ -n "$UV_HOME" ]]; then
    export PATH="$UV_HOME/.local/bin:$UV_HOME/.cargo/bin:$PATH"
  fi
done

UV=""
UV_CANDIDATES=()
if [[ -n "${UV_BIN:-}" ]]; then
  UV_CANDIDATES+=("$UV_BIN")
fi
UV_CANDIDATES+=("$(command -v uv || true)")
for UV_HOME in "$USER_HOME" "$SCRIPT_OWNER_HOME"; do
  if [[ -n "$UV_HOME" ]]; then
    UV_CANDIDATES+=("$UV_HOME/.local/bin/uv" "$UV_HOME/.cargo/bin/uv")
  fi
done
UV_CANDIDATES+=("/opt/homebrew/bin/uv" "/usr/local/bin/uv" "/usr/bin/uv")

for UV_CANDIDATE in "${UV_CANDIDATES[@]}"; do
  if [[ -n "$UV_CANDIDATE" && -x "$UV_CANDIDATE" ]]; then
    UV="$UV_CANDIDATE"
    break
  fi
done

if [[ -z "$UV" ]]; then
  cat >&2 <<EOF
Error: uv was not found.

Stash often runs plugins with a smaller PATH than your interactive shell.
The plugin looked for uv on PATH, under the Stash runtime user's home, and
under this plugin owner's home.

Install uv for the user running Stash, add uv to that service's PATH, or set
UV_BIN to the full uv executable path in the Stash service environment.
EOF
  exit 127
fi

# Run the Python script with UV
"$UV" run python "$@"
