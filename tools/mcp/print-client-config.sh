#!/bin/sh
# Print the MCP client configuration for libre-dt-lab-mcp.
#
# Usage:
#   tools/mcp/print-client-config.sh [path-to-libre-dt-lab-mcp]
#
# With no path, look in build/bin and then PATH.

set -eu

bin="${1:-}"
if [ -z "$bin" ]; then
  script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
  for candidate in \
    "$script_dir/../../build/bin/libre-dt-lab-mcp" \
    "$script_dir/../../build-make/bin/libre-dt-lab-mcp"; do
    if [ -x "$candidate" ]; then bin=$candidate; break; fi
  done
fi
if [ -z "$bin" ]; then
  bin=$(command -v libre-dt-lab-mcp 2>/dev/null || true)
fi
if [ -z "$bin" ] || [ ! -x "$bin" ]; then
  echo "error: libre-dt-lab-mcp not found; pass its path as the first argument" >&2
  exit 1
fi

# absolute path without relying on GNU realpath
bin_dir=$(CDPATH= cd -- "$(dirname -- "$bin")" && pwd)
bin="$bin_dir/$(basename -- "$bin")"

config="${XDG_CONFIG_HOME:-$HOME/.config}/libre-dt-lab-mcp"
cache="${XDG_CACHE_HOME:-$HOME/.cache}/libre-dt-lab-mcp"

cat <<TXT
Create the dedicated server directories:

  mkdir -p "$config" "$cache"

Claude Code:

  claude mcp add libre-dt-lab -- \\
    "$bin" \\
    --core --configdir "$config" \\
           --cachedir  "$cache"

Claude Desktop (claude_desktop_config.json):

{
  "mcpServers": {
    "libre-dt-lab": {
      "command": "$bin",
      "args": [
        "--core",
        "--configdir", "$config",
        "--cachedir",  "$cache"
      ]
    }
  }
}

Read-only mode: put --read-only before --core.
Real catalog: replace --configdir/--cachedir with:
  --core --library /path/to/library.db
The GUI must not hold that library.
TXT
