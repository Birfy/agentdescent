#!/usr/bin/env bash
# Install a freshly built wheel in a clean virtualenv outside the checkout and
# exercise its CLI and, where supported, its MCP stdio protocol.
set -euo pipefail

# Ambient Python path overrides can make both pip and the smoke checks load
# modules from outside the venv. Keep the entire wheel gate isolated.
unset PYTHONPATH PYTHONHOME

if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
  echo "usage: $0 path/to/agentdescent-*.whl" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
wheel_path="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

python3 -m venv "$temp_dir/venv"
cd "$temp_dir"
export PATH="$temp_dir/venv/bin:$PATH"
"$temp_dir/venv/bin/python" -m pip install --disable-pip-version-check "${wheel_path}[mcp]"
"$temp_dir/venv/bin/python" "$repo_root/scripts/smoke-installed-wheel.py"
