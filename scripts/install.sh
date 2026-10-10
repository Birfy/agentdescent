#!/usr/bin/env bash
# Install current AgentDescent main and wire up supported agent CLIs already on PATH.
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo 'AgentDescent needs Python 3.10+ (python3 was not found).' >&2
  exit 1
fi
if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo 'The AgentDescent MCP plugin needs Python 3.10 or newer.' >&2
  exit 1
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo 'Python pip is missing. Install pip for python3, then run this script again.' >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo 'Git is needed to install the current main branch. Install git, then rerun.' >&2
  exit 1
fi

printf '\n[1/3] Installing AgentDescent from current main with MCP support...\n'
python3 -m pip install --upgrade 'agentdescent[mcp] @ git+https://github.com/Birfy/agentdescent.git@main'

# pip may install a user script outside the current PATH.
if ! command -v agentdescent >/dev/null 2>&1; then
  user_bin="$(python3 -m site --user-base)/bin"
  if [ -x "$user_bin/agentdescent" ]; then
    export PATH="$user_bin:$PATH"
  fi
fi
if ! command -v agentdescent >/dev/null 2>&1; then
  echo 'Installed, but agentdescent is not on PATH. Add the pip scripts directory to PATH and rerun.' >&2
  echo 'Find it with: python3 -m site --user-base' >&2
  exit 1
fi

printf '\n[2/3] Connecting agent CLIs found on PATH...\n'
connected=0
for pair in 'claude:claude-code' 'codex:codex' 'opencode:opencode' 'dsh:dsh'; do
  cli="${pair%%:*}"
  host="${pair#*:}"
  if command -v "$cli" >/dev/null 2>&1; then
    printf '  %s → agentdescent install %s\n' "$cli" "$host"
    agentdescent install "$host"
    connected=$((connected + 1))
  fi
done
if [ "$connected" -eq 0 ]; then
  echo '  No Claude Code, Codex, OpenCode or DeepSeek Harness CLI found.'
  echo '  Install your agent CLI, then rerun this script to connect it.'
fi

printf '\n[3/3] Checking your setup...\n'
if ! agentdescent doctor; then
  echo 'Doctor listed items to configure before a model-backed run; installation is complete.'
fi
printf '\nDone. Restart your agent, then ask it to improve a skill or prompt.\n'
printf 'For Claude Code, launch: claude --plugin-dir ~/.agentdescent/plugins/claude-code\n'
