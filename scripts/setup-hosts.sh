#!/usr/bin/env bash
#
# Install AgentDescent and wire it into every agent host you have.
#
#   bash scripts/setup-hosts.sh              # install AgentDescent, wire up what is present
#   bash scripts/setup-hosts.sh --with-clis  # also npm-install the four agent CLIs
#   bash scripts/setup-hosts.sh --dry-run    # print what it would do, change nothing
#
# It is safe to re-run: every step checks first, and `agentdescent install`
# rewrites only the block it owns.
#
set -uo pipefail

BRANCH="claude/agentdescent-multi-platform-plugin-s41qma"
REPO="https://github.com/Birfy/agentdescent"
DRY=0
WITH_CLIS=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --with-clis) WITH_CLIS=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
run()  { if [ "$DRY" = 1 ]; then printf '   would run: %s\n' "$*"; else eval "$@"; fi; }

# --- 1. prerequisites -------------------------------------------------------
say "Checking prerequisites"
missing=0
for tool in python3 git; do
  if command -v "$tool" >/dev/null 2>&1; then
    info "$tool  $($tool --version 2>&1 | head -1)"
  else
    info "MISSING: $tool"; missing=1
  fi
done
py_ok=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)' 2>/dev/null || echo 0)
if [ "$py_ok" != "1" ]; then
  info "note: Python < 3.10 -- the CLI works, but the MCP server does not"
  info "      (the mcp package requires 3.10+; agentdescent will say so)"
fi
[ "$missing" = 1 ] && { echo "install the missing tools first"; exit 1; }

# --- 2. the agent CLIs ------------------------------------------------------
if [ "$WITH_CLIS" = 1 ]; then
  say "Installing the agent CLIs (npm globals)"
  if ! command -v npm >/dev/null 2>&1; then
    info "npm is not installed -- get Node 18+ from https://nodejs.org, then re-run"
    exit 1
  fi
  # The four packages, at the versions this was verified against. Drop any you
  # do not want; AgentDescent wires up whichever it finds.
  for pkg in "@anthropic-ai/claude-code" "@openai/codex" "opencode-ai" "@deepseek-ai/dsh"; do
    info "npm install -g $pkg"
    run "npm install -g '$pkg' >/dev/null 2>&1 || echo '   (failed: $pkg -- skipping)'"
  done
fi

say "Agent CLIs on PATH"
found=0
for pair in "claude:Claude Code" "codex:Codex" "dsh:DeepSeek Harness" "opencode:OpenCode"; do
  bin="${pair%%:*}"; name="${pair#*:}"
  if command -v "$bin" >/dev/null 2>&1; then
    info "$name ($bin) -- $("$bin" --version 2>&1 | head -1)"
    found=$((found + 1))
  else
    info "$name ($bin) -- not installed"
  fi
done
if [ "$found" = 0 ]; then
  info ""
  info "None found. Re-run with --with-clis to install them, or install the one"
  info "you use. AgentDescent's own demo needs none of them."
fi

# --- 3. AgentDescent --------------------------------------------------------
say "Installing AgentDescent"
if [ -f "pyproject.toml" ] && grep -q 'name = "agentdescent"' pyproject.toml 2>/dev/null; then
  info "from this checkout (editable)"
  run "python3 -m pip install -q -e '.[mcp]'"
else
  info "from the branch (not yet in a PyPI release)"
  run "python3 -m pip install -q 'agentdescent[mcp] @ git+$REPO@$BRANCH'"
fi

if ! command -v agentdescent >/dev/null 2>&1 && [ "$DRY" = 0 ]; then
  info ""
  info "WARNING: 'agentdescent' installed but is not on PATH."
  info "  Hosts start it as a subprocess, so it must be on the PATH of whatever"
  info "  launches your agent -- not just this shell. Add pip's bin directory:"
  info "    export PATH=\"\$(python3 -m site --user-base)/bin:\$PATH\""
  info "  Until then, 'python3 -m agentdescent.cli' works everywhere."
fi

# --- 4. wire up each host ---------------------------------------------------
say "Wiring up the hosts that are present"
declare -A HOST_BIN=( [claude-code]=claude [codex]=codex [dsh]=dsh [opencode]=opencode )
for host in claude-code codex dsh opencode; do
  bin="${HOST_BIN[$host]}"
  if command -v "$bin" >/dev/null 2>&1; then
    info "agentdescent install $host"
    run "agentdescent install '$host' 2>&1 | sed 's/^/     /'"
  else
    info "skipping $host ($bin not installed)"
  fi
done

# --- 5. report --------------------------------------------------------------
say "What this machine can run"
run "agentdescent doctor"

cat <<'NEXT'

== Next

  agentdescent demo          a complete evolution, offline, no API key, ~10s

Then, per host:

  Claude Code   claude --plugin-dir ~/.agentdescent/plugins/claude-code
  Codex         codex plugin marketplace add Birfy/agentdescent
                codex plugin add agentdescent@agentdescent
  DSH           dsh --profile web --dump-config | grep agentdescent
  OpenCode      opencode mcp list

Inside the agent, say it in your own words:

  "Evolve the skill at ./my-skill against ./cases.jsonl"

Full guide: docs/testing-guide.md
NEXT
