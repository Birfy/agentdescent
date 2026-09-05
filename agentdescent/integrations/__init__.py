"""Wire the skill and the MCP server into a host: ``agentdescent install <host>``.

All the logic is in the package (:mod:`agentdescent.cli`, :mod:`agentdescent.mcp`);
what a host needs from this module is *manifests* -- where its skills live,
how it declares an MCP server, whether it reads hooks -- plus one shared
``SKILL.md`` that teaches the host model when and how to call the tools. Each
host is a few dozen lines here and no code anywhere else, which is what keeps
"add a host" from touching the core.

The shared files (``SKILL.md``, ``hooks.json``) ship inside the package so an
installed wheel can write them; the repository's ``integrations/claude-code``
plugin directory is *rendered* from the same files (:func:`render_claude_plugin`)
and a test checks it has not drifted.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

__all__ = ["HOSTS", "install", "render_claude_plugin", "skill_text", "hooks_text"]

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Where the MCP server comes from, in every manifest.
MCP_COMMAND = "agentdescent"
MCP_ARGS = ["mcp"]

#: Provider keys forwarded into the MCP server under dsh, which scrubs any
#: ambient variable matching KEY|PASSWORD|SECRET|TOKEN before starting one.
DSH_FORWARDED_KEYS = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY")


def skill_text() -> str:
    with open(os.path.join(_HERE, "SKILL.md"), encoding="utf-8") as fh:
        return fh.read()


def hooks_text() -> str:
    with open(os.path.join(_HERE, "hooks.json"), encoding="utf-8") as fh:
        return fh.read()


def _version() -> str:
    from .. import __version__
    return __version__


# ---------------------------------------------------------------------------
# file writing, with dry-run and "append if absent"
# ---------------------------------------------------------------------------


class _Writer:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.lines: List[str] = []

    def write(self, path: str, text: str) -> None:
        exists = os.path.exists(path)
        verb = "would write" if self.dry_run else ("updated" if exists else "wrote")
        if not self.dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        self.lines.append(f"{verb} {path}")

    def append_unless(self, path: str, marker: str, block: str, *, what: str) -> None:
        current = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                current = fh.read()
        if marker in current:
            self.lines.append(f"kept {path} ({what} already present)")
            return
        verb = "would append to" if self.dry_run else "appended to"
        if not self.dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                if current and not current.endswith("\n"):
                    fh.write("\n")
                fh.write(block)
        self.lines.append(f"{verb} {path} ({what})")

    def note(self, text: str) -> None:
        self.lines.append(text)


# ---------------------------------------------------------------------------
# DeepSeek Harness
# ---------------------------------------------------------------------------


def dsh_patch_block() -> str:
    """The two ``cordis.patch.yml`` entries: the MCP server and the hooks bridge.

    The ``env`` block is not optional: dsh scrubs every ambient variable matching
    ``KEY|PASSWORD|SECRET|TOKEN`` before starting an MCP server, so without it
    the server has no provider credentials and ``doctor`` reports every reflector
    as unavailable with nothing else to explain why.
    """
    env = "\n".join(f"      {k}: !!js process.env.{k}" for k in DSH_FORWARDED_KEYS)
    return (
        "# --- agentdescent (written by `agentdescent install dsh`) ---\n"
        "- id: mcp-agentdescent\n"
        "  name: '@deepseek-ai/dsh-mcp-client'\n"
        "  config:\n"
        f"    serverName: {MCP_COMMAND}\n"
        "    transport: stdio\n"
        f"    command: {MCP_COMMAND}\n"
        f"    args: {json.dumps(MCP_ARGS)}\n"
        "    toolCallTimeoutMs: 120000\n"
        "    env:\n"
        "      # dsh scrubs KEY|PASSWORD|SECRET|TOKEN from the ambient env; forward explicitly\n"
        f"{env}\n"
        "- id: hooks-agentdescent\n"
        "  name: '@deepseek-ai/dsh-hooks-claude-code'\n"
        "  config:\n"
        "    configPath: ~/.dsh/skills/agentdescent/hooks.json\n"
    )


def install_dsh(home: str, w: _Writer) -> None:
    dsh_home = os.environ.get("DSH_HOME") or os.path.join(home, ".dsh")
    skill_dir = os.path.join(dsh_home, "skills", "agentdescent")
    w.write(os.path.join(skill_dir, "SKILL.md"), skill_text())
    w.write(os.path.join(skill_dir, "hooks.json"), hooks_text())
    w.append_unless(os.path.join(dsh_home, "cordis.patch.yml"), "serverName: agentdescent",
                    dsh_patch_block(), what="mcp-client + hooks entries")
    w.note(f"forwarded into the MCP server env: {', '.join(DSH_FORWARDED_KEYS)}")
    w.note("verify: dsh --profile web --dump-config | grep -n agentdescent")


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------


def plugin_manifest() -> Dict[str, object]:
    return {
        "name": "agentdescent",
        "description": "Evolve skills, agents, prompts, code and plugins against examples "
                       "with a parallel, merge-based optimiser.",
        "version": _version(),
        "author": {"name": "AgentDescent"},
        "homepage": "https://github.com/Birfy/agentdescent",
        "keywords": ["evolution", "skills", "optimisation", "self-improvement"],
    }


def marketplace_manifest(source: str = "./integrations/claude-code") -> Dict[str, object]:
    return {
        "name": "agentdescent",
        "owner": {"name": "Birfy"},
        "metadata": {"description": "AgentDescent: evolve your skills, agents and plugins."},
        "plugins": [{
            "name": "agentdescent",
            "source": source,
            "description": plugin_manifest()["description"],
            "version": _version(),
        }],
    }


def mcp_manifest() -> Dict[str, object]:
    return {"mcpServers": {"agentdescent": {"command": MCP_COMMAND, "args": MCP_ARGS}}}


EVOLVE_COMMAND = """---
description: Evolve a skill, agent, prompt, codebase or plugin against examples with AgentDescent
argument-hint: <path> [cases.jsonl]
---

Use the agentdescent skill. The target is `$ARGUMENTS` (a path, optionally
followed by a cases file). Run `doctor`, build the spec, `plan` it, show me the
spec and the estimate, and wait for my yes before `start`.
"""


def render_claude_plugin(dest: str, w: Optional[_Writer] = None) -> List[str]:
    """Materialise the Claude Code plugin directory at ``dest``."""
    w = w or _Writer(dry_run=False)
    w.write(os.path.join(dest, ".claude-plugin", "plugin.json"),
            json.dumps(plugin_manifest(), indent=2) + "\n")
    w.write(os.path.join(dest, "skills", "agentdescent", "SKILL.md"), skill_text())
    w.write(os.path.join(dest, "commands", "evolve.md"), EVOLVE_COMMAND)
    w.write(os.path.join(dest, ".mcp.json"), json.dumps(mcp_manifest(), indent=2) + "\n")
    w.write(os.path.join(dest, "hooks", "hooks.json"), hooks_text())
    return w.lines


def install_claude_code(home: str, w: _Writer) -> None:
    dest = os.path.join(home, ".agentdescent", "plugins", "claude-code")
    render_claude_plugin(dest, w)
    w.note(f"load it:   claude --plugin-dir {dest}")
    w.note("or, from the repository's marketplace: /plugin marketplace add Birfy/agentdescent "
           "&& /plugin install agentdescent@agentdescent")


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


def codex_config_block() -> str:
    return ("\n# --- agentdescent (written by `agentdescent install codex`) ---\n"
            "[mcp_servers.agentdescent]\n"
            f'command = "{MCP_COMMAND}"\n'
            f"args = {json.dumps(MCP_ARGS)}\n")


def install_codex(home: str, w: _Writer) -> None:
    codex_home = os.environ.get("CODEX_HOME") or os.path.join(home, ".codex")
    w.write(os.path.join(codex_home, "skills", "agentdescent", "SKILL.md"), skill_text())
    w.append_unless(os.path.join(codex_home, "config.toml"), "[mcp_servers.agentdescent]",
                    codex_config_block(), what="mcp_servers entry")
    w.note("Codex has no hooks; add `run agentdescent status --brief at the start of a "
           "session` to AGENTS.md if you want in-progress runs surfaced.")


HOSTS: Dict[str, Callable[[str, _Writer], None]] = {
    "dsh": install_dsh,
    "claude-code": install_claude_code,
    "codex": install_codex,
}


def mcp_sdk_missing() -> bool:
    """Whether ``agentdescent mcp`` would fail for want of the protocol SDK."""
    try:
        __import__("mcp")
        return False
    except ImportError:
        return True


def install(host: str, *, dry_run: bool = False, home: Optional[str] = None) -> List[str]:
    """Wire ``host`` up. Returns the lines to print: what was written or kept."""
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}; choose from {sorted(HOSTS)}")
    w = _Writer(dry_run)
    HOSTS[host](os.path.expanduser(home or "~"), w)
    if mcp_sdk_missing():
        # Every manifest written above tells the host to run `agentdescent mcp`.
        # Without the SDK that subprocess exits immediately and the host reports
        # only "CONNECTION_CLOSED" -- so say it here, where the user is looking,
        # rather than let them discover it as a silent missing tool later.
        w.note("WARNING: the 'mcp' package is not installed, so the server this "
               "just wired up cannot start. Fix with: pip install \"agentdescent[mcp]\"")
    return w.lines
