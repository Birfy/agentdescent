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

__all__ = ["HOSTS", "install", "render_claude_plugin", "render_dsh_plugin",
           "skill_text", "hooks_text"]

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


def _skill_frontmatter() -> Dict[str, str]:
    """The YAML frontmatter of the shared skill, as a flat mapping.

    Hand-parsed rather than with PyYAML: the core has no dependencies, and the
    file is this package's own with a known two-key header.
    """
    text = skill_text()
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md has no frontmatter")
    header = text.split("---\n", 2)[1]
    out: Dict[str, str] = {}
    key = None
    for line in header.splitlines():
        if line[:1].isspace() and key:
            out[key] += " " + line.strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            out[key] = value.strip()
    return out


def _skill_description() -> str:
    return _skill_frontmatter()["description"]


def _skill_body() -> str:
    """The skill markdown, without the frontmatter the registry supplies itself."""
    return skill_text().split("---\n", 2)[2].lstrip("\n")


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
    """The two ``cordis.patch.yml`` rows: the MCP server and the hooks bridge.

    Wrapped in ``- insert:`` because a dsh patch file **overrides existing rows
    by id** -- a bare list of new rows is read as an override of ids that do not
    exist, and dsh says so (``patch: entry "mcp-agentdescent" not found``) and
    then composes without them. ``insert`` is how ``dsh-base`` itself adds its
    rows; verified against dsh 0.1.2-rc.1 with ``dsh --profile headless
    --dump-config``.

    The ``env`` block is not optional either: dsh scrubs every ambient variable
    matching ``KEY|PASSWORD|SECRET|TOKEN`` before starting an MCP server, so
    without it the server has no provider credentials and ``doctor`` reports
    every reflector as unavailable with nothing else to explain why.
    """
    env = "\n".join(f"          {k}: !!js process.env.{k}" for k in DSH_FORWARDED_KEYS)
    return (
        "# --- agentdescent (written by `agentdescent install dsh`) ---\n"
        "# `insert` adds rows; a bare row would be read as an override by id.\n"
        "- insert:\n"
        "    - id: mcp-agentdescent\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        f"        serverName: {MCP_COMMAND}\n"
        "        transport: stdio\n"
        f"        command: {MCP_COMMAND}\n"
        f"        args: {json.dumps(MCP_ARGS)}\n"
        "        toolCallTimeoutMs: 120000\n"
        "        env:\n"
        "          # dsh scrubs KEY|PASSWORD|SECRET|TOKEN from the ambient env\n"
        f"{env}\n"
        "    - id: hooks-agentdescent\n"
        "      name: '@deepseek-ai/dsh-hooks-claude-code'\n"
        "      config:\n"
        "        configPath: ~/.dsh/skills/agentdescent/hooks.json\n"
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


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------


def opencode_mcp_entry() -> Dict[str, object]:
    """The ``mcp`` entry, in the shape ``opencode mcp add`` itself writes.

    Verified by running ``opencode mcp add agentdescent -- agentdescent mcp``
    against opencode 1.18 and reading the file back: ``type: "local"``, the
    command as one array (not command + args), and ``environment`` for env vars.
    """
    return {"type": "local", "command": [MCP_COMMAND, *MCP_ARGS]}


def install_opencode(home: str, w: _Writer) -> None:
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    root = os.path.join(config_home, "opencode")
    # `{skill,skills}/**/SKILL.md` -- both spellings are accepted; `skill` is
    # what the sibling `agent` / `command` directories are named.
    w.write(os.path.join(root, "skill", "agentdescent", "SKILL.md"), skill_text())

    path = os.path.join(root, "opencode.jsonc")
    current, raw = {}, ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        try:
            current = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            # JSONC allows comments, which json cannot read. Merging by hand
            # would corrupt the file, so say what to add instead of guessing.
            w.note("NOTE: could not parse opencode.jsonc (comments?); add this "
                   "under \"mcp\" yourself: "
                   + json.dumps({"agentdescent": opencode_mcp_entry()}))
            return
    if (current.get("mcp") or {}).get("agentdescent"):
        w.lines.append(f"kept {path} (mcp entry already present)")
        return
    merged = dict(current)
    merged.setdefault("$schema", "https://opencode.ai/config.json")
    merged["mcp"] = {**(current.get("mcp") or {}), "agentdescent": opencode_mcp_entry()}
    w.write(path, json.dumps(merged, indent=2) + "\n")
    w.note("verify: opencode mcp list")


# ---------------------------------------------------------------------------
# The dsh Cordis plugin
# ---------------------------------------------------------------------------

#: npm name of the native DeepSeek Harness plugin rendered into
#: ``integrations/dsh-agentdescent`` and installed with
#: ``dsh plugin --profile <p> add link:<path>``.
DSH_PLUGIN_NAME = "dsh-agentdescent"

#: The plugin module, with the skill text filled in. Kept as a template so the
#: skill can never drift from the one the file-based install writes.
DSH_PLUGIN_JS = """// Generated by `agentdescent.integrations.render_dsh_plugin()`. Do not edit by
// hand: the skill body below is the same SKILL.md every other host installs,
// and a test fails if this file drifts from it.
//
// A Cordis plugin is `{{ name, inject, apply }}` -- the shape every bundled dsh
// plugin uses (see @deepseek-ai/dsh-skill-filesystem). `inject: ['skills']`
// makes Cordis wait for the skill registry to exist before calling `apply`.

export const name = 'agentdescent';

export const inject = ['skills'];

/** The shared SKILL.md body, minus the frontmatter the registry supplies. */
const CONTENT = {content};

const DESCRIPTION = {description};

export function apply(ctx) {{
  // `register` returns the Cordis disposer; yielding it from `ctx.effect` is
  // what removes the skill again when the plugin is unloaded or reloaded.
  ctx.effect(function* () {{
    yield ctx.skills.register({{
      name: 'agentdescent',
      description: DESCRIPTION,
      content: CONTENT,
      source: 'runtime',
    }});
  }}, 'agentdescent skill');

  if (ctx.logger && ctx.logger.info) {{
    ctx.logger.info(
      'agentdescent: skill registered. The MCP tools come from the ' +
      'mcp-agentdescent row in the cordis.patch.yml of this package; ' +
      'run `agentdescent doctor` if they are missing.');
  }}
}}
"""


def dsh_plugin_package() -> Dict[str, object]:
    """``package.json`` for the Cordis plugin.

    ``peerDependencies`` rather than ``dependencies``: a Cordis plugin runs
    inside the host's own copy of the framework, and bundling a second one is
    how two incompatible registries end up in one process. This mirrors what
    ``@deepseek-ai/dsh-skill-filesystem`` declares.
    """
    return {
        "name": DSH_PLUGIN_NAME,
        "version": _version(),
        "description": "Evolve skills, agents, prompts, code and plugins from inside "
                       "DeepSeek Harness, with AgentDescent.",
        "keywords": ["dsh-plugin", "dsh", "agentdescent", "self-improvement"],
        "license": "MIT",
        "type": "module",
        "main": "lib/index.js",
        "files": ["lib/index.js", "cordis.patch.yml", "README.md"],
        # Without `dsh.bundle` the package installs as a plain dependency and
        # its patch never applies -- `dsh plugin add` warns "declares no
        # dsh.bundle ... not a profile layer" and the plugin is inert. This is
        # what dsh-base / dsh-web-app / dsh-headless declare.
        "dsh": {"bundle": {"patch": "./cordis.patch.yml"}},
        "peerDependencies": {"@deepseek-ai/dsh-skill": "*"},
        "repository": {
            "type": "git",
            "url": "git+https://github.com/Birfy/agentdescent.git",
            "directory": "integrations/dsh-agentdescent",
        },
    }


def dsh_plugin_source() -> str:
    """The plugin module: registers the shared skill at runtime.

    Registering through ``ctx.skills.register()`` means the skill needs no file
    on disk and cannot fall out of step with the package -- the reason to prefer
    the plugin over the files ``agentdescent install dsh`` writes.
    """
    return DSH_PLUGIN_JS.format(content=json.dumps(_skill_body()),
                                description=json.dumps(_skill_description()))


def dsh_plugin_patch() -> str:
    """The plugin's own ``cordis.patch.yml``: itself, plus the MCP server row."""
    env = "\n".join("          %s: !!js process.env.%s" % (k, k)
                    for k in DSH_FORWARDED_KEYS)
    return (
        "# Rows this plugin contributes. `insert` adds rows; a bare row would be\n"
        "# read as an override of an id that does not exist, which dsh reports as\n"
        '# `patch: entry "..." not found` and then composes without it.\n'
        "- insert:\n"
        "    - id: %s\n" % DSH_PLUGIN_NAME +
        "      name: '%s'\n" % DSH_PLUGIN_NAME +
        "    - id: mcp-agentdescent\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        "        serverName: %s\n" % MCP_COMMAND +
        "        transport: stdio\n"
        "        command: %s\n" % MCP_COMMAND +
        "        args: %s\n" % json.dumps(MCP_ARGS) +
        "        toolCallTimeoutMs: 120000\n"
        "        env:\n"
        "          # dsh scrubs KEY|PASSWORD|SECRET|TOKEN from the ambient env\n"
        + env + "\n"
    )


DSH_PLUGIN_README = """# dsh-agentdescent

Evolve skills, agent definitions, prompts, code and host plugins from inside
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), using
[AgentDescent](https://github.com/Birfy/agentdescent).

```bash
pip install "agentdescent[mcp]"            # the tools this plugin exposes
dsh plugin --profile web add link:/path/to/dsh-agentdescent
```

The plugin registers the `agentdescent` skill at runtime -- no file to install,
and it cannot drift from the package -- and its `cordis.patch.yml` adds the
`agentdescent mcp` server, so the model gets `mcp__agentdescent__plan`,
`__start`, `__status`, `__show`, `__apply` and the rest.

Check it composed:

```bash
dsh --profile web --dump-config | grep -n agentdescent
```

`agentdescent install dsh` is the alternative that needs no npm install: it
writes the same skill as a file into `$DSH_HOME/skills/` and the same MCP row
into `$DSH_HOME/cordis.patch.yml`. Use one or the other, not both.

Generated from the Python package; see `agentdescent/integrations/__init__.py`.
"""


def render_dsh_plugin(dest: str, w: Optional["_Writer"] = None) -> List[str]:
    """Materialise the Cordis plugin package at ``dest``."""
    w = w or _Writer(dry_run=False)
    w.write(os.path.join(dest, "package.json"),
            json.dumps(dsh_plugin_package(), indent=2) + "\n")
    w.write(os.path.join(dest, "lib", "index.js"), dsh_plugin_source())
    w.write(os.path.join(dest, "cordis.patch.yml"), dsh_plugin_patch())
    w.write(os.path.join(dest, "README.md"), DSH_PLUGIN_README)
    return w.lines


HOSTS: Dict[str, Callable[[str, _Writer], None]] = {
    "dsh": install_dsh,
    "claude-code": install_claude_code,
    "codex": install_codex,
    "opencode": install_opencode,
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
