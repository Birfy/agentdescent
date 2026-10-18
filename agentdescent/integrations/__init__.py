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
import ntpath
import os
import re
import shlex
import sys
import sysconfig
from typing import Callable, Dict, List, Optional

__all__ = ["HOSTS", "host_status", "install", "uninstall", "render_claude_plugin", "render_dsh_plugin",
           "skill_text", "hooks_text"]

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Where the MCP server comes from, in every manifest.
MCP_COMMAND = "agentdescent"
MCP_ARGS = ["mcp"]

#: Provider keys forwarded into the MCP server under dsh, which scrubs any
#: ambient variable matching KEY|PASSWORD|SECRET|TOKEN before starting one.
DSH_FORWARDED_KEYS = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY")

#: What Codex forwards into the MCP server. Codex starts a stdio server with a
#: scrubbed environment (``HOME LOGNAME PATH SHELL USER LANG LC_ALL TERM TMPDIR
#: TZ``) plus whatever ``env_vars`` names (#404). Names only, no globs: the
#: provider keys the reflectors read, ``ARK_API_KEY`` (the key the docs' Ark
#: examples set via ``api_key_env``), and ``AGENTDESCENT_HOME`` so a run started
#: from Codex lands in the same run store the CLI reads. A variable that is not
#: set in the launching shell is simply not forwarded. A spec whose
#: ``api_key_env`` names some other variable needs that name added by hand.
FORWARDED_ENV = DSH_FORWARDED_KEYS + (
    "ANTHROPIC_BASE_URL", "ARK_API_KEY", "AGENTDESCENT_HOME")


def skill_text() -> str:
    with open(os.path.join(_HERE, "SKILL.md"), encoding="utf-8") as fh:
        return fh.read()


def hooks_text(launcher: Optional[List[str]] = None) -> str:
    with open(os.path.join(_HERE, "hooks.json"), encoding="utf-8") as fh:
        text = fh.read()
    if launcher is None:
        return text
    hooks = json.loads(text)
    # Hooks are shell commands, unlike the argv arrays used for MCP.
    # The shipped hook uses POSIX redirection, including Git Bash on Windows.
    command = " ".join(shlex.quote(arg) for arg in launcher)
    for group in hooks["hooks"]["SessionStart"]:
        for hook in group["hooks"]:
            hook["command"] = command + hook["command"][len(MCP_COMMAND):]
    return json.dumps(hooks, indent=2) + "\n"


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
    def __init__(self, dry_run: bool, launcher: Optional[List[str]] = None) -> None:
        self.dry_run = dry_run
        self.launcher = launcher or [MCP_COMMAND]
        self.lines: List[str] = []

    def write(self, path: str, text: str) -> None:
        exists = os.path.exists(path)
        verb = "would write" if self.dry_run else ("updated" if exists else "wrote")
        if not self.dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        self.lines.append(f"{verb} {path}")

    def append_unless(self, path: str, marker: str, block: str, *, what: str,
                      previous: Optional[Callable[[str], Optional[str]]] = None) -> None:
        current = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                current = fh.read()
        old_block = previous(current) if previous else None
        if old_block and old_block in current:
            tail = current.split(old_block, 1)[1].lstrip()
            # Old installs appended this block last. Only repair that exact
            # shape; extra fields or nested tables may customize the launcher.
            if not tail and old_block.rstrip() != block.rstrip():
                self.write(path, current.replace(old_block, block, 1))
                return
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

    def sync_block(self, path: str, start: str, end: str, block: str, *,
                   what: str) -> None:
        """Append our block, or bring an existing one up to date.

        `append_unless` could only ever add: once a marker was in the file it
        said "already present" and left whatever was there. That is wrong for a
        block whose *content* is a bug fix -- an install that had written a
        broken dsh entry could never repair it, and dsh refuses to start at all
        on a bad entry, so the user's only route back was editing the file by
        hand.

        A block written before the end marker existed has no end marker, and
        was always appended last, so it runs to the end of the file: that is
        what gets replaced, and the caller is told it happened.
        """
        current = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                current = fh.read()
        wanted = f"{start}\n{block.rstrip()}\n{end}\n"
        if start not in current:
            verb = "would append to" if self.dry_run else "appended to"
            new_text = (current + ("" if not current or current.endswith("\n") else "\n")
                        + wanted)
            note = f"{verb} {path} ({what})"
        else:
            head = current[:current.index(start)]
            rest = current[current.index(start):]
            tail = rest[rest.index(end) + len(end):] if end in rest else ""
            new_text = head + wanted + tail.lstrip("\n")
            if new_text == current:
                self.lines.append(f"kept {path} ({what} already up to date)")
                return
            verb = "would rewrite" if self.dry_run else "rewrote"
            note = f"{verb} {path} ({what} were out of date)"
        if not self.dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new_text)
        self.lines.append(note)

    def note(self, text: str) -> None:
        self.lines.append(text)


# ---------------------------------------------------------------------------
# DeepSeek Harness
# ---------------------------------------------------------------------------


#: Delimiters for the block `install dsh` owns in `cordis.patch.yml`. The start
#: marker is the one older installs wrote, so it still finds them; the end marker
#: is what lets a later install rewrite the block without eating the rest of the
#: file. See `_Writer.sync_block`.
DSH_BLOCK_START = "# --- agentdescent (written by `agentdescent install dsh`) ---"
DSH_BLOCK_END = "# --- end agentdescent ---"


def _dsh_env_expression() -> str:
    """The ``env:`` value both dsh patch files use.

    One quoted ``!!js`` expression that drops unset variables, and it has to be
    all three of those things:

    * **one expression**, because ``env`` is validated as
      ``{[key: string]: string}`` and ``process.env.X`` for an unset variable is
      ``undefined``. Per-key lines therefore made dsh reject the entry -- and it
      fails the *whole plugin tree* on a bad entry, so an install left dsh
      unable to start for anyone without those keys exported;
    * **filtered**, for the same reason: the keys that are missing must not
      appear at all;
    * **quoted**, because the expression contains ``": "`` and YAML would read
      ``OPENAI_API_KEY: process.env...`` inside it as a nested mapping entry
      ("bad indentation of a mapping entry") and refuse the file.

    Reproduced and fixed against dsh 0.1.2-rc.1 with no provider keys set.
    """
    pairs = ", ".join(f"{k}: process.env.{k}" for k in DSH_FORWARDED_KEYS)
    return ("!!js 'Object.fromEntries(Object.entries({" + pairs +
            "}).filter(([, v]) => v != null))'")


def dsh_patch_block(launcher: Optional[List[str]] = None) -> str:
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

    It is **one** ``!!js`` expression that filters unset keys, rather than one
    line per key, and that is not a style choice. ``env`` is validated as
    ``{[key: string]: string}``, and ``process.env.X`` for a variable that is
    not set is ``undefined``, which is not a string. Writing the keys one per
    line therefore made dsh reject the entry -- and dsh does not skip a bad
    entry, it fails the whole plugin tree, so `install dsh` left dsh unable to
    start at all for anyone without those variables set. (The error even prints
    the offending object as ``"env":{}``, because JSON.stringify drops the
    undefined values that are the actual problem.) Reproduced and fixed against
    dsh 0.1.2-rc.1 with no provider keys in the environment.
    """
    launcher = launcher or [MCP_COMMAND]
    env = "          " + _dsh_env_expression()
    return (
        "# `insert` adds rows; a bare row would be read as an override by id.\n"
        "- insert:\n"
        "    - id: mcp-agentdescent\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        f"        serverName: {MCP_COMMAND}\n"
        "        transport: stdio\n"
        f"        command: {json.dumps(launcher[0], ensure_ascii=False)}\n"
        f"        args: {json.dumps(launcher[1:] + MCP_ARGS)}\n"
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
    w.write(os.path.join(skill_dir, "hooks.json"), hooks_text(w.launcher))
    w.sync_block(os.path.join(dsh_home, "cordis.patch.yml"),
                 DSH_BLOCK_START, DSH_BLOCK_END, dsh_patch_block(w.launcher),
                 what="mcp-client + hooks entries")
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


def mcp_manifest(launcher: Optional[List[str]] = None) -> Dict[str, object]:
    launcher = launcher or [MCP_COMMAND]
    return {"mcpServers": {"agentdescent": {
        "command": launcher[0], "args": launcher[1:] + MCP_ARGS,
        # Claude Code already hands the server its whole environment, so this is
        # a no-op there; the Codex plugin route would otherwise scrub it (#404).
        "env_vars": list(FORWARDED_ENV)}}}


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
    w.write(os.path.join(dest, ".mcp.json"), json.dumps(mcp_manifest(w.launcher), indent=2) + "\n")
    w.write(os.path.join(dest, "hooks", "hooks.json"),
            hooks_text(w.launcher) if w.launcher != [MCP_COMMAND] else hooks_text())
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


def codex_config_block(launcher: Optional[List[str]] = None) -> str:
    return _codex_block(launcher, env_vars=True)


def _codex_block(launcher: Optional[List[str]], *, env_vars: bool) -> str:
    launcher = launcher or [MCP_COMMAND]
    return ("\n# --- agentdescent (written by `agentdescent install codex`) ---\n"
            "[mcp_servers.agentdescent]\n"
            f"command = {json.dumps(launcher[0], ensure_ascii=False)}\n"
            f"args = {json.dumps(launcher[1:] + MCP_ARGS)}\n"
            + (f"env_vars = {json.dumps(list(FORWARDED_ENV))}\n" if env_vars else ""))


def _legacy_codex_block(launcher: Optional[List[str]] = None) -> str:
    """What installs before #404 wrote: no ``env_vars``, so Codex scrubbed the keys."""
    return _codex_block(launcher, env_vars=False)


_PYTHON_NAME = re.compile(r"(?:pythonw?|pypy)(?:[0-9]+(?:\.[0-9]+)*)?(?:\.exe)?", re.IGNORECASE)
_SCRIPT_NAME = re.compile(r"agentdescent(?:\.exe)?", re.IGNORECASE)


def _stock_mcp_argv(command: object) -> bool:
    """Recognize only the complete argv shapes emitted by this installer.

    An absolute conventional Python interpreter with the exact module invocation
    -- with or without the ``-P`` that keeps the host's cwd off ``sys.path`` (#394)
    -- or the absolute path of the ``agentdescent`` console script, is a pinned
    stock launcher, even after its environment disappears. Unknown executable
    names, extra arguments and relative paths remain user-owned.
    """
    if command == [MCP_COMMAND, *MCP_ARGS]:
        return True
    if not (isinstance(command, list) and command and isinstance(command[0], str)
            and (os.path.isabs(command[0]) or ntpath.isabs(command[0]))):
        return False
    name = ntpath.basename(command[0])
    if _SCRIPT_NAME.fullmatch(name):
        return command[1:] == MCP_ARGS
    return (_PYTHON_NAME.fullmatch(name) is not None
            and command[1:] in (["-m", "agentdescent.cli", *MCP_ARGS],
                                ["-P", "-m", "agentdescent.cli", *MCP_ARGS]))


def _stock_codex_block(current: str) -> Optional[str]:
    """Find an untouched installer-owned final block, legacy or pinned.

    Match our marker and exact serialization, not merely a TOML table name.
    In particular, inline fields and nested env tables must not be overwritten.
    """
    marker = "\n# --- agentdescent (written by `agentdescent install codex`) ---\n"
    if current.count(marker) != 1 or current.count("[mcp_servers.agentdescent]") != 1:
        return None
    previous = current[current.index(marker):]
    lines = previous.rstrip().splitlines()
    if (len(lines) not in (5, 6) or lines[2] != "[mcp_servers.agentdescent]"
            or not lines[3].startswith("command = ")
            or not lines[4].startswith("args = ")):
        return None
    try:
        command = json.loads(lines[3][len("command = "):])
        args = json.loads(lines[4][len("args = "):])
    except (ValueError, TypeError):
        return None
    if not isinstance(args, list) or not _stock_mcp_argv([command, *args]):
        return None
    launcher = [command, *args[:-len(MCP_ARGS)]]
    stock = (codex_config_block(launcher), _legacy_codex_block(launcher))
    return previous if previous.rstrip() in (b.rstrip() for b in stock) else None


def install_codex(home: str, w: _Writer) -> None:
    codex_home = os.environ.get("CODEX_HOME") or os.path.join(home, ".codex")
    w.write(os.path.join(codex_home, "skills", "agentdescent", "SKILL.md"), skill_text())
    w.append_unless(os.path.join(codex_home, "config.toml"), "[mcp_servers.agentdescent]",
                    codex_config_block(w.launcher), what="mcp_servers entry",
                    previous=_stock_codex_block)
    w.note(f"forwarded into the MCP server env (Codex scrubs the rest): "
           f"{', '.join(FORWARDED_ENV)}")
    w.note("verify: codex mcp list   (the Env column names them)")
    w.note("Codex has no hooks; add `run agentdescent status --brief at the start of a "
           "session` to AGENTS.md if you want in-progress runs surfaced.")
    # Codex reads the same plugin format as Claude Code, and installing the
    # plugin brings the MCP server with it -- one command instead of the two
    # edits above. Verified against codex-cli 0.153.4.
    w.note("or, in one command instead of the two edits above: "
           "codex plugin marketplace add Birfy/agentdescent && "
           "codex plugin add agentdescent@agentdescent")


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------


def opencode_mcp_entry(launcher: Optional[List[str]] = None) -> Dict[str, object]:
    """The ``mcp`` entry, in the shape ``opencode mcp add`` itself writes.

    Verified by running ``opencode mcp add agentdescent -- agentdescent mcp``
    against opencode 1.18 and reading the file back: ``type: "local"``, the
    command as one array (not command + args), and ``environment`` for env vars.
    """
    return {"type": "local", "command": [*(launcher or [MCP_COMMAND]), *MCP_ARGS]}


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
                   + json.dumps({"agentdescent": opencode_mcp_entry(w.launcher)}))
            return
    existing = (current.get("mcp") or {}).get("agentdescent")
    stock = (isinstance(existing, dict) and set(existing) == {"type", "command"}
             and existing["type"] == "local" and _stock_mcp_argv(existing["command"]))
    if existing and not stock:
        w.lines.append(f"kept {path} (mcp entry already present)")
        return
    merged = dict(current)
    merged.setdefault("$schema", "https://opencode.ai/config.json")
    merged["mcp"] = {**(current.get("mcp") or {}), "agentdescent": opencode_mcp_entry(w.launcher)}
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


#: The browser bundle, in the shipped `window.__ModuleLoader__.load` shape.
#: Hand-written rather than compiled: dsh's client packages are built by their
#: own toolchain (TSX + CSS modules), and the loaded module is plain JS with
#: `React.createElement` and inline styles, which needs no build step here.
DSH_CLIENT_JS = """window.__ModuleLoader__.load({
  id: 'dsh-agentdescent',
  factory: (require) => {
    var module = { exports: {} };
    var exports = module.exports;
    const React = require('react');

    // Where `agentdescent serve` listens. Read-only and loopback-only; it
    // answers with CORS only to a loopback Origin, which is why this panel can
    // read it from the dsh web page and a random website cannot.
    const PANEL_URL = @@PANEL_URL@@;
    const POLL_MS = 5000;

    const LIVE = { running: 1, created: 1 };

    function useRuns() {
      const [state, setState] = React.useState({ runs: [], error: null, loaded: false });
      React.useEffect(() => {
        let alive = true;
        const tick = () => {
          fetch(PANEL_URL + 'api/runs', { cache: 'no-store' })
            .then((r) => r.json())
            .then((runs) => { if (alive) setState({ runs: runs, error: null, loaded: true }); })
            .catch(() => {
              // Not running is the ordinary case, not an error to shout about:
              // the panel just says how to start it.
              if (alive) setState({ runs: [], error: 'offline', loaded: true });
            });
        };
        tick();
        const id = setInterval(tick, POLL_MS);
        return () => { alive = false; clearInterval(id); };
      }, []);
      return state;
    }

    const S = {
      root: { position: 'relative' },
      trigger: {
        minHeight: 28, cursor: 'pointer', background: 'none', border: 0, borderRadius: 6,
        display: 'inline-flex', alignItems: 'center', gap: 4, padding: '3px 6px',
        fontSize: 12, lineHeight: '18px', color: 'var(--dsw-alias-label-tertiary, #6b7280)',
      },
      dot: (live) => ({
        width: 6, height: 6, borderRadius: 3,
        background: live ? 'var(--dsw-alias-label-accent, #1d4ed8)'
                         : 'var(--dsw-alias-label-tertiary, #9ca3af)',
      }),
      menu: {
        position: 'absolute', top: 'calc(100% + 5px)', left: 0, zIndex: 100,
        width: 380, maxWidth: 'calc(100vw - 32px)', maxHeight: 420, overflow: 'auto',
        padding: 8, borderRadius: 12, background: 'var(--dsw-specific-menu, #fff)',
        boxShadow: 'var(--dsw-elevation-prominent, 0 8px 24px rgba(0,0,0,.18))',
        color: 'var(--dsw-alias-label-primary, #111)', fontSize: 12,
      },
      row: { display: 'flex', gap: 8, alignItems: 'baseline', padding: '4px 2px' },
      id: { fontFamily: 'var(--dsw-font-mono, ui-monospace, monospace)', fontSize: 11 },
      dim: { color: 'var(--dsw-alias-label-tertiary, #6b7280)' },
      num: { fontVariantNumeric: 'tabular-nums' },
      target: {
        color: 'var(--dsw-alias-label-tertiary, #6b7280)', flex: 1, minWidth: 0,
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      },
    };

    function Row(props) {
      const r = props.run;
      const rounds = r.rounds ? r.round + '/' + r.rounds : String(r.round);
      const best = (r.best_reward === null || r.best_reward === undefined)
        ? '-' : r.best_reward.toFixed(3);
      return React.createElement('div', { style: S.row },
        React.createElement('span', { style: S.dot(LIVE[r.state] === 1) }),
        React.createElement('span', { style: S.id }, r.run_id),
        React.createElement('span', { style: S.dim }, r.state),
        React.createElement('span', { style: S.num }, rounds),
        React.createElement('span', { style: S.num }, best),
        React.createElement('span', { style: S.target, title: r.target || '' },
          (r.kind || '?') + ': ' + (r.target || '?')));
    }

    /** Header action: how many runs are live, and the list behind a click. */
    function AgentDescentRuns() {
      const [open, setOpen] = React.useState(false);
      const state = useRuns();
      const live = state.runs.filter((r) => LIVE[r.state] === 1).length;

      const body = state.error
        ? React.createElement('div', { style: S.dim },
            'No run panel. Start it with: agentdescent serve')
        : state.runs.length === 0
          ? React.createElement('div', { style: S.dim }, 'No runs yet.')
          : state.runs.slice(0, 40).map((r) =>
              React.createElement(Row, { key: r.run_id, run: r }));

      return React.createElement('div', { style: S.root },
        React.createElement('button', {
          type: 'button', style: S.trigger, onClick: () => setOpen(!open),
          title: 'AgentDescent runs',
        },
          React.createElement('span', { style: S.dot(live > 0) }),
          'evolve',
          live > 0 ? React.createElement('span', { style: S.num }, ' ' + live) : null),
        open ? React.createElement('div', { style: S.menu }, body) : null);
    }

    const inject = ['slots'];

    function apply(ctx) {
      ctx.slots.inject('conversation.session.header.actions', () => ctx.slots.register({
        name: 'conversation.session.header.actions',
        id: 'agentdescent-runs',
        order: 30,
      }, AgentDescentRuns));
    }

    exports.apply = apply;
    exports.inject = inject;
    exports.AgentDescentRuns = AgentDescentRuns;
    return module.exports;
  },
});
"""

#: Where the panel the client frame reads is served from.
#: :func:`~agentdescent.runstore.serve_http` defaults to the same port.
DSH_PANEL_URL = "http://127.0.0.1:8787/"


def dsh_client_source(panel_url: str = DSH_PANEL_URL) -> str:
    """The client bundle, with the panel URL baked in."""
    return DSH_CLIENT_JS.replace("@@PANEL_URL@@", json.dumps(panel_url))


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
        # `./client` is how dsh finds the browser half; `dsh.client.platform`
        # is what marks it as a web contribution (see
        # @deepseek-ai/dsh-client-ui-jobs, whose shape this follows).
        "exports": {
            ".": {"default": "./lib/index.js"},
            "./client": {"default": "./lib/client.js"},
            "./package.json": "./package.json",
        },
        "files": ["lib/index.js", "lib/client.js", "cordis.patch.yml", "README.md"],
        # Without `dsh.bundle` the package installs as a plain dependency and
        # its patch never applies -- `dsh plugin add` warns "declares no
        # dsh.bundle ... not a profile layer" and the plugin is inert. This is
        # what dsh-base / dsh-web-app / dsh-headless declare.
        "dsh": {
            "bundle": {"patch": "./cordis.patch.yml"},
            "client": {"platform": "web"},
        },
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
    env = "          " + _dsh_env_expression()
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
    w.write(os.path.join(dest, "lib", "client.js"), dsh_client_source())
    w.write(os.path.join(dest, "cordis.patch.yml"), dsh_plugin_patch())
    w.write(os.path.join(dest, "README.md"), DSH_PLUGIN_README)
    return w.lines


#: The binary each host is driven through, by ``install`` target.
HOST_CLI = {"dsh": "dsh", "claude-code": "claude", "codex": "codex", "opencode": "opencode"}

HOSTS: Dict[str, Callable[[str, _Writer], None]] = {
    "dsh": install_dsh,
    "claude-code": install_claude_code,
    "codex": install_codex,
    "opencode": install_opencode,
}


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def _skill_state(path: str) -> str:
    """``ok`` / ``stale`` / ``missing`` for one installed ``SKILL.md``.

    Compared byte for byte with the file this package would write now, so a
    ``pip install -U`` that changed the procedure shows up as ``stale``."""
    have = _read(path)
    if have is None:
        return "missing"
    return "ok" if have == skill_text() else "stale"


def _opencode_mcp_state(root: str) -> str:
    """The MCP entry in OpenCode's config: ``opencode.jsonc`` (what ``install``
    writes) or ``opencode.json`` (what a user may keep by hand).

    ``unknown`` when neither file holds the entry and one of them cannot be
    parsed -- JSONC allows comments and ``json`` cannot read them, so the entry
    might be there and guessing either way would be wrong."""
    unreadable = False
    for name in ("opencode.jsonc", "opencode.json"):
        raw = _read(os.path.join(root, name))
        if raw is None:
            continue
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            unreadable = True
            continue
        if isinstance(data, dict) and (data.get("mcp") or {}).get("agentdescent"):
            return "ok"
    return "unknown" if unreadable else "missing"


def _claude_mcp_state(path: str) -> str:
    raw = _read(path)
    if raw is None:
        return "missing"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "missing"          # a manifest the host cannot read starts no server
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    return "ok" if isinstance(servers, dict) and servers.get("agentdescent") else "missing"


def host_status(host: str, *, home: Optional[str] = None) -> Dict[str, str]:
    """What ``install(host)`` left behind, read-only: is the host wired up, and
    is what it holds still what this package would write?

    ``skill`` is ``ok`` / ``stale`` / ``missing``; ``mcp`` is ``ok`` /
    ``missing`` / ``unknown`` (an OpenCode config with comments cannot be read
    safely; it is also what ``doctor`` reports if reading a host's files raises).
    Two limits: a launcher pinned to an interpreter that has since been deleted
    still reads ``ok`` (only presence is checked), and an install made through a
    host's own marketplace (``/plugin install``, ``codex plugin add``) leaves no
    files here, so it reads ``missing``. ``state`` summarises them: ``ok``, ``stale`` (the skill is
    behind the package), ``partial`` (one half is there) or ``missing``. Paths
    follow ``install`` exactly, including ``DSH_HOME`` / ``CODEX_HOME`` /
    ``XDG_CONFIG_HOME``. Codex can also be wired through ``codex plugin``, which
    leaves nothing here, so its ``missing`` says where else to look.
    """
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}; choose from {sorted(HOSTS)}")
    home = os.path.expanduser(home or "~")
    if host == "dsh":
        root = os.environ.get("DSH_HOME") or os.path.join(home, ".dsh")
        skill = os.path.join(root, "skills", "agentdescent", "SKILL.md")
        patch = _read(os.path.join(root, "cordis.patch.yml")) or ""
        # Both markers: an interrupted write can leave only the start.
        mcp = "ok" if DSH_BLOCK_START in patch and DSH_BLOCK_END in patch else "missing"
    elif host == "claude-code":
        root = os.path.join(home, ".agentdescent", "plugins", "claude-code")
        skill = os.path.join(root, "skills", "agentdescent", "SKILL.md")
        mcp = _claude_mcp_state(os.path.join(root, ".mcp.json"))
    elif host == "codex":
        root = os.environ.get("CODEX_HOME") or os.path.join(home, ".codex")
        skill = os.path.join(root, "skills", "agentdescent", "SKILL.md")
        cfg = _read(os.path.join(root, "config.toml")) or ""
        # A live table header, not a substring: a commented-out one is not wired.
        mcp = ("ok" if re.search(r"^\s*\[mcp_servers\.agentdescent\]", cfg, re.M)
               else "missing")
        # Our own block from before #404 has no `env_vars`: Codex starts that
        # server without the provider keys. Re-running `install` repairs it.
        legacy_block = mcp == "ok" and _is_legacy_codex_block(cfg)
    else:
        root = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"),
                            "opencode")
        skill = os.path.join(root, "skill", "agentdescent", "SKILL.md")
        mcp = _opencode_mcp_state(root)
    skill_state = _skill_state(skill)
    if skill_state == "missing" and mcp in ("missing", "unknown"):
        # Nothing of ours is there. `unknown` only says a config this user wrote
        # could not be read, which is no evidence that we were ever installed.
        state = "missing"
    elif skill_state == "stale" or (host == "codex" and legacy_block):
        state = "stale"
    elif skill_state == "ok" and mcp in ("ok", "unknown"):
        state = "ok"
    else:
        state = "partial"
    return {"state": state, "skill": skill_state, "mcp": mcp, "path": skill}


def mcp_sdk_missing() -> bool:
    """Whether ``agentdescent mcp`` would fail for want of the protocol SDK."""
    try:
        __import__("mcp")
        return False
    except ImportError:
        return True


def _runs_this_interpreter(script: str) -> bool:
    """Whether the console script at ``script`` launches ``sys.executable``.

    ``~/.local/bin`` or ``/usr/local/bin`` can hold the script of a *different*
    interpreter's install; pinning it would start the wrong environment. A pip
    script names its interpreter in the shebang (or in the ``exec`` line of the
    ``/bin/sh`` trampoline pip writes for long paths); a Windows launcher embeds
    the path in the executable. Look for the interpreter path, as given or with
    symlinks resolved, in the file.
    """
    try:
        with open(script, "rb") as fh:
            blob = fh.read(1 << 20)
    except OSError:
        return False
    exe = os.path.abspath(sys.executable)
    return any(p.encode("utf-8", "surrogateescape") in blob
               for p in {exe, os.path.realpath(exe)})


def _console_script() -> Optional[str]:
    """Absolute path of the ``agentdescent`` console script of *this* interpreter.

    Looked up beside ``sys.executable`` first (a venv's ``bin``/``Scripts``), then
    in the interpreter's default and per-user script directories (including
    macOS framework builds' own user scheme), and accepted only if the script
    runs ``sys.executable`` (see :func:`_runs_this_interpreter`).
    """
    dirs = [os.path.dirname(os.path.abspath(sys.executable)), sysconfig.get_path("scripts")]
    schemes = ["nt_user" if os.name == "nt" else "posix_user", "osx_framework_user"]
    preferred = getattr(sysconfig, "get_preferred_scheme", None)
    if preferred is not None:
        try:
            schemes.insert(0, preferred("user"))
        except (KeyError, ValueError):
            pass
    for scheme in schemes:
        try:
            dirs.append(sysconfig.get_path("scripts", scheme))
        except (KeyError, ValueError):
            pass
    for d in dirs:
        for name in ("agentdescent", "agentdescent.exe"):
            path = os.path.join(d or "", name)
            if (d and os.path.isfile(path) and os.access(path, os.X_OK)
                    and _runs_this_interpreter(path)):
                return os.path.abspath(path)
    return None


def mcp_launcher() -> List[str]:
    """The machine-local launcher prefix ``install`` pins into every host config.

    ``python -m agentdescent.cli`` puts the *host's* current directory first on
    ``sys.path``, so a directory called ``agentdescent/`` there (typically the
    parent of a source checkout) shadows the installed package and the server dies
    on import (#394). So the launcher must not depend on the cwd: on Python >= 3.11
    ``-P`` keeps it off ``sys.path`` (``-I`` would also drop the user site, which
    breaks ``pip install --user``); older interpreters have no ``-P``, so use the
    console script, whose ``sys.path[0]`` is its own directory, and only if there
    is none fall back to the plain ``-m`` form.
    """
    python = os.path.abspath(sys.executable)
    if sys.version_info >= (3, 11):
        return [python, "-P", "-m", "agentdescent.cli"]
    script = _console_script()
    return [script] if script else [python, "-m", "agentdescent.cli"]


def install(host: str, *, dry_run: bool = False, home: Optional[str] = None) -> List[str]:
    """Wire ``host`` up. Returns the lines to print: what was written or kept."""
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}; choose from {sorted(HOSTS)}")
    # Bind machine-local configuration to this installation, not the host's
    # future PATH. Do not resolve symlinks: a venv's Python must stay in its
    # venv. Portable render_* templates deliberately keep the bare command.
    w = _Writer(dry_run, mcp_launcher())
    HOSTS[host](os.path.expanduser(home or "~"), w)
    if "-P" not in w.launcher and len(w.launcher) > 1:
        # Python < 3.11 (no `-P`) and no console script for this interpreter: the
        # plain `-m` form is all that is left, and it still honours the host's cwd.
        w.note("WARNING: this Python has no -P and no `agentdescent` console script was "
               "found for it, so the launcher written above is `python -m agentdescent.cli` "
               "and a host started in a directory containing an `agentdescent/` folder "
               "will fail to connect (#394). Use Python >= 3.11, or install the package "
               "into an environment that has the `agentdescent` script and re-run install")
    # Imported here, not at module scope: `cli` reaches into this module from
    # `cmd_install`, and the reason for a missing SDK belongs in one place.
    from ..cli import mcp_unavailable

    why = mcp_unavailable()
    if why:
        # Every manifest written above tells the host to run `agentdescent mcp`.
        # Without the SDK that subprocess exits immediately and the host reports
        # only "CONNECTION_CLOSED" -- so say it here, where the user is looking,
        # rather than let them discover it as a silent missing tool later. On
        # 3.9 the fix is not a pip line, so the reason comes from `cli` rather
        # than being spelled out again here.
        w.note("WARNING: the server this just wired up cannot start -- " + why)
    return w.lines


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


class _Eraser:
    """What ``uninstall`` does to disk, with ``--dry-run`` and one report line each.

    It removes only what ``install`` wrote and only when it can tell that it is
    still ours; anything the user has changed is left in place and said so.
    """

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.lines: List[str] = []
        self.touched = False

    def remove_file(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        self.touched = True
        if not self.dry_run:
            os.remove(path)
        self.lines.append(f"{'would remove' if self.dry_run else 'removed'} {path}")

    def prune_dirs(self, start: str, stop: str) -> None:
        """Remove ``start`` and its parents while they are empty, never past ``stop``.

        ``stop`` itself is kept: it is the host's own directory (``~/.dsh``,
        ``~/.codex``), which is not ours to delete. Under ``--dry-run`` nothing
        is removed, so the loop only reports files that would be left alone."""
        stop = os.path.abspath(stop)
        cur = os.path.abspath(start)
        while cur != stop and cur.startswith(stop + os.sep):
            try:
                if self.dry_run or os.listdir(cur):
                    return
                os.rmdir(cur)
            except OSError:
                return
            cur = os.path.dirname(cur)

    def rewrite(self, path: str, text: str, what: str) -> None:
        self.touched = True
        if not self.dry_run:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        self.lines.append(f"{'would update' if self.dry_run else 'updated'} {path} ({what})")

    def kept(self, path: str, why: str) -> None:
        self.lines.append(f"kept {path} ({why})")

    def note(self, text: str) -> None:
        self.lines.append(text)


def _uninstall_dsh(home: str, e: _Eraser) -> None:
    root = os.environ.get("DSH_HOME") or os.path.join(home, ".dsh")
    skill_dir = os.path.join(root, "skills", "agentdescent")
    for name in ("SKILL.md", "hooks.json"):
        e.remove_file(os.path.join(skill_dir, name))
    e.prune_dirs(skill_dir, root)
    path = os.path.join(root, "cordis.patch.yml")
    current = _read(path)
    if current is None or DSH_BLOCK_START not in current:
        return
    head = current[:current.index(DSH_BLOCK_START)]
    rest = current[current.index(DSH_BLOCK_START):]
    if DSH_BLOCK_END in rest:
        # The block is owned by its two markers and removed whole, whatever was
        # edited inside it -- the same ownership `sync_block` applies on install.
        tail = rest[rest.index(DSH_BLOCK_END) + len(DSH_BLOCK_END):]
    else:
        # A block from before the end marker existed. `sync_block` reads it as
        # running to the end of the file, but that was only true when it was
        # written: rows the user appended since are theirs. The block's last
        # line was always the hooks row's `configPath: .../agentdescent/hooks.json`,
        # so end it there; if that line is not found, say so rather than guess.
        lines = rest.split("\n")
        last = next((i for i, line in enumerate(lines)
                     if line.strip().startswith("configPath:")
                     and "agentdescent/hooks.json" in line), None)
        if last is None:
            e.kept(path, "an old agentdescent block with no end marker whose last row "
                         "was not recognised; delete it from the marker line yourself")
            return
        tail = "\n".join(lines[last + 1:])
    remaining = head + tail.lstrip("\n")
    if not remaining.strip():
        e.remove_file(path)
    else:
        e.rewrite(path, remaining, "agentdescent mcp-client + hooks entries removed")


def _uninstall_claude_code(home: str, e: _Eraser) -> None:
    dest = os.path.join(home, ".agentdescent", "plugins", "claude-code")
    for rel in ((".claude-plugin", "plugin.json"), ("skills", "agentdescent", "SKILL.md"),
                ("commands", "evolve.md"), (".mcp.json",), ("hooks", "hooks.json")):
        path = os.path.join(dest, *rel)
        e.remove_file(path)
        e.prune_dirs(os.path.dirname(path), os.path.join(home, ".agentdescent"))
    # `~/.agentdescent` also holds the run store; it is never ours to remove.
    if os.path.isdir(dest) and not e.dry_run and os.listdir(dest):
        e.kept(dest, "holds files this install did not write")


def _codex_block_to_remove(current: str) -> Optional[str]:
    """The exact text ``install codex`` appended, wherever it now sits in the file.

    ``_stock_codex_block`` only recognises it as the *last* thing in
    ``config.toml``, which is what install's repair needs and is wrong here:
    Codex itself appends tables after it (``[projects."..."]`` when a folder is
    trusted). So: our marker line, the table header, ``command`` and ``args``
    byte for byte as ``codex_config_block`` writes them for a stock launcher, and
    then the end of the file or the next table header (blank lines and comments
    aside). A further key under our table means the user edited it.
    """
    marker = "\n# --- agentdescent (written by `agentdescent install codex`) ---\n"
    if current.count(marker) != 1 or current.count("[mcp_servers.agentdescent]") != 1:
        return None
    start = current.index(marker)
    lines = current[start:].split("\n")       # "", marker, header, command, args, ...
    if (len(lines) < 5 or lines[2] != "[mcp_servers.agentdescent]"
            or not lines[3].startswith("command = ") or not lines[4].startswith("args = ")):
        return None
    try:
        command = json.loads(lines[3][len("command = "):])
        args = json.loads(lines[4][len("args = "):])
    except (ValueError, TypeError):
        return None
    if not isinstance(args, list) or not _stock_mcp_argv([command, *args]):
        return None
    launcher = [command, *args[:-len(MCP_ARGS)]]
    # Current form first, then the one written before #404 (no `env_vars`).
    block = next((b for b in (codex_config_block(launcher), _legacy_codex_block(launcher))
                  if current[start:].startswith(b)), None)
    if block is None:
        return None
    after = current[start + len(block):]
    for line in after.split("\n"):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if not text.startswith("["):
            return None                          # another key of our table
        break
    return block


def _is_legacy_codex_block(current: str) -> bool:
    """Whether ``config.toml`` holds the untouched pre-#404 block (no ``env_vars``)."""
    block = _codex_block_to_remove(current)
    return block is not None and "\nenv_vars = " not in block


def _uninstall_codex(home: str, e: _Eraser) -> None:
    root = os.environ.get("CODEX_HOME") or os.path.join(home, ".codex")
    skill_dir = os.path.join(root, "skills", "agentdescent")
    e.remove_file(os.path.join(skill_dir, "SKILL.md"))
    e.prune_dirs(skill_dir, root)
    path = os.path.join(root, "config.toml")
    current = _read(path)
    if current is None or "[mcp_servers.agentdescent]" not in current:
        return
    block = _codex_block_to_remove(current)
    if block is None:
        e.kept(path, "its [mcp_servers.agentdescent] table is not the one `install` wrote "
                     "(edited, or added by hand); remove it yourself if you want it gone")
        return
    remaining = current.replace(block, "", 1)
    if not remaining.strip():
        e.remove_file(path)
    else:
        e.rewrite(path, remaining, "[mcp_servers.agentdescent] removed")


def _uninstall_opencode(home: str, e: _Eraser) -> None:
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    root = os.path.join(config_home, "opencode")
    skill_dir = os.path.join(root, "skill", "agentdescent")
    e.remove_file(os.path.join(skill_dir, "SKILL.md"))
    e.prune_dirs(skill_dir, root)
    path = os.path.join(root, "opencode.jsonc")
    raw = _read(path)
    if raw is None or "agentdescent" not in raw:
        return
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        # JSONC allows comments, which json cannot read; editing it blind could
        # corrupt the file, so say what to remove instead of guessing.
        e.kept(path, "could not parse it (comments?); remove the \"agentdescent\" "
                     "entry under \"mcp\" yourself")
        return
    servers = data.get("mcp") if isinstance(data, dict) else None
    existing = servers.get("agentdescent") if isinstance(servers, dict) else None
    if existing is None:
        return
    # Pinned launcher only. The bare `["agentdescent", "mcp"]` shape is also what
    # `opencode mcp add agentdescent -- agentdescent mcp` writes, and with no
    # ownership marker it cannot be told from the user's own entry, so it stays.
    stock = (isinstance(existing, dict) and set(existing) == {"type", "command"}
             and existing["type"] == "local"
             and existing["command"] != [MCP_COMMAND, *MCP_ARGS]
             and _stock_mcp_argv(existing["command"]))
    if not stock:
        e.kept(path, "its mcp entry is not the one `install` wrote (edited, or added by "
                     "hand); remove it yourself if you want it gone")
        return
    rest = {k: v for k, v in servers.items() if k != "agentdescent"}
    out = {k: v for k, v in data.items() if k != "mcp"}
    if rest:
        out["mcp"] = rest
    if out == {"$schema": "https://opencode.ai/config.json"}:
        e.remove_file(path)        # what `install` creates, with nothing else in it
    else:
        e.rewrite(path, json.dumps(out, indent=2) + "\n", "mcp entry removed")


_UNINSTALL: Dict[str, Callable[[str, _Eraser], None]] = {
    "dsh": _uninstall_dsh,
    "claude-code": _uninstall_claude_code,
    "codex": _uninstall_codex,
    "opencode": _uninstall_opencode,
}


def uninstall(host: str, *, dry_run: bool = False, home: Optional[str] = None) -> List[str]:
    """Undo ``install(host)``. Returns the lines to print: what was removed or kept.

    Removes the files ``install`` wrote and the block or entry it added to the
    host's config. A config entry that is no longer exactly what ``install``
    writes -- you edited the launcher, or wrote it yourself -- is kept and
    reported, and a config file with comments (OpenCode's ``opencode.jsonc``) is
    never rewritten. The run store under ``~/.agentdescent`` is untouched. An
    install made through the host's own marketplace is removed with that host's
    plugin command. ``dry_run`` reports without changing anything.
    """
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}; choose from {sorted(HOSTS)}")
    e = _Eraser(dry_run)
    _UNINSTALL[host](os.path.expanduser(home or "~"), e)
    if not e.lines:
        e.note(f"nothing to remove: agentdescent is not installed for {host} "
               "(files from the host's own marketplace are removed with its plugin command)")
    return e.lines
