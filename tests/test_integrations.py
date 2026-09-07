"""`agentdescent install <host>`: manifests into a fake home, and no drift."""

import json
import os
import shutil
import subprocess

import pytest

from agentdescent import cli
from agentdescent.integrations import (
    DSH_FORWARDED_KEYS, HOSTS, install, marketplace_manifest, render_claude_plugin, skill_text,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_the_skill_teaches_the_procedure_and_the_guardrails():
    text = skill_text()
    assert text.startswith("---\nname: agentdescent\n")
    for word in ("doctor", "plan", "start", "status", "show", "apply", "nested", "cases.jsonl"):
        assert word in text, word
    assert "Never evolve against data the user has not seen" in text
    assert "Ask before `apply`" in text


def test_the_skill_names_every_tool_kind_and_verb_that_exists():
    """A skill that omits a tool is a capability the host model never uses.

    `cancel` and `resume` were missing here, so a model watching a run that was
    going badly had no way to know it could stop it."""
    import re

    from agentdescent import cli
    from agentdescent.evolvespec import KINDS, SHORT_REFS
    from agentdescent.mcp import TOOL_DESCRIPTIONS

    text = skill_text()
    missing = [t for t in TOOL_DESCRIPTIONS if not re.search(rf"\b{t}\b", text)]
    assert not missing, f"tools the skill never mentions: {missing}"
    assert not [k for k in KINDS if f"`{k}`" not in text], "every kind must be named"
    # nothing the skill names may be made up
    verbs = set(re.findall(r"^agentdescent (\w+)", text, re.M))
    real = set(next(a for a in cli.build_parser()._actions if a.dest == "cmd").choices)
    assert verbs <= real, f"skill shows CLI verbs that do not exist: {sorted(verbs - real)}"
    agents = set(re.findall(r"`(claude_code|codex|dsh|opencode|openai_compatible|claude)`", text))
    assert agents <= set(SHORT_REFS), f"unknown agent short names: {sorted(agents - set(SHORT_REFS))}"


def test_install_dsh_writes_skill_hooks_and_patch(tmp_path, monkeypatch):
    monkeypatch.delenv("DSH_HOME", raising=False)
    lines = install("dsh", home=str(tmp_path))
    skill = tmp_path / ".dsh" / "skills" / "agentdescent" / "SKILL.md"
    assert skill.exists() and (skill.parent / "hooks.json").exists()
    patch = _read(tmp_path / ".dsh" / "cordis.patch.yml")
    assert "name: '@deepseek-ai/dsh-mcp-client'" in patch
    assert "serverName: agentdescent" in patch and "args: [\"mcp\"]" in patch
    for key in DSH_FORWARDED_KEYS:
        # Each key is forwarded past dsh's scrubbing -- but as part of one
        # filtered !!js expression, never as its own entry: unset, it would be
        # `undefined`, and dsh refuses the whole plugin tree over it.
        assert f"{key}: process.env.{key}" in patch
    assert "dsh-hooks-claude-code" in patch
    # A dsh patch file OVERRIDES rows by id; new rows must be under `insert:` or
    # dsh warns `patch: entry "mcp-agentdescent" not found` and composes without
    # them (verified against dsh 0.1.2-rc.1 with --dump-config).
    assert "- insert:" in patch
    assert patch.index("- insert:") < patch.index("id: mcp-agentdescent")
    assert any("dump-config" in l for l in lines)
    # idempotent: a second install keeps the patch file as it is
    before = patch
    lines = install("dsh", home=str(tmp_path))
    assert _read(tmp_path / ".dsh" / "cordis.patch.yml") == before
    # "up to date" rather than "present": the check is now on the block's
    # *content*, so a stale one is repaired instead of kept (see the repair test).
    assert any("already up to date" in l for l in lines)


def test_install_dsh_honours_dsh_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "custom"))
    install("dsh", home=str(tmp_path))
    assert (tmp_path / "custom" / "skills" / "agentdescent" / "SKILL.md").exists()


def test_install_codex_writes_skill_and_config(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    cfg = tmp_path / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('model = "x"\n')
    install("codex", home=str(tmp_path))
    text = _read(cfg)
    assert text.startswith('model = "x"\n') and "[mcp_servers.agentdescent]" in text
    assert (tmp_path / ".codex" / "skills" / "agentdescent" / "SKILL.md").exists()
    install("codex", home=str(tmp_path))
    assert text == _read(cfg)


def test_install_claude_code_renders_a_loadable_plugin_dir(tmp_path):
    lines = install("claude-code", home=str(tmp_path))
    dest = tmp_path / ".agentdescent" / "plugins" / "claude-code"
    manifest = json.loads(_read(dest / ".claude-plugin" / "plugin.json"))
    assert manifest["name"] == "agentdescent" and manifest["version"]
    assert json.loads(_read(dest / ".mcp.json"))["mcpServers"]["agentdescent"]["args"] == ["mcp"]
    hooks = json.loads(_read(dest / "hooks" / "hooks.json"))
    assert "SessionStart" in hooks["hooks"]
    assert (dest / "skills" / "agentdescent" / "SKILL.md").exists()
    assert "$ARGUMENTS" in _read(dest / "commands" / "evolve.md")
    assert any("--plugin-dir" in l for l in lines)


def test_install_opencode_writes_the_shape_opencode_itself_writes(tmp_path, monkeypatch):
    """`opencode mcp add` writes {type: local, command: [...]} into opencode.jsonc."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    install("opencode", home=str(tmp_path))
    root = tmp_path / "cfg" / "opencode"
    assert (root / "skill" / "agentdescent" / "SKILL.md").exists()
    cfg = json.loads(_read(root / "opencode.jsonc"))
    assert cfg["mcp"]["agentdescent"] == {"type": "local",
                                          "command": ["agentdescent", "mcp"]}
    assert cfg["$schema"].startswith("https://opencode.ai")
    # merging keeps what was already there, and is idempotent
    cfg["model"] = "anthropic/claude"
    cfg["mcp"]["other"] = {"type": "local", "command": ["x"]}
    (root / "opencode.jsonc").write_text(json.dumps(cfg))
    install("opencode", home=str(tmp_path))
    again = json.loads(_read(root / "opencode.jsonc"))
    assert again["model"] == "anthropic/claude" and "other" in again["mcp"]


def test_install_opencode_does_not_corrupt_a_commented_config(tmp_path, monkeypatch):
    """opencode.jsonc may hold comments, which json cannot parse: say what to add."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    root = tmp_path / "cfg" / "opencode"
    root.mkdir(parents=True)
    original = '{\n  // my notes\n  "model": "x"\n}\n'
    (root / "opencode.jsonc").write_text(original)
    lines = install("opencode", home=str(tmp_path))
    assert _read(root / "opencode.jsonc") == original      # untouched
    assert any(l.startswith("NOTE:") and "mcp" in l for l in lines), lines


def test_dry_run_writes_nothing(tmp_path):
    for host in HOSTS:
        lines = install(host, dry_run=True, home=str(tmp_path))
        assert lines and all(l.startswith(("would", "kept", "forwarded", "verify", "load",
                                           "or,", "Codex", "NOTE", "WARNING"))
                             for l in lines), lines
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())


def test_unknown_host_is_refused():
    with pytest.raises(ValueError, match="unknown host"):
        install("emacs")


def test_cli_install_dry_run(tmp_path):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(["install", "dsh", "--dry-run", "--home", str(tmp_path)])
    assert code == 0 and "would write" in buf.getvalue()


def test_checked_in_claude_plugin_matches_the_package(tmp_path):
    """`integrations/claude-code` is rendered from the package; it must not drift."""
    render_claude_plugin(str(tmp_path))
    checked_in = os.path.join(ROOT, "integrations", "claude-code")
    for dirpath, _, files in os.walk(tmp_path):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), tmp_path)
            assert _read(os.path.join(checked_in, rel)) == _read(os.path.join(dirpath, f)), rel
    expected = json.dumps(marketplace_manifest(), indent=2) + "\n"
    assert _read(os.path.join(ROOT, ".claude-plugin", "marketplace.json")) == expected


def test_package_data_ships_the_shared_files():
    cfg = _read(os.path.join(ROOT, "pyproject.toml"))
    assert 'agentdescent = ["integrations/*.md", "integrations/*.json"]' in cfg


def test_install_warns_when_the_mcp_sdk_is_missing(tmp_path, monkeypatch):
    """Every manifest tells the host to run `agentdescent mcp`; without the SDK
    that subprocess dies and the host reports only "CONNECTION_CLOSED"."""
    import sys

    import agentdescent.integrations as integrations
    from agentdescent.cli import MCP_MIN_PYTHON

    monkeypatch.setattr(integrations, "mcp_sdk_missing", lambda: True)
    lines = install("claude-code", dry_run=True, home=str(tmp_path))
    assert any("agentdescent[mcp]" in l and l.startswith("WARNING") for l in lines), lines

    monkeypatch.setattr(integrations, "mcp_sdk_missing", lambda: False)
    quiet = install("dsh", dry_run=True, home=str(tmp_path))
    if sys.version_info >= MCP_MIN_PYTHON:
        assert not any(l.startswith("WARNING") for l in quiet), quiet
    else:
        # Below 3.10 the stub cannot make the server runnable: `mcp_unavailable`
        # checks the interpreter before it asks whether the package is here, and
        # on 3.9 the extra installs nothing whatever the stub says. The warning
        # is right; asserting silence here would be asserting a lie.
        assert any("Python >= " in l for l in quiet if l.startswith("WARNING")), quiet


def test_agentdescent_mcp_without_the_sdk_says_how_to_get_it(monkeypatch, capsys):
    """A traceback here is invisible: the host shows the user a closed pipe."""
    import builtins
    import sys

    from agentdescent import cli

    real = builtins.__import__

    def fake(name, *a, **k):
        # level is the 5th positional arg; a relative `from .mcp import ...`
        # arrives as name="mcp" with level=1 and must not be intercepted.
        level = k.get("level", a[3] if len(a) > 3 else 0)
        if level == 0 and (name == "mcp" or name.startswith("mcp.")):
            raise ImportError("no mcp")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    code = cli.main(["mcp"])
    assert code == 3
    err = capsys.readouterr().err
    assert "Traceback" not in err
    # What it should say depends on the interpreter, and only one of the two is
    # ever true advice: below 3.10 the extra installs nothing, so naming it as
    # the fix would send the reader in a circle.
    if sys.version_info >= cli.MCP_MIN_PYTHON:
        assert 'pip install "agentdescent[mcp]"' in err, err
    else:
        assert "needs Python >= " in err, err


# ---------------------------------------------------------------------------
# the native DeepSeek Harness plugin
# ---------------------------------------------------------------------------


def _dsh_skill_entry():
    """The installed `@deepseek-ai/dsh-skill` entry module, or None.

    Walks up from whatever `dsh` resolves to until the package root, rather than
    guessing a depth: the launcher is `.../@deepseek-ai/dsh/lib/bin.js` today and
    the layout is not ours to pin."""
    dsh = shutil.which("dsh")
    if not dsh:
        return None
    node = os.path.realpath(dsh)
    for _ in range(6):
        node = os.path.dirname(node)
        entry = os.path.join(node, "node_modules", "@deepseek-ai", "dsh-skill",
                             "lib", "index.js")
        if os.path.exists(entry):
            return entry
    return None


def test_dsh_plugin_package_declares_the_bundle_field(tmp_path):
    """Without `dsh.bundle` the package installs but its patch never applies.

    `dsh plugin add` warns "declares no dsh.bundle ... not a profile layer" and
    the plugin is inert -- verified against dsh 0.1.2-rc.1."""
    from agentdescent.integrations import dsh_plugin_package, render_dsh_plugin

    assert dsh_plugin_package()["dsh"]["bundle"] == {"patch": "./cordis.patch.yml"}
    render_dsh_plugin(str(tmp_path))
    pkg = json.loads(_read(tmp_path / "package.json"))
    assert pkg["type"] == "module" and pkg["main"] == "lib/index.js"
    assert "dsh-plugin" in pkg["keywords"]              # the discovery topic
    assert "@deepseek-ai/dsh-skill" in pkg["peerDependencies"]
    assert "dependencies" not in pkg                   # a plugin shares the host's cordis


def test_dsh_plugin_patch_inserts_itself_and_the_mcp_row(tmp_path):
    from agentdescent.integrations import render_dsh_plugin

    render_dsh_plugin(str(tmp_path))
    patch = _read(tmp_path / "cordis.patch.yml")
    assert patch.lstrip().startswith("#") and "- insert:" in patch
    assert "id: dsh-agentdescent" in patch and "id: mcp-agentdescent" in patch
    for key in DSH_FORWARDED_KEYS:
        # In the shared filtered expression, not one entry per key -- an
        # unset key would be `undefined` and dsh would refuse the tree.
        assert f"{key}: process.env.{key}" in patch


def test_dsh_plugin_embeds_the_shared_skill_without_frontmatter(tmp_path):
    from agentdescent.integrations import _skill_body, _skill_description, render_dsh_plugin

    render_dsh_plugin(str(tmp_path))
    src = _read(tmp_path / "lib" / "index.js")
    assert json.dumps(_skill_body()) in src           # the same text every host gets
    assert json.dumps(_skill_description()) in src
    assert not _skill_body().startswith("---")        # the registry supplies the metadata
    assert "export const inject = ['skills'];" in src


def test_checked_in_dsh_plugin_matches_the_package(tmp_path):
    """`integrations/dsh-agentdescent` is rendered; it must not drift."""
    from agentdescent.integrations import render_dsh_plugin

    render_dsh_plugin(str(tmp_path))
    checked_in = os.path.join(ROOT, "integrations", "dsh-agentdescent")
    for dirpath, _, files in os.walk(tmp_path):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), tmp_path)
            assert _read(os.path.join(checked_in, rel)) == _read(os.path.join(dirpath, f)), rel


def test_the_dsh_plugin_registers_a_valid_skill_through_the_real_registry(tmp_path):
    """Run apply() with dsh's own `isSkillName`, not a mock of it."""
    node = shutil.which("node")
    entry = _dsh_skill_entry()
    if not node or not entry:
        pytest.skip("needs node and an installed dsh")
    from agentdescent.integrations import render_dsh_plugin

    render_dsh_plugin(str(tmp_path))
    harness = os.path.join(ROOT, "tests", "fixtures", "verify_dsh_plugin.mjs")
    proc = subprocess.run([node, harness, str(tmp_path / "lib" / "index.js"), entry],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["name"] == "agentdescent" and out["inject"] == ["skills"]
    assert out["apply"] == "function"
    assert out["registered"]["source"] == "runtime"
    assert out["registered"]["contentLength"] > 500
    assert out["registered"]["contentHead"].startswith("# AgentDescent")
    assert out["disposed"] is True          # ctx.effect captured the disposer


def _node_module(name):
    """Resolve a node module, or None. React is not a dependency of this repo."""
    node = shutil.which("node")
    if not node:
        return None
    probe = subprocess.run(
        [node, "-e", f"try{{console.log(require.resolve({name!r}))}}catch(e){{}}"],
        capture_output=True, text=True, timeout=60,
        cwd=os.environ.get("AGENTDESCENT_NODE_PROBE") or None)
    path = probe.stdout.strip()
    return path or None


def test_the_dsh_client_bundle_registers_a_slot_and_renders(tmp_path):
    """Load the browser half the way dsh's module loader does, with real React.

    Everything but the visual result is checkable here: the loader wrapper, the
    injected slot name, the registration spec, and that the component renders.
    Set AGENTDESCENT_NODE_PROBE to a directory with react + react-dom installed
    to run it; without them it skips."""
    node = shutil.which("node")
    react = _node_module("react")
    react_dom = _node_module("react-dom/server")
    if not (node and react and react_dom):
        pytest.skip("needs node with react and react-dom resolvable "
                    "(set AGENTDESCENT_NODE_PROBE)")
    from agentdescent.integrations import render_dsh_plugin

    render_dsh_plugin(str(tmp_path))
    harness = os.path.join(ROOT, "tests", "fixtures", "verify_dsh_client.mjs")
    proc = subprocess.run(
        [node, harness, str(tmp_path / "lib" / "client.js"), react, react_dom],
        capture_output=True, text=True, timeout=120,
        cwd=os.environ.get("AGENTDESCENT_NODE_PROBE") or None)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["id"] == "dsh-agentdescent"
    assert out["inject"] == ["slots"] and out["apply"] == "function"
    assert out["injectedSlot"] == "conversation.session.header.actions"
    assert out["slot"]["id"] == "agentdescent-runs"
    assert out["rendersTrigger"] and out["htmlLength"] > 100


def test_the_dsh_client_half_is_declared_where_dsh_looks(tmp_path):
    """`exports['./client']` and `dsh.client.platform` are how dsh finds it."""
    from agentdescent.integrations import dsh_plugin_package, render_dsh_plugin

    pkg = dsh_plugin_package()
    assert pkg["exports"]["./client"] == {"default": "./lib/client.js"}
    assert pkg["dsh"]["client"] == {"platform": "web"}
    assert "lib/client.js" in pkg["files"]
    render_dsh_plugin(str(tmp_path))
    src = _read(tmp_path / "lib" / "client.js")
    assert src.startswith("window.__ModuleLoader__.load({")
    assert "conversation.session.header.actions" in src
    # the panel it reads is the loopback one `agentdescent serve` provides
    assert '"http://127.0.0.1:8787/"' in src


def test_codex_installs_the_same_plugin_as_claude_code(tmp_path):
    """Codex reads the Claude Code plugin format and marketplace.

    The plugin brings the MCP server with it, so the file route is the
    alternative rather than the only way -- verified against codex-cli
    0.153.4, which resolves .claude-plugin/marketplace.json and lists the
    plugin at integrations/claude-code."""
    codex = shutil.which("codex")
    if not codex:
        pytest.skip("needs the codex CLI")
    env = {**os.environ, "CODEX_HOME": str(tmp_path / "codex")}
    os.makedirs(env["CODEX_HOME"])

    add = subprocess.run([codex, "plugin", "marketplace", "add", ROOT],
                         capture_output=True, text=True, timeout=180, env=env)
    assert add.returncode == 0, add.stderr
    listed = subprocess.run([codex, "plugin", "list"],
                            capture_output=True, text=True, timeout=180, env=env)
    assert listed.returncode == 0, listed.stderr
    assert "agentdescent@agentdescent" in listed.stdout, listed.stdout
    assert os.path.join("integrations", "claude-code") in listed.stdout

    install_ = subprocess.run([codex, "plugin", "add", "agentdescent@agentdescent"],
                              capture_output=True, text=True, timeout=300, env=env)
    assert install_.returncode == 0, install_.stderr
    # the server the plugin declares is live without anything in mcp_servers
    cfg = _read(os.path.join(env["CODEX_HOME"], "config.toml"))
    assert "[mcp_servers.agentdescent]" not in cfg
    mcp = subprocess.run([codex, "mcp", "list"],
                         capture_output=True, text=True, timeout=180, env=env)
    assert "agentdescent" in mcp.stdout and "enabled" in mcp.stdout, mcp.stdout


def test_the_codex_installer_points_at_the_plugin_route(tmp_path):
    lines = install("codex", dry_run=True, home=str(tmp_path))
    assert any("codex plugin marketplace add" in l and "codex plugin add" in l
               for l in lines), lines


# ---------------------------------------------------------------------------
# The `[mcp]` extra on Python 3.9
# ---------------------------------------------------------------------------
#
# Reported from a real 3.9 install: `pip install "agentdescent[mcp]"` -- the
# line every install doc gives -- failed with a screen of "Ignored the following
# versions that require a different python version" and installed *nothing*,
# not even the CLI, because every published `mcp` requires >=3.10 and the extra
# had no marker. These pin both halves of the fix on an interpreter that cannot
# reproduce it.


def test_the_mcp_extra_is_gated_on_the_python_it_needs():
    """Without the marker, 3.9 users get no package at all -- not even the CLI."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        text = fh.read()
    line = [ln for ln in text.splitlines() if ln.startswith("mcp = [")]
    assert line, "the [mcp] extra disappeared from pyproject.toml"
    assert "python_version >= '3.10'" in line[0], (
        "the [mcp] extra must carry a python_version marker: every published "
        "mcp requires >=3.10 while this project supports 3.9, and an unmarked "
        "extra fails the whole install there instead of degrading to the CLI")


def test_an_old_interpreter_is_told_the_truth_not_a_pip_line(monkeypatch):
    """On 3.9 `pip install agentdescent[mcp]` is the one thing that cannot help."""
    from agentdescent import cli

    monkeypatch.setattr(cli.sys, "version_info", (3, 9, 23, "final", 0))
    why = cli.mcp_unavailable()
    assert why and "3.10" in why
    assert "pip install" not in why, (
        "on 3.9 the extra installs nothing, so advising pip sends the user in "
        "a circle; the only true instruction is a newer interpreter")

    # And the same reason reaches the two places a user actually looks.
    assert any("3.10" in p for p in cli.doctor_report()["problems"])
    lines = install("claude-code", dry_run=True, home="/tmp/ad-py39-probe")
    assert any("3.10" in ln and "WARNING" in ln for ln in lines)


# ---------------------------------------------------------------------------
# The dsh patch has to survive dsh, not just look right
# ---------------------------------------------------------------------------
#
# `--dump-config` composes the profile but does not validate an entry's config,
# so it passed while `dsh --profile headless` failed on the same file: the env
# block wrote one `!!js process.env.X` per key, and a variable that is not set
# is `undefined`, which is not the `string` the schema wants. dsh does not skip
# a bad entry -- it fails the whole plugin tree -- so `install dsh` left dsh
# unable to start at all for anyone without those keys exported.


def test_neither_dsh_patch_can_produce_undefined_env_values():
    """One filtered expression, not one line per key -- in *both* patch files.

    `install dsh` and the native plugin package each write a cordis patch, and
    only the first was fixed at first; the plugin route carried the same broken
    env block until they were made to share one generator.
    """
    from agentdescent.integrations import dsh_patch_block, dsh_plugin_patch

    for name, block in (("install", dsh_patch_block()), ("plugin", dsh_plugin_patch())):
        assert "filter(" in block and "!= null" in block, name
        for key in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
            assert f"\n          {key}: !!js" not in block, (
                f"{name}: {key} is written as its own YAML entry again; unset it "
                "becomes undefined and dsh refuses the whole plugin tree")


def test_the_dsh_env_expression_is_quoted_so_yaml_reads_one_scalar():
    """`OPENAI_API_KEY: process.env...` inside the expression is a mapping entry
    to a YAML parser -- unquoted, dsh rejects the file before composing."""
    import yaml

    from agentdescent.integrations import dsh_patch_block, dsh_plugin_patch

    for block in (dsh_patch_block(), dsh_plugin_patch()):
        doc = yaml.safe_load(block.replace("!!js ", ""))
        rows = doc[0]["insert"]
        env = next(r for r in rows if r["id"] == "mcp-agentdescent")["config"]["env"]
        assert isinstance(env, str) and env.startswith("Object.fromEntries"), env


def test_install_dsh_repairs_a_block_it_wrote_before(tmp_path, monkeypatch):
    """An install that wrote a broken entry could never fix it: the check was
    "is a marker present", so the second install said "already present" and left
    dsh unable to start. The only route back was editing the file by hand."""
    from agentdescent.integrations import DSH_BLOCK_START

    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    patch = tmp_path / "dsh" / "cordis.patch.yml"
    patch.parent.mkdir(parents=True)
    patch.write_text(
        "- id: something-of-the-users\n  name: keep-me\n"
        + DSH_BLOCK_START + "\n- insert:\n    - id: mcp-agentdescent\n"
        "      config:\n        env:\n"
        "          DEEPSEEK_API_KEY: !!js process.env.DEEPSEEK_API_KEY\n",
        encoding="utf-8")

    lines = install("dsh", home=str(tmp_path))
    text = patch.read_text(encoding="utf-8")
    assert any("rewrote" in l and "out of date" in l for l in lines), lines
    assert "filter(" in text                       # the fixed expression
    assert "keep-me" in text                       # the user's own rows survive
    assert text.count(DSH_BLOCK_START) == 1        # not stacked

    # ...and a second run changes nothing.
    again = install("dsh", home=str(tmp_path))
    assert any("already up to date" in l for l in again), again
    assert patch.read_text(encoding="utf-8") == text


def test_dsh_actually_boots_with_the_patch_installed(tmp_path, monkeypatch):
    """The check that would have caught it: load the plugin tree, not the config.

    Skips without dsh. `MISSING_CREDENTIAL` is a pass -- that is dsh reaching
    its provider, having loaded every plugin including ours.
    """
    if not shutil.which("dsh"):
        pytest.skip("needs an installed dsh")
    home = tmp_path / "home"
    monkeypatch.delenv("DSH_HOME", raising=False)
    install("dsh", home=str(home))
    out = subprocess.run(["dsh", "--profile", "headless", "say ok"],
                         capture_output=True, text=True, timeout=300,
                         env={**os.environ, "DSH_HOME": str(home / ".dsh"),
                              "HOME": str(home)}, cwd=str(tmp_path))
    combined = out.stdout + out.stderr
    for fatal in ("failed to parse patches", "plugin tree failed to load",
                  "invalid config", "YAMLException"):
        assert fatal not in combined, combined[:2000]


def test_the_native_dsh_plugin_also_boots(tmp_path):
    """The plugin route writes its own patch, and had the same broken env block.

    Skips without dsh and pnpm, and when corepack cannot lay out the pnpm the
    profile pins. As above, `MISSING_CREDENTIAL` is a pass.
    """
    if not (shutil.which("dsh") and shutil.which("pnpm")):
        pytest.skip("needs an installed dsh and pnpm")
    from agentdescent.integrations import render_dsh_plugin

    pkg = tmp_path / "pkg"
    render_dsh_plugin(str(pkg))
    home = tmp_path / "home"
    (home / ".dsh").mkdir(parents=True)
    env = {**os.environ, "DSH_HOME": str(home / ".dsh"), "HOME": str(home)}
    add = subprocess.run(["dsh", "plugin", "--profile", "headless", "add",
                          f"link:{pkg}"], capture_output=True, text=True,
                         timeout=300, env=env, cwd=str(tmp_path))
    assert "declares no dsh.bundle" not in add.stdout + add.stderr
    # `dsh plugin add` installs the profile's own pinned pnpm through corepack.
    # A corepack too old for that pin (node 22 ships 0.34.0, which cannot lay
    # out pnpm 12) fails here, and the only symptom downstream is a dumped
    # config missing our rows -- which reads like our bug and is not one.
    if "pnpm failed in profile directory" in add.stdout + add.stderr:
        pytest.skip("dsh could not provision its pinned pnpm (corepack too "
                    "old? try `npm i -g corepack@latest`): "
                    + (add.stdout + add.stderr)[-400:])
    out = subprocess.run(["dsh", "--profile", "headless", "--dump-config"],
                         capture_output=True, text=True, timeout=300,
                         env=env, cwd=str(tmp_path))
    combined = out.stdout + out.stderr
    for fatal in ("failed to parse patches", "plugin tree failed to load",
                  "invalid config", "YAMLException", "not found"):
        assert fatal not in combined, combined[:2000]
    assert "id: dsh-agentdescent" in combined and "id: mcp-agentdescent" in combined


# ---------------------------------------------------------------------------
# `plan` is the step that exists so nobody spends a run finding out
# ---------------------------------------------------------------------------
#
# Driving the skill in plain language against a real host, an agent wrote
# `"agent": {"ref": "claude"}` describing it as "the local Claude CLI, no API
# key needed". It is the Anthropic SDK completion; the CLI is `claude_code`.
# `plan` accepted it and priced 72 calls on a machine with neither the
# `anthropic` package nor a key, and the first call would have failed. Shape is
# not the same as "will run here".


def test_plan_warns_when_the_named_agent_cannot_run_here(monkeypatch, tmp_path):
    from agentdescent.cli import plan_payload
    from agentdescent.evolvespec import EvolveSpec

    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"prompt": "q", "gold": "a"}\n', encoding="utf-8")
    target = tmp_path / "prompt.txt"
    target.write_text("hi\n", encoding="utf-8")

    def spec_for(ref):
        return EvolveSpec.from_dict({
            "kind": "text", "target": str(target),
            "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
            "score": "contains", "agent": {"ref": ref, "model": "m"}})

    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    warnings = plan_payload(spec_for("claude"))["warnings"]
    assert any("Anthropic SDK" in w and "claude_code" in w for w in warnings), warnings

    warnings = plan_payload(spec_for("openai_compatible"))["warnings"]
    assert any("OPENAI_API_KEY" in w for w in warnings), warnings

    # And a usable one is quiet: no warning just because a ref was named.
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert plan_payload(spec_for("openai_compatible"))["warnings"] == []


def test_a_provider_error_survives_to_the_user_whole():
    """The run's cause of death was truncated where it mattered, and named twice.

    `executor` described the rollout failure and capped it at 200; `evolve`
    described the same exception again. What reached `status` was
    "RuntimeError: RuntimeError: ... does not support the c" -- the half that
    said *"refer to the documentation ... to select a compatible model"* fell
    off the end, and that half is the only actionable part.
    """
    from agentdescent.pipeline import describe as _describe

    long = ("https://endpoint/v3 returned HTTP 404 for model 'x': " + "detail. " * 60)
    once = _describe(RuntimeError(long))
    assert once.startswith("RuntimeError: ") and long in once

    # described a second time, it is not stamped a second time
    assert _describe(RuntimeError(once)) == once
    assert once.count("RuntimeError: ") == 1

    # and a different type still says what it was
    assert _describe(ValueError("boom")) == "ValueError: boom"


def test_plan_says_when_workers_buy_selection_rather_than_merging(tmp_path):
    """The one case left where `n_workers` does not buy a merge.

    A one-key artifact -- every `kind: "text"` target is a `SingleSlot` -- has
    worker proposals that contradict by construction, so without a fusion
    policy conflict resolution collapses them to one candidate and `n_workers`
    is per-round best-of-N. The reflective pair is the default now, built from
    the model the spec names, so the warning must stay quiet for an ordinary
    spec and fire only where no model is reachable to merge with: a spec whose
    only agent is a file-editing CLI, where paying an agent session per merge
    is not something to switch on unasked.
    """
    from agentdescent.cli import plan_payload
    from agentdescent.evolvespec import EvolveSpec

    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"prompt": "q", "gold": "a"}\n', encoding="utf-8")
    target = tmp_path / "prompt.txt"
    target.write_text("hi\n", encoding="utf-8")

    def warnings_for(agent, **evolve):
        spec = EvolveSpec.from_dict({
            "kind": "text", "target": str(target),
            "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
            "score": "contains", "agent": agent, "evolve": evolve})
        return [w for w in plan_payload(spec)["warnings"] if "best-of-N" in w]

    model = {"ref": "openai_compatible", "model": "m"}
    assert not warnings_for(model, n_workers=4), \
        "the merge pair is the default; there is nothing to warn about"
    assert not warnings_for(model, n_workers=1), \
        "one worker has nothing to say about merging either way"
    assert warnings_for({"ref": "claude_code"}, n_workers=4), \
        "no model to merge with, so four workers really are selection"


def test_doctor_reports_the_base_url_not_just_that_one_is_set():
    """Given only "OPENAI_API_KEY: true", an agent wrote `"model": "gpt-4o-mini"`
    against an endpoint that serves nothing of the sort. The URL is not a
    secret, and it is the only clue that the provider is not OpenAI."""
    import os

    from agentdescent.cli import doctor_report

    before = os.environ.get("OPENAI_BASE_URL")
    os.environ["OPENAI_BASE_URL"] = "https://example.invalid/v3"
    try:
        report = doctor_report()
        assert report["openai_base_url"] == "https://example.invalid/v3"
        assert any("model names are its own" in p for p in report["problems"])
    finally:
        if before is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = before


def test_the_skill_says_how_to_choose_an_agent():
    """The skill listed `agent` as one of the four things a spec needs and never
    said how to pick one. Driven in plain language, an agent chose a CLI coding
    agent to evolve a *prompt* -- one whole agent session per case to answer a
    question a model answers in one call."""
    text = skill_text()
    assert "Never a CLI coding" in text          # for kind: text
    assert "Never invent a model name" in text   # it guessed gpt-4o-mini
    assert "Anthropic SDK" in text               # `claude` is not `claude_code`
