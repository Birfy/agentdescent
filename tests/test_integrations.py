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
        assert f"{key}: !!js process.env.{key}" in patch      # the scrubbing workaround
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
    assert any("already present" in l for l in lines)


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
    import agentdescent.integrations as integrations

    monkeypatch.setattr(integrations, "mcp_sdk_missing", lambda: True)
    lines = install("claude-code", dry_run=True, home=str(tmp_path))
    assert any("agentdescent[mcp]" in l and l.startswith("WARNING") for l in lines), lines
    monkeypatch.setattr(integrations, "mcp_sdk_missing", lambda: False)
    assert not any(l.startswith("WARNING") for l in install("dsh", dry_run=True, home=str(tmp_path)))


def test_agentdescent_mcp_without_the_sdk_says_how_to_get_it(monkeypatch, capsys):
    """A traceback here is invisible: the host shows the user a closed pipe."""
    import builtins

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
    assert 'pip install "agentdescent[mcp]"' in err, err
    assert "Traceback" not in err


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
        assert f"{key}: !!js process.env.{key}" in patch


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
