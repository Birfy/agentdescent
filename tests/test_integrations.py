"""`agentdescent install <host>`: manifests into a fake home, and no drift."""

import json
import os

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
    agents = set(re.findall(r"`(claude_code|codex|dsh|openai_compatible|claude)`", text))
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


def test_dry_run_writes_nothing(tmp_path):
    for host in HOSTS:
        lines = install(host, dry_run=True, home=str(tmp_path))
        assert lines and all(l.startswith(("would", "kept", "forwarded", "verify", "load", "or,", "Codex"))
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
