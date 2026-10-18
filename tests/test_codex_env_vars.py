"""Codex scrubs an MCP server's environment, so every Codex route must name
what to forward (#404): ``env_vars`` in ``config.toml`` and in the plugin's
``.mcp.json``."""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

from agentdescent.integrations import (
    FORWARDED_ENV, _legacy_codex_block, codex_config_block, host_status, install, mcp_manifest,
    render_claude_plugin, uninstall,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("DSH_HOME", "CODEX_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)


def _cfg(home):
    return home / ".codex" / "config.toml"


def test_forwarded_env_names_the_keys_and_the_run_store():
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY",
                 "ARK_API_KEY", "AGENTDESCENT_HOME"):
        assert name in FORWARDED_ENV
    assert len(set(FORWARDED_ENV)) == len(FORWARDED_ENV)


def test_codex_block_declares_env_vars_as_valid_toml():
    block = codex_config_block(["/opt/py", "-P", "-m", "agentdescent.cli"])
    assert f"env_vars = {json.dumps(list(FORWARDED_ENV))}\n" in block
    if sys.version_info >= (3, 11):
        import tomllib
        table = tomllib.loads(block)["mcp_servers"]["agentdescent"]
        assert table["env_vars"] == list(FORWARDED_ENV)
        assert table["command"] == "/opt/py"
    else:
        assert re.search(r'^env_vars = \[("[A-Z_]+"(, )?)+\]$', block, re.M)


def test_install_codex_writes_env_vars_once(tmp_path):
    lines = install("codex", home=str(tmp_path))
    text = _cfg(tmp_path).read_text()
    assert text.count("env_vars = ") == 1 and "AGENTDESCENT_HOME" in text
    assert any("forwarded into the MCP server env" in l and "OPENAI_API_KEY" in l for l in lines)
    again = install("codex", home=str(tmp_path))
    assert _cfg(tmp_path).read_text() == text
    assert any(l.startswith("kept ") for l in again), again


@pytest.mark.parametrize("launcher", [None, ["/old/python", "-m", "agentdescent.cli"],
                                      ["/old/python", "-P", "-m", "agentdescent.cli"]])
def test_reinstall_repairs_the_legacy_block(tmp_path, monkeypatch, launcher):
    monkeypatch.setattr(sys, "executable", "/new/python")
    _cfg(tmp_path).parent.mkdir()
    legacy = _legacy_codex_block(launcher)
    assert "env_vars" not in legacy
    _cfg(tmp_path).write_text('model = "x"\n' + legacy)
    assert host_status("codex", home=str(tmp_path))["mcp"] == "ok"
    install("codex", home=str(tmp_path))
    text = _cfg(tmp_path).read_text()
    assert text.startswith('model = "x"\n')
    assert text.count("[mcp_servers.agentdescent]") == 1
    assert "env_vars = " in text and "AGENTDESCENT_HOME" in text
    install("codex", home=str(tmp_path))
    assert _cfg(tmp_path).read_text() == text


def test_doctor_reads_the_legacy_block_as_stale_and_the_new_one_as_ok(tmp_path):
    install("codex", home=str(tmp_path))
    assert host_status("codex", home=str(tmp_path))["state"] == "ok"
    _cfg(tmp_path).write_text(_legacy_codex_block())
    assert host_status("codex", home=str(tmp_path))["state"] == "stale"
    install("codex", home=str(tmp_path))
    assert host_status("codex", home=str(tmp_path))["state"] == "ok"


@pytest.mark.parametrize("make", [codex_config_block, _legacy_codex_block])
def test_uninstall_removes_legacy_and_new_blocks(tmp_path, make):
    home = str(tmp_path)
    install("codex", home=home)
    _cfg(tmp_path).write_text('model = "keep"\n' + make()
                              + '\n[projects."/p"]\ntrust_level = "trusted"\n')
    uninstall("codex", home=home)
    assert _cfg(tmp_path).read_text() == 'model = "keep"\n\n[projects."/p"]\ntrust_level = "trusted"\n'


@pytest.mark.parametrize("edit", [
    'env_vars = ["MINE"]\n',
    'env = { PATH = "/x" }\n',
])
def test_a_user_edited_table_is_kept(tmp_path, edit):
    home = str(tmp_path)
    install("codex", home=home)
    legacy = _legacy_codex_block()
    # a hand-added env_vars after the legacy block, and an extra key after the new one
    for base in (legacy, codex_config_block()):
        _cfg(tmp_path).write_text(base + edit)
        install("codex", home=home)
        assert _cfg(tmp_path).read_text() == base + edit
        lines = uninstall("codex", home=home)
        assert _cfg(tmp_path).read_text() == base + edit
        assert any("not the one `install` wrote" in l for l in lines)


def test_plugin_mcp_manifest_forwards_the_same_variables(tmp_path):
    entry = mcp_manifest()["mcpServers"]["agentdescent"]
    assert entry["env_vars"] == list(FORWARDED_ENV)
    checked_in = json.load(open(os.path.join(ROOT, "integrations", "claude-code", ".mcp.json")))
    assert checked_in["mcpServers"]["agentdescent"] == entry
    dest = tmp_path / "plugin"
    render_claude_plugin(str(dest))
    rendered = json.loads((dest / ".mcp.json").read_text())["mcpServers"]["agentdescent"]
    assert rendered["env_vars"] == checked_in["mcpServers"]["agentdescent"]["env_vars"]


PROJECTS = '\n[projects."/home/u/proj"]\ntrust_level = "trusted"\n'


@pytest.mark.parametrize("launcher", [None, ["/old/python", "-P", "-m", "agentdescent.cli"]])
def test_legacy_block_followed_by_a_codex_table_is_repaired_in_place(tmp_path, launcher):
    """Codex appends `[projects."..."]` once a folder is trusted -- the usual state
    of anyone who used the old install. The block is not last, but it is ours."""
    home = str(tmp_path)
    legacy = _legacy_codex_block(launcher)
    before = 'model = "x"\n' + legacy + PROJECTS
    _cfg(tmp_path).parent.mkdir()
    _cfg(tmp_path).write_text(before)
    assert host_status("codex", home=home)["state"] == "stale"

    lines = install("codex", home=home)
    after = _cfg(tmp_path).read_text()
    assert after == ('model = "x"\n' + legacy + f"env_vars = {json.dumps(list(FORWARDED_ENV))}\n"
                     + PROJECTS), after
    assert not any(l.startswith("kept ") for l in lines), lines
    assert host_status("codex", home=home)["state"] == "ok"
    # idempotent, and uninstall still removes the repaired block from the middle
    again = install("codex", home=home)
    assert _cfg(tmp_path).read_text() == after and any(l.startswith("kept ") for l in again)
    uninstall("codex", home=home)
    assert _cfg(tmp_path).read_text() == 'model = "x"\n' + PROJECTS


def test_dry_run_reports_the_in_place_repair_and_writes_nothing(tmp_path):
    _cfg(tmp_path).parent.mkdir()
    text = _legacy_codex_block() + PROJECTS
    _cfg(tmp_path).write_text(text)
    lines = install("codex", dry_run=True, home=str(tmp_path))
    assert any(l.startswith("would write") and "config.toml" in l for l in lines), lines
    assert _cfg(tmp_path).read_text() == text


def test_a_hand_edited_legacy_block_before_a_table_is_left_alone(tmp_path):
    _cfg(tmp_path).parent.mkdir()
    text = _legacy_codex_block() + 'startup_timeout_sec = 90\n' + PROJECTS
    _cfg(tmp_path).write_text(text)
    install("codex", home=str(tmp_path))
    assert _cfg(tmp_path).read_text() == text
