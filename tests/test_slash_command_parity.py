"""Slash-command parity: OpenCode command/ + Codex prompts vs Claude (#165, #384)."""

from __future__ import annotations

import os

import pytest

from agentdescent.integrations import (
    ALGORITHMS_COMMAND,
    CODEX_ALGORITHMS_REL,
    CODEX_EVOLVE_REL,
    EVOLVE_COMMAND,
    OPENCODE_ALGORITHMS_REL,
    OPENCODE_EVOLVE_REL,
    install,
    uninstall,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("DSH_HOME", "CODEX_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_install_opencode_writes_nested_evolve_command(tmp_path):
    home = str(tmp_path)
    lines = install("opencode", home=home)
    path = os.path.join(home, ".config", "opencode", *OPENCODE_EVOLVE_REL)
    assert os.path.isfile(path)
    assert _read(path) == EVOLVE_COMMAND
    assert any("slash: /agentdescent/evolve" in line for line in lines)


def test_install_opencode_writes_nested_algorithms_command(tmp_path):
    home = str(tmp_path)
    lines = install("opencode", home=home)
    path = os.path.join(home, ".config", "opencode", *OPENCODE_ALGORITHMS_REL)
    assert os.path.isfile(path)
    assert _read(path) == ALGORITHMS_COMMAND
    assert any("slash: /agentdescent/algorithms" in line for line in lines)


def test_install_codex_writes_prompts_evolve(tmp_path):
    home = str(tmp_path)
    lines = install("codex", home=home)
    path = os.path.join(home, ".codex", *CODEX_EVOLVE_REL)
    assert os.path.isfile(path)
    assert _read(path) == EVOLVE_COMMAND
    assert any("slash: /prompts:evolve" in line for line in lines)


def test_install_codex_writes_prompts_algorithms(tmp_path):
    home = str(tmp_path)
    lines = install("codex", home=home)
    path = os.path.join(home, ".codex", *CODEX_ALGORITHMS_REL)
    assert os.path.isfile(path)
    assert _read(path) == ALGORITHMS_COMMAND
    assert any("slash: /prompts:algorithms" in line for line in lines)


def test_slash_commands_match_claude_plugin_files(tmp_path):
    """One source of truth: Claude plugin, OpenCode command/, Codex prompts."""
    from agentdescent.integrations import render_claude_plugin

    dest = str(tmp_path / "plugin")
    render_claude_plugin(dest)
    assert _read(os.path.join(dest, "commands", "evolve.md")) == EVOLVE_COMMAND
    assert _read(os.path.join(dest, "commands", "algorithms.md")) == ALGORITHMS_COMMAND


def test_uninstall_removes_stock_slash_entries(tmp_path):
    home = str(tmp_path)
    install("opencode", home=home)
    install("codex", home=home)
    uninstall("opencode", home=home)
    uninstall("codex", home=home)
    assert not os.path.exists(
        os.path.join(home, ".config", "opencode", *OPENCODE_EVOLVE_REL)
    )
    assert not os.path.exists(
        os.path.join(home, ".config", "opencode", *OPENCODE_ALGORITHMS_REL)
    )
    assert not os.path.exists(os.path.join(home, ".codex", *CODEX_EVOLVE_REL))
    assert not os.path.exists(os.path.join(home, ".codex", *CODEX_ALGORITHMS_REL))


def test_uninstall_keeps_an_edited_opencode_command(tmp_path):
    home = str(tmp_path)
    install("opencode", home=home)
    path = os.path.join(home, ".config", "opencode", *OPENCODE_EVOLVE_REL)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("edited by hand\n")
    lines = uninstall("opencode", home=home)
    assert _read(path) == "edited by hand\n"
    assert any(line.startswith("kept ") and "evolve slash" in line for line in lines)


def test_uninstall_keeps_an_edited_opencode_algorithms_command(tmp_path):
    home = str(tmp_path)
    install("opencode", home=home)
    path = os.path.join(home, ".config", "opencode", *OPENCODE_ALGORITHMS_REL)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("edited by hand\n")
    lines = uninstall("opencode", home=home)
    assert _read(path) == "edited by hand\n"
    assert any(line.startswith("kept ") and "algorithms slash" in line for line in lines)


def test_uninstall_keeps_an_edited_codex_prompt(tmp_path):
    home = str(tmp_path)
    install("codex", home=home)
    path = os.path.join(home, ".codex", *CODEX_EVOLVE_REL)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("edited by hand\n")
    lines = uninstall("codex", home=home)
    assert _read(path) == "edited by hand\n"
    assert any(line.startswith("kept ") and "evolve slash" in line for line in lines)


def test_uninstall_keeps_an_edited_codex_algorithms_prompt(tmp_path):
    home = str(tmp_path)
    install("codex", home=home)
    path = os.path.join(home, ".codex", *CODEX_ALGORITHMS_REL)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("edited by hand\n")
    lines = uninstall("codex", home=home)
    assert _read(path) == "edited by hand\n"
    assert any(line.startswith("kept ") and "algorithms slash" in line for line in lines)
