"""``agentdescent uninstall <host>`` reverses ``install`` and nothing else (#165)."""

from __future__ import annotations

import json
import os

import pytest

from agentdescent import cli
from agentdescent.integrations import (DSH_BLOCK_END, DSH_BLOCK_START, HOSTS,
                                       host_status, install, uninstall)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("DSH_HOME", "CODEX_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)


def _snapshot(home):
    out = {}
    for base, _dirs, files in os.walk(home):
        for name in files:
            path = os.path.join(base, name)
            with open(path, encoding="utf-8") as fh:
                out[os.path.relpath(path, home)] = fh.read()
    return out


def _dirs(home):
    return sorted(os.path.relpath(os.path.join(b, d), home)
                  for b, ds, _ in os.walk(home) for d in ds)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# the user's own configuration, present before agentdescent ever ran
USER_FILES = {
    "dsh": (".dsh/cordis.patch.yml",
            "- insert:\n    - id: my-row\n      name: '@acme/thing'\n"),
    "codex": (".codex/config.toml",
              'model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "other"\nargs = []\n'),
    "opencode": (".config/opencode/opencode.jsonc",
                 json.dumps({"$schema": "https://opencode.ai/config.json",
                             "mcp": {"other": {"type": "local", "command": ["other"]}},
                             "theme": "dark"}, indent=2) + "\n"),
}


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_install_then_uninstall_restores_a_fresh_home(tmp_path, host):
    home = str(tmp_path)
    install(host, home=home)
    assert host_status(host, home=home)["state"] == "ok"
    lines = uninstall(host, home=home)
    assert host_status(host, home=home)["state"] == "missing"
    assert any(line.startswith("removed ") for line in lines)
    assert _snapshot(home) == {}
    # no empty directory of ours left behind (the host's own dir, and the
    # `~/.agentdescent` that also holds the run store, may remain)
    assert not [d for d in _dirs(home) if d.endswith("agentdescent") and d != ".agentdescent"
                or d.endswith("plugins")]


@pytest.mark.parametrize("host", sorted(USER_FILES))
def test_uninstall_gives_back_the_users_own_configuration(tmp_path, host):
    home = str(tmp_path)
    rel, text = USER_FILES[host]
    _write(os.path.join(home, rel), text)
    install(host, home=home)
    assert _snapshot(home)[rel] != text           # install did change it
    uninstall(host, home=home)
    assert host_status(host, home=home)["state"] == "missing"
    after = _snapshot(home)
    if host == "opencode":
        assert json.loads(after[rel]) == json.loads(text)
    else:
        assert after[rel] == text
    assert list(after) == [rel]


def test_claude_code_uninstall_leaves_the_run_store_alone(tmp_path):
    home = str(tmp_path)
    _write(os.path.join(home, ".agentdescent", "runs", "r1", "state.json"), "{}")
    install("claude-code", home=home)
    uninstall("claude-code", home=home)
    assert list(_snapshot(home)) == [os.path.join(".agentdescent", "runs", "r1", "state.json")]
    assert not os.path.exists(os.path.join(home, ".agentdescent", "plugins", "claude-code"))


def test_claude_code_keeps_a_plugin_dir_that_holds_other_files(tmp_path):
    home = str(tmp_path)
    install("claude-code", home=home)
    extra = os.path.join(home, ".agentdescent", "plugins", "claude-code", "mine.txt")
    _write(extra, "keep me")
    lines = uninstall("claude-code", home=home)
    assert os.path.isfile(extra)
    assert any(line.startswith("kept ") and "did not write" in line for line in lines)


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_dry_run_reports_and_changes_nothing(tmp_path, host):
    home = str(tmp_path)
    if host in USER_FILES:
        _write(os.path.join(home, USER_FILES[host][0]), USER_FILES[host][1])
    install(host, home=home)
    before = _snapshot(home)
    lines = uninstall(host, dry_run=True, home=home)
    assert _snapshot(home) == before
    assert lines and all(line.startswith(("would ", "kept ")) for line in lines)


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_uninstalling_twice_is_a_no_op(tmp_path, host):
    home = str(tmp_path)
    install(host, home=home)
    uninstall(host, home=home)
    assert uninstall(host, home=home)[0].startswith("nothing to remove")
    assert uninstall(host, home=str(tmp_path / "never-used"))[0].startswith("nothing to remove")


def test_unknown_host_is_refused():
    with pytest.raises(ValueError, match="unknown host"):
        uninstall("vim")


def test_dsh_block_without_an_end_marker_runs_to_the_end_of_the_file(tmp_path):
    """Installs from before the end marker existed appended the block last."""
    home = str(tmp_path)
    patch = os.path.join(home, ".dsh", "cordis.patch.yml")
    _write(patch, "- insert:\n    - id: mine\n" + DSH_BLOCK_START + "\n- insert:\n    - id: x\n")
    uninstall("dsh", home=home)
    assert _snapshot(home) == {os.path.join(".dsh", "cordis.patch.yml"): "- insert:\n    - id: mine\n"}


def test_dsh_keeps_what_the_user_wrote_after_the_block(tmp_path):
    home = str(tmp_path)
    install("dsh", home=home)
    patch = os.path.join(home, ".dsh", "cordis.patch.yml")
    with open(patch, "a", encoding="utf-8") as fh:
        fh.write("- insert:\n    - id: later\n")
    uninstall("dsh", home=home)
    text = _snapshot(home)[os.path.join(".dsh", "cordis.patch.yml")]
    assert "later" in text and DSH_BLOCK_START not in text and DSH_BLOCK_END not in text
    assert "mcp-agentdescent" not in text


def test_uninstall_follows_the_hosts_home_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dshhome"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codexhome"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    home = str(tmp_path / "home")
    for host in ("dsh", "codex", "opencode"):
        install(host, home=home)
    for host in ("dsh", "codex", "opencode"):
        uninstall(host, home=home)
        assert host_status(host, home=home)["state"] == "missing"
    assert _snapshot(str(tmp_path)) == {}


def test_codex_keeps_a_table_the_user_edited(tmp_path):
    home = str(tmp_path)
    install("codex", home=home)
    cfg = os.path.join(home, ".codex", "config.toml")
    with open(cfg, encoding="utf-8") as fh:
        text = fh.read()
    edited = text + 'env = { AGENTDESCENT_HOME = "/x" }\n'
    _write(cfg, edited)
    lines = uninstall("codex", home=home)
    assert _snapshot(home)[os.path.join(".codex", "config.toml")] == edited
    assert any(line.startswith("kept ") and "not the one `install` wrote" in line
               for line in lines)
    # the skill, which is ours alone, is still removed
    assert not os.path.exists(os.path.join(home, ".codex", "skills", "agentdescent"))


def test_codex_keeps_a_table_written_by_hand(tmp_path):
    home = str(tmp_path)
    cfg = os.path.join(home, ".codex", "config.toml")
    hand = '[mcp_servers.agentdescent]\ncommand = "/opt/ad/bin/agentdescent"\nargs = ["mcp"]\n'
    _write(cfg, hand)
    lines = uninstall("codex", home=home)
    assert _snapshot(home) == {os.path.join(".codex", "config.toml"): hand}
    assert any(line.startswith("kept ") for line in lines)


def test_opencode_keeps_an_entry_the_user_customised(tmp_path):
    home = str(tmp_path)
    path = os.path.join(home, ".config", "opencode", "opencode.jsonc")
    custom = json.dumps({"mcp": {"agentdescent": {
        "type": "local", "command": ["agentdescent", "mcp"],
        "environment": {"AGENTDESCENT_HOME": "/x"}}}}, indent=2) + "\n"
    _write(path, custom)
    lines = uninstall("opencode", home=home)
    assert _snapshot(home) == {os.path.relpath(path, home): custom}
    assert any(line.startswith("kept ") and "not the one `install` wrote" in line
               for line in lines)


def test_opencode_with_comments_is_never_rewritten(tmp_path):
    home = str(tmp_path)
    path = os.path.join(home, ".config", "opencode", "opencode.jsonc")
    commented = ('// my config\n{"mcp": {"agentdescent": {"type": "local", '
                 '"command": ["agentdescent", "mcp"]}}}\n')
    _write(path, commented)
    _write(os.path.join(home, ".config", "opencode", "skill", "agentdescent", "SKILL.md"), "x")
    lines = uninstall("opencode", home=home)
    assert _snapshot(home) == {os.path.relpath(path, home): commented}
    assert any("could not parse it" in line for line in lines)


def test_opencode_keeps_the_other_servers_and_settings(tmp_path):
    home = str(tmp_path)
    rel, text = USER_FILES["opencode"]
    _write(os.path.join(home, rel), text)
    install("opencode", home=home)
    uninstall("opencode", home=home)
    after = json.loads(_snapshot(home)[rel])
    assert after["mcp"] == {"other": {"type": "local", "command": ["other"]}}
    assert after["theme"] == "dark"


# -- the CLI ----------------------------------------------------------------------


def test_cli_uninstall_prints_what_it_did(tmp_path, capsys):
    home = str(tmp_path)
    install("codex", home=home)
    capsys.readouterr()
    assert cli.main(["uninstall", "codex", "--home", home, "--dry-run"]) == 0
    assert "would remove" in capsys.readouterr().out
    assert host_status("codex", home=home)["state"] == "ok"
    assert cli.main(["uninstall", "codex", "--home", home]) == 0
    assert "removed" in capsys.readouterr().out
    assert host_status("codex", home=home)["state"] == "missing"


def test_cli_uninstall_rejects_an_unknown_host():
    with pytest.raises(SystemExit):
        cli.main(["uninstall", "vim"])
