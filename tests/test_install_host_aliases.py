"""`install` / `uninstall` take host aliases and list hosts when none is named (#721)."""

import json

import pytest

from agentdescent import cli
from agentdescent.integrations import HOSTS, canonical_host, install


@pytest.mark.parametrize("alias, host", [
    ("claude", "claude-code"), ("Claude-Code", "claude-code"), ("claude_code", "claude-code"),
    ("grok", "grok-bot"), ("GrokBot", "grok-bot"), ("grok_bot", "grok-bot"),
    ("deepseek-harness", "dsh"), ("DSH", "dsh"), ("open-code", "opencode"),
    ("codex", "codex"),
])
def test_canonical_host(alias, host):
    assert canonical_host(alias) == host
    assert host in HOSTS


def test_unknown_host_is_folded_not_mapped():
    assert canonical_host("Nope") == "nope"
    with pytest.raises(ValueError, match="unknown host 'nope'"):
        install("Nope", dry_run=True, home="/tmp")


def test_install_alias_dry_run(tmp_path, capsys):
    assert cli.main(["install", "claude", "--dry-run", "--home", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "plugins/claude-code" in out
    assert not any(tmp_path.iterdir())          # dry run wrote nothing


def test_install_alias_json_reports_canonical_host(tmp_path, capsys):
    assert cli.main(["install", "Grok", "--dry-run", "--home", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["host"] == "grok-bot"


def test_unknown_host_still_usage_error(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["install", "nope"])
    assert ei.value.code == 2
    assert "invalid choice: 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize("verb", ["install", "uninstall"])
def test_bare_verb_lists_hosts(tmp_path, capsys, verb):
    assert cli.main([verb, "--home", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert f"{verb}: name a host" in err
    for host in HOSTS:
        assert host in err


@pytest.mark.parametrize("verb", ["install", "uninstall"])
def test_bare_verb_json(tmp_path, capsys, verb):
    assert cli.main([verb, "--home", str(tmp_path), "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert set(payload["hosts"]) == set(HOSTS)
    assert isinstance(payload["suggested"], list)


def test_bare_install_does_not_write(tmp_path):
    cli.main(["install", "--home", str(tmp_path)])
    assert not any(tmp_path.iterdir())
