"""``--home`` is the host's whole world: ``CODEX_HOME`` / ``DSH_HOME`` /
``XDG_CONFIG_HOME`` in the environment must not redirect it to the real
configuration (#410). Without ``--home`` they still apply."""

from __future__ import annotations

import os

import pytest

from agentdescent import cli
from agentdescent.integrations import HOSTS, host_status, install, uninstall

ENV = {"dsh": "DSH_HOME", "codex": "CODEX_HOME", "opencode": "XDG_CONFIG_HOME"}


def _tree(root):
    out = {}
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            with open(path, encoding="utf-8") as fh:
                out[os.path.relpath(path, root)] = fh.read()
    return out


@pytest.fixture
def worlds(tmp_path, monkeypatch):
    """A 'real' home whose variables are exported, and a throwaway --home."""
    real = tmp_path / "real"
    probe = tmp_path / "probe"
    for d in (real, probe):
        d.mkdir()
    monkeypatch.setenv("HOME", str(real))
    for var in ENV.values():
        monkeypatch.setenv(var, str(real / var.lower()))
    return real, probe


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_install_with_home_leaves_the_exported_directories_alone(worlds, host):
    real, probe = worlds
    install(host, home=str(probe))
    assert _tree(real) == {}, "wrote through the host's home variable"
    assert _tree(probe), "wrote nothing under --home"
    assert host_status(host, home=str(probe))["state"] == "ok"
    # status of the exported directories is independent of the probe
    assert host_status(host)["state"] == "missing"


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_dry_run_with_home_reports_paths_under_home(worlds, host):
    real, probe = worlds
    lines = install(host, dry_run=True, home=str(probe))
    written = [l for l in lines if l.startswith("would")]
    assert written and all(str(probe) in l and str(real) not in l for l in written), lines
    assert _tree(probe) == {} and _tree(real) == {}


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_uninstall_with_home_touches_only_that_home(worlds, host):
    real, probe = worlds
    install(host)                                   # the real, variable-directed install
    install(host, home=str(probe))
    real_before = _tree(real)
    assert real_before
    uninstall(host, home=str(probe))
    assert _tree(real) == real_before, "uninstall --home removed from the real configuration"
    assert _tree(probe) == {}
    lines = uninstall(host, dry_run=True, home=str(probe))
    assert any(l.startswith("nothing to remove") for l in lines), lines
    uninstall(host)                                 # and without --home the variables apply
    assert _tree(real) == {}


@pytest.mark.parametrize("host", sorted(set(HOSTS) - {"claude-code", "grok-bot"}))
def test_without_home_the_host_variable_still_applies(worlds, host):
    real, probe = worlds
    install(host)
    exported = real / ENV[host].lower()
    assert _tree(exported), f"{ENV[host]} was ignored"
    assert host_status(host)["state"] == "ok"
    assert _tree(probe) == {}


def test_the_cli_flag_wins_too(worlds, capsys):
    real, probe = worlds
    assert cli.main(["install", "codex", "--home", str(probe)]) == 0
    out = capsys.readouterr().out
    assert str(probe) in out and str(real) not in out
    assert _tree(real) == {}
    assert cli.main(["uninstall", "codex", "--home", str(probe)]) == 0
    assert _tree(probe) == {}
