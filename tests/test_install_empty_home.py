"""An explicit empty ``--home`` is refused, not read as "no --home" (#412).

``--home "$UNSET_VAR"`` expands to ``""``; treating that as the real home would
let ``CODEX_HOME`` / ``DSH_HOME`` / ``XDG_CONFIG_HOME`` and ``~`` back in, and a
probe would edit (or remove) the real configuration #410 protects."""

from __future__ import annotations

import os
import pathlib

import pytest

from agentdescent import cli
from agentdescent.integrations import HOSTS, host_status, install, uninstall

ENV = ("DSH_HOME", "CODEX_HOME", "XDG_CONFIG_HOME")


@pytest.fixture
def real(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    monkeypatch.chdir(tmp_path)    # a relative "   " home must not land in the repo
    monkeypatch.setenv("HOME", str(real))
    for var in ENV:
        monkeypatch.setenv(var, str(real / var.lower()))
    return real


def _files(root):
    return sorted(os.path.relpath(os.path.join(b, f), root)
                  for b, _d, fs in os.walk(root) for f in fs)


@pytest.mark.parametrize("home", ["", "   "])
@pytest.mark.parametrize("host", sorted(HOSTS))
def test_api_refuses_an_empty_home(real, host, home):
    for fn in (lambda: install(host, home=home), lambda: uninstall(host, home=home),
               lambda: host_status(host, home=home)):
        with pytest.raises(ValueError, match="empty"):
            fn()
    assert _files(real) == []


@pytest.mark.parametrize("cmd", ["install", "uninstall"])
@pytest.mark.parametrize("host", sorted(HOSTS))
def test_cli_empty_home_exits_2_and_touches_nothing(real, host, cmd, capsys):
    if cmd == "uninstall":
        install(host)                       # a real install the probe must not remove
    before = _files(real)
    assert cli.main([cmd, host, "--home", ""]) == 2
    err = capsys.readouterr().err
    assert "empty" in err and "unset shell variable" in err
    assert "Traceback" not in err
    assert _files(real) == before


def test_none_and_pathlike_homes_still_work(real, tmp_path):
    # No --home: the host variables still apply (unchanged by #412).
    install("codex", dry_run=True)
    lines = install("codex", dry_run=True)
    assert any(str(real / "codex_home") in line for line in lines)
    # A Path is an explicit home like a str.
    probe = tmp_path / "probe"
    lines = install("codex", dry_run=True, home=pathlib.Path(probe))
    assert any(str(probe) in line for line in lines)
    assert not any(str(real) in line for line in lines)
