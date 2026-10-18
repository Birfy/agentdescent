"""A literal ``~`` in ``--store`` or ``$AGENTDESCENT_HOME`` is expanded everywhere (#402).

No shell expands it when it comes from an MCP host config's ``args``/``env`` or a
quoted ``--store '~/st'``. Before the fix ``create`` wrote under ``$HOME`` (via
``RunDir``) while ``list_runs`` looked in ``./~`` and left a stray directory there.
"""

import os

from agentdescent import cli, runstore
from agentdescent.mcp import Tools


def _home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    return home, cwd


def test_tilde_store_create_list_get_agree(tmp_path, monkeypatch):
    home, cwd = _home(tmp_path, monkeypatch)
    rd = runstore.create({"kind": "x"}, store="~/st")
    assert rd.path == str(home / "st" / rd.run_id)
    assert [s.run_id for s in runstore.list_runs(store="~/st")] == [rd.run_id]
    assert runstore.get(rd.run_id, store="~/st").path == rd.path
    assert not (cwd / "~").exists()


def test_tilde_agentdescent_home_is_expanded(tmp_path, monkeypatch):
    home, cwd = _home(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENTDESCENT_HOME", "~/adh")
    assert runstore.root() == str(home / "adh" / "runs")
    rd = runstore.create({"kind": "x"})
    assert rd.path.startswith(str(home / "adh" / "runs"))
    assert [s.run_id for s in runstore.list_runs()] == [rd.run_id]
    assert not (cwd / "~").exists()


def test_store_path_is_absolute_for_relative_and_tilde(tmp_path, monkeypatch):
    home, cwd = _home(tmp_path, monkeypatch)
    assert runstore.store_path("./s") == str(cwd / "s")
    assert runstore.store_path("~/s") == str(home / "s")
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "h"))
    assert runstore.store_path(None) == str(tmp_path / "h" / "runs")


def test_mcp_status_and_cli_status_name_the_expanded_store(tmp_path, monkeypatch, capsys):
    home, cwd = _home(tmp_path, monkeypatch)
    out = Tools(store="~/st").status()
    assert out["store"] == str(home / "st")
    assert out["runs"] == []
    assert cli.main(["--store", "~/st", "status"]) == 0
    assert str(home / "st") in capsys.readouterr().out
