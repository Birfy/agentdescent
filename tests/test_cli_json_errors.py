"""Run-store and archive errors answer as JSON under --json (#658)."""
import json

import pytest

from agentdescent import cli

RUN_VERBS = ["status", "show", "apply", "cancel", "resume", "diagnose"]


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))


def _store(tmp_path):
    return str(tmp_path / "home" / "runs")


@pytest.mark.parametrize("verb", RUN_VERBS)
@pytest.mark.parametrize("before", [True, False])
def test_unknown_run_is_json_with_json(tmp_path, capsys, verb, before):
    argv = ["--store", _store(tmp_path)]
    argv += ["--json", verb, "nosuchrun"] if before else [verb, "nosuchrun", "--json"]
    assert cli.main(argv) == 2
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "nosuchrun" in payload["error"]
    assert err == ""


@pytest.mark.parametrize("verb", RUN_VERBS)
def test_unknown_run_is_a_stderr_line_without_json(tmp_path, capsys, verb):
    assert cli.main(["--store", _store(tmp_path), verb, "nosuchrun"]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: no run ")


def test_missing_archive_is_json_with_json(tmp_path, capsys):
    missing = str(tmp_path / "nosuch")
    assert cli.main(["portfolio", missing, "--json"]) == 2
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "does not exist" in payload["error"]
    assert err == ""


def test_missing_archive_is_a_stderr_line_without_json(tmp_path, capsys):
    assert cli.main(["portfolio", str(tmp_path / "nosuch")]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: archive ")
