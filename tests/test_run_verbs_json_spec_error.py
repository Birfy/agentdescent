"""Verbs that start work answer spec errors as JSON under --json (#650)."""
import json

import pytest

from agentdescent import cli

VERBS = [["evolve"], ["campaign"], ["tournament", "--algorithm", "gepa"]]


@pytest.mark.parametrize("verb", VERBS)
def test_missing_spec_is_json_with_json(tmp_path, capsys, verb):
    missing = str(tmp_path / "missing.yaml")
    assert cli.main([verb[0], missing, *verb[1:], "--json"]) == 2
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "missing.yaml" in payload["error"]
    assert err == ""


@pytest.mark.parametrize("verb", VERBS)
def test_malformed_spec_is_json_with_json(tmp_path, capsys, verb):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert cli.main([verb[0], str(bad), *verb[1:], "--json"]) == 2
    out, err = capsys.readouterr()
    assert json.loads(out)["ok"] is False
    assert err == ""


@pytest.mark.parametrize("verb", VERBS)
def test_missing_spec_is_a_stderr_line_without_json(tmp_path, capsys, verb):
    missing = str(tmp_path / "missing.yaml")
    assert cli.main([verb[0], missing, *verb[1:]]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("spec error: ")
    assert "missing.yaml" in err
