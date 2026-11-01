"""Foreground `evolve` reports a provider refusal cleanly (#652).

A reflector that raises `ProviderConfigError` used to escape `cmd_evolve` as a
Python traceback, with nothing parseable on stdout under `--json`.
"""
import json

import pytest

from agentdescent import cli
from agentdescent.agents import ProviderConfigError

from tests.test_evolvespec import _dir_spec

HINT = "endpoint https://example.invalid/v3 does not serve model 'nope'"


def refusing_reflect(prompt: str) -> str:
    raise ProviderConfigError(f"returned HTTP 404 for model 'nope' -- {HINT}")


@pytest.fixture
def spec_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTDESCENT_DEBUG", raising=False)
    spec = _dir_spec(str(tmp_path),
                     reflect={"ref": "tests.test_evolve_provider_error:refusing_reflect",
                              "call": False},
                     evolve={"rounds": 1, "n_workers": 1, "seed": 0})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    return str(path), str(tmp_path / "runs")


def test_json_is_one_ok_false_object(spec_path, capsys):
    path, store = spec_path
    assert cli.main(["--store", store, "evolve", path, "--json"]) == 1
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["state"] == "failed"
    assert "ProviderConfigError" in payload["error"] and HINT in payload["error"]
    assert payload["run_id"] and payload["dir"]
    assert "Traceback" not in err


def test_text_is_an_error_line_not_a_traceback(spec_path, capsys):
    path, store = spec_path
    assert cli.main(["--store", store, "evolve", path]) == 1
    out, err = capsys.readouterr()
    assert "Traceback" not in out + err
    assert "error: " in err and HINT in err
    assert "next: agentdescent diagnose " in err


def test_status_still_records_the_failure(spec_path, capsys):
    path, store = spec_path
    cli.main(["--store", store, "evolve", path, "--json"])
    rid = json.loads(capsys.readouterr().out)["run_id"]
    assert cli.main(["--store", store, "--json", "status", rid]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["state"] == "failed" and HINT in st["error"]


def test_debug_env_keeps_the_traceback(spec_path, monkeypatch):
    path, store = spec_path
    monkeypatch.setenv("AGENTDESCENT_DEBUG", "1")
    with pytest.raises(Exception, match="ProviderConfigError|does not serve"):
        cli.main(["--store", store, "evolve", path])
