"""`plan` errors are human-readable unless --json is asked for (#551)."""
import json

import pytest

from agentdescent import cli


def _spec(tmp_path):
    data = tmp_path / "cases.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"q{i}", "gold": "a"}) + "\n"
                            for i in range(10)))
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"kind": "text", "target": "instruction",
                                "data": {"path": str(data)},
                                "agent": "openai_compatible"}))
    return str(path)


@pytest.mark.parametrize("flags", [["--algorithm", "bogus"],
                                   ["--algorithm", "gepa", "--param", "nope=1"]])
def test_plan_error_is_a_stderr_line_without_json(tmp_path, capsys, flags):
    assert cli.main(["plan", _spec(tmp_path), *flags]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: ")


@pytest.mark.parametrize("flags", [["--algorithm", "bogus"],
                                   ["--algorithm", "gepa", "--param", "nope=1"]])
def test_plan_error_stays_json_with_json(tmp_path, capsys, flags):
    assert cli.main(["plan", _spec(tmp_path), *flags, "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]


def _few_tasks_spec(tmp_path, n=3):
    data = tmp_path / "few.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"q{i}", "gold": "a"}) + "\n"
                            for i in range(n)))
    path = tmp_path / "few.json"
    path.write_text(json.dumps({"kind": "text", "target": "instruction",
                                "data": {"path": str(data)}, "score": "contains",
                                "agent": "echo", "reflect": "echo"}))
    return str(path)


def test_plan_hard_check_is_a_stderr_line_without_json(tmp_path, capsys):
    # A spec that compiles but fails a hard check (#343 too few tasks) (#553).
    assert cli.main(["plan", _few_tasks_spec(tmp_path)]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: ")
    assert "plan saw 3" in err


def test_plan_hard_check_stays_json_with_json(tmp_path, capsys):
    assert cli.main(["plan", _few_tasks_spec(tmp_path), "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "plan saw 3" in payload["error"]


def test_unreadable_plan_spec_stays_json_with_json(tmp_path, capsys):
    # A missing spec file failed before the guard, so `--json` got a bare
    # stderr line instead of the JSON error object (#648).
    missing = str(tmp_path / "missing.yaml")
    assert cli.main(["plan", missing, "--json"]) == 2
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "missing.yaml" in payload["error"]
    assert err == ""


def test_malformed_plan_spec_stays_json_with_json(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert cli.main(["--json", "plan", str(bad)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]


def test_unreadable_plan_spec_is_a_stderr_line_without_json(tmp_path, capsys):
    assert cli.main(["plan", str(tmp_path / "missing.yaml")]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("error: spec ")
