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
