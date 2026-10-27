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
