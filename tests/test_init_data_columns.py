"""``init --data`` maps common column names; spec loading names spec fields (#561)."""

import json

import pytest

from agentdescent import cli
from agentdescent.evolvespec import EvolveSpec, SpecError, build_tasks


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# x")
    return tmp_path


def _cases(ws, rows, name="cases.jsonl"):
    (ws / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return name


def _init(*argv):
    return cli.main(["init", "my-skill", "--kind", "skill_dir", "--agent", "codex",
                     *argv, "--out", "spec.json"])


def test_init_maps_input_expected(ws, capsys):
    data = _cases(ws, [{"input": "2 inches in cm?", "expected": "5.08"}])
    assert _init("--data", data) == 0
    spec = json.loads((ws / "spec.json").read_text())
    assert spec["data"]["prompt"] == "input"
    assert spec["data"]["gold"] == "expected"
    out = capsys.readouterr().out
    assert "data.prompt: using column 'input'" in out
    assert "data.gold: using column 'expected'" in out
    assert len(build_tasks(EvolveSpec.from_dict(spec))) == 1


def test_init_default_columns_unchanged(ws, capsys):
    data = _cases(ws, [{"prompt": "q", "gold": "a", "input": "x"}])
    assert _init("--data", data) == 0
    spec = json.loads((ws / "spec.json").read_text())
    assert spec["data"]["prompt"] == "prompt" and spec["data"]["gold"] == "gold"
    assert "data." not in capsys.readouterr().out


def test_init_warns_on_unknown_columns(ws, capsys):
    data = _cases(ws, [{"q": "x", "a": "y"}])
    assert _init("--data", data) == 0
    spec = json.loads((ws / "spec.json").read_text())
    assert spec["data"]["prompt"] == "prompt"
    out = capsys.readouterr().out
    assert "warning: data.prompt" in out and "columns: q, a" in out
    assert "warning: data.gold" in out


def test_init_ambiguous_alias_warns(ws, capsys):
    data = _cases(ws, [{"prompt": "x", "answer": "1", "expected": "1"}])
    assert _init("--data", data) == 0
    spec = json.loads((ws / "spec.json").read_text())
    assert spec["data"]["gold"] == "gold"
    assert "several candidates (expected, answer)" in capsys.readouterr().out


def _spec(rows, **data):
    return EvolveSpec.from_dict({"version": 1, "kind": "text", "target": "p.txt",
                                 "data": {"inline": rows, **data},
                                 "score": "contains", "agent": {"ref": "echo"}})


def test_missing_prompt_names_spec_field():
    with pytest.raises(SpecError) as e:
        build_tasks(_spec([{"input": "x", "expected": "y"}]))
    msg = str(e.value)
    assert "data.prompt" in msg and "columns: expected, input" in msg
    assert '"data": {"prompt": "<column>"}' in msg


def test_missing_gold_column_refused_for_gold_scorer():
    with pytest.raises(SpecError) as e:
        build_tasks(_spec([{"input": "x", "expected": "y"}], prompt="input"))
    assert "data.gold" in str(e.value) and "'gold'" in str(e.value)


def test_some_rows_with_gold_still_ok():
    tasks = build_tasks(_spec([{"prompt": "x", "gold": "y"}, {"prompt": "z"}]))
    assert tasks[0].meta == {"gold": "y"} and tasks[1].meta == {}


def test_command_scorer_needs_no_gold():
    spec = EvolveSpec.from_dict({"version": 1, "kind": "text", "target": "p.txt",
                                 "data": {"inline": [{"prompt": "x"}]},
                                 "score": {"cmd": "echo 1"}, "agent": {"ref": "echo"}})
    assert len(build_tasks(spec)) == 1
