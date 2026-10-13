"""Policy-slot onboarding uses problem refs rather than row-data paths."""

import json

import pytest

from agentdescent import cli
from agentdescent.evolvespec import load_spec


@pytest.mark.parametrize("problems", [None, "custom.problems:build"])
def test_init_policy_slot_writes_spec_and_explains_problem_ref(
    tmp_path, monkeypatch, capsys, problems
):
    monkeypatch.chdir(tmp_path)
    args = ["init", "selection", "--kind", "policy_slot"]
    if problems is not None:
        args.extend(["--data", problems])
    assert cli.main(args) == 0

    spec_path = tmp_path / ".agentdescent" / "selection.evolve.json"
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    expected = problems or "mypkg.problems:build"
    assert payload["kind"] == "policy_slot"
    assert payload["target"] == "selection"
    assert payload["data"] == {"problems": expected, "seeds": [0]}
    assert payload["allow"] == ["mypkg"]  # init does not expand import permissions
    assert load_spec(str(spec_path)).target == "selection"

    out = capsys.readouterr().out
    assert expected in out
    assert "zero-argument builder returning a mapping of inner problems" in out
    assert "importable" in out and "allow list" in out
    assert "then: agentdescent plan .agentdescent/selection.evolve.json" in out
    assert "JSON object per line" not in out
