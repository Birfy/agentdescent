"""The WorkspaceAgent hint lists every CLI worker, opencode included (#362)."""

import json

import pytest

from agentdescent import cli
from agentdescent.evolvespec import (WORKSPACE_AGENT_REFS, EvolveSpec, SpecError,
                                     compose)

_REFS = ("claude_code", "codex", "dsh", "opencode", "cli_agent")


def _spec(tmp_path, kind, agent):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps({"prompt": f"q{i}", "gold": f"a{i}"})
                               for i in range(8)) + "\n")
    if kind == "text":
        target = tmp_path / "prompt.txt"
        target.write_text("Answer.\n")
    else:
        target = tmp_path / "skill"
        target.mkdir()
        (target / "SKILL.md").write_text("# x\n")
    return EvolveSpec.from_dict({
        "kind": kind, "target": str(target), "data": {"path": str(cases)},
        "score": "contains", "agent": agent, "reflect": "echo",
        "evolve": {"rounds": 1, "n_workers": 1, "seed": 0},
    })


def test_shared_phrase_names_every_cli_worker():
    for ref in _REFS:
        assert ref in WORKSPACE_AGENT_REFS


@pytest.mark.parametrize("kind", ["skill_dir", "agent_dir"])
def test_directory_kind_with_plain_completion_lists_opencode(tmp_path, kind):
    spec = _spec(tmp_path, kind, "echo")
    with pytest.raises(SpecError) as exc:
        compose(spec)
    msg = str(exc.value)
    assert "needs a WorkspaceAgent" in msg and "opencode" in msg
    for ref in _REFS:
        assert ref in msg


def test_plan_error_for_skill_dir_with_plain_agent_mentions_opencode(tmp_path, capsys):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_spec(tmp_path, "skill_dir", "echo").to_dict()))
    assert cli.main(["--json", "plan", str(path)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and "opencode" in out["error"]


def test_text_kind_with_cli_agent_still_lists_opencode(tmp_path):
    """Regression for #340: the reverse refusal keeps its list."""
    spec = _spec(tmp_path, "text", {"ref": "claude_code", "isolate": False})
    with pytest.raises(SpecError) as exc:
        compose(spec)
    msg = str(exc.value)
    assert "kind='text'" in msg and "opencode" in msg
    assert WORKSPACE_AGENT_REFS in msg


def test_tree_runner_hint_names_opencode():
    from agentdescent.agents import echo
    from agentdescent.runners import tree_runner

    with pytest.raises(TypeError, match="opencode"):
        tree_runner(echo(), layout="claude_skill", name="x")
