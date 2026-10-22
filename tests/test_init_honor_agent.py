"""``init --agent`` is honored or refused, never silently swapped (#456)."""

import json

import pytest

from agentdescent import cli


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompt.txt").write_text("Answer.")
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# x")
    return tmp_path


def _init(*argv):
    return cli.main(["init", *argv, "--out", "spec.json"])


@pytest.mark.parametrize("ref", ["host_model", "claude"])
def test_text_keeps_explicit_plain_agent(ws, ref):
    assert _init("prompt.txt", "--kind", "text", "--agent", ref) == 0
    spec = json.loads((ws / "spec.json").read_text())
    assert spec["agent"] == {"ref": ref}


def test_text_default_unchanged_without_agent(ws):
    assert _init("prompt.txt", "--kind", "text") == 0
    spec = json.loads((ws / "spec.json").read_text())
    # Without --agent, kind=text keeps its default: the key-aware model block
    # init writes (#451), here with OPENAI_API_KEY set and no --model.
    assert spec["agent"] == {"ref": "openai_compatible", "model": cli.PLACEHOLDER_MODEL}


def test_text_explicit_openai_compatible_keeps_model(ws):
    assert _init("prompt.txt", "--kind", "text", "--agent", "openai_compatible") == 0
    spec = json.loads((ws / "spec.json").read_text())
    assert spec["agent"]["ref"] == "openai_compatible"
    assert spec["agent"]["model"]


@pytest.mark.parametrize("ref", ["opencode", "claude_code", "agentdescent.agents:codex"])
def test_text_refuses_workspace_agent(ws, capsys, ref):
    assert _init("prompt.txt", "--kind", "text", "--agent", ref) == 2
    assert not (ws / "spec.json").exists()
    err = capsys.readouterr().err
    assert "kind='text'" in err and ref in err


@pytest.mark.parametrize("ref", ["host_model", "openai_compatible"])
def test_skill_dir_refuses_plain_agent(ws, capsys, ref):
    assert _init(str(ws / "my-skill"), "--agent", ref) == 2
    assert not (ws / "spec.json").exists()
    assert "WorkspaceAgent" in capsys.readouterr().err


def test_skill_dir_keeps_workspace_agent(ws):
    assert _init(str(ws / "my-skill"), "--agent", "opencode") == 0
    assert json.loads((ws / "spec.json").read_text())["agent"] == {"ref": "opencode"}


def test_custom_ref_is_not_second_guessed(ws):
    assert _init("prompt.txt", "--kind", "text", "--agent", "mypkg.models:mine") == 0
    assert json.loads((ws / "spec.json").read_text())["agent"] == {"ref": "mypkg.models:mine"}


def test_policy_slot_keeps_explicit_agent(ws):
    assert _init("selection", "--kind", "policy_slot", "--agent", "host_model") == 0
    assert json.loads((ws / "spec.json").read_text())["agent"] == {"ref": "host_model"}
