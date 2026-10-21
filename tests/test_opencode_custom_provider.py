"""An isolated OpenCode worker cannot see a custom provider; say so (#373).

A custom provider (an OpenAI-compatible endpoint such as Ark, a local vLLM, a
gateway) lives in the user's `opencode.json`. An isolated worker has its config
redirected into the workspace, so `-m ark/model` died with OpenCode's opaque
`UnknownError`, `plan` was `ok: true`, and the docs said `-m` pins a model
"without giving up isolation".
"""
import json
import os
import stat
import sys

import pytest

from agentdescent import cli
from agentdescent.agents import AgentError, cli_agent, opencode
from agentdescent.evolvespec import EvolveSpec

_UNKNOWN_ERROR = json.dumps({
    "name": "UnknownError",
    "data": {"message": "Unexpected server error. Check server logs for details.",
             "ref": "err_ecf57953"}}, indent=2)


def _stub_opencode(tmp_path, body):
    """A fake `opencode` on a private PATH dir; returns the dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "opencode"
    exe.write_text(f"#!{sys.executable}\n{body}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(bin_dir)


_FAILS = ("import sys\n"
          f"sys.stderr.write('Error: ' + {_UNKNOWN_ERROR!r} + '\\n')\n"
          "sys.exit(1)")


@pytest.fixture
def failing_opencode(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", _stub_opencode(tmp_path, _FAILS) + os.pathsep
                       + os.environ["PATH"])
    ws = tmp_path / "ws"
    ws.mkdir()
    return str(ws)


@pytest.mark.skipif(os.name == "nt", reason="shebang stub")
def test_isolated_opencode_unknown_error_names_the_cause_and_both_fixes(failing_opencode):
    agent = opencode(workspace=failing_opencode, extra_args=["-m", "ark/deepseek-v4-flash"])
    with pytest.raises(AgentError) as exc:
        agent("hi")
    msg = str(exc.value)
    assert "UnknownError" in msg                      # the original detail is kept
    assert "isolate" in msg and "OPENCODE_CONFIG" in msg
    assert "providers.json" in msg and "{env:VAR}" in msg


@pytest.mark.skipif(os.name == "nt", reason="shebang stub")
def test_non_isolated_opencode_keeps_its_plain_message(failing_opencode):
    agent = opencode(workspace=failing_opencode, isolate=False,
                     extra_args=["-m", "ark/deepseek-v4-flash"])
    with pytest.raises(AgentError) as exc:
        agent("hi")
    assert "UnknownError" in str(exc.value)
    assert "hint" not in str(exc.value) and "OPENCODE_CONFIG" not in str(exc.value)


@pytest.mark.skipif(os.name == "nt", reason="shebang stub")
def test_no_hint_when_the_caller_already_set_opencode_config(failing_opencode):
    agent = opencode(workspace=failing_opencode, env={"OPENCODE_CONFIG": "/x/p.json"})
    with pytest.raises(AgentError) as exc:
        agent("hi")
    assert "hint" not in str(exc.value)


@pytest.mark.skipif(os.name == "nt", reason="shebang stub")
def test_an_unrelated_opencode_failure_gets_no_hint(tmp_path, monkeypatch):
    bin_dir = _stub_opencode(tmp_path, "import sys\nsys.stderr.write('boom')\nsys.exit(2)")
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ["PATH"])
    with pytest.raises(AgentError) as exc:
        opencode(workspace=str(tmp_path))("hi")
    assert str(exc.value) == "opencode exited 2: boom"


def test_other_clis_keep_their_message(tmp_path):
    cmd = [sys.executable, "-c",
           f"import sys; sys.stderr.write({_UNKNOWN_ERROR!r}); sys.exit(1)"]
    with pytest.raises(AgentError) as exc:
        cli_agent(cmd, workspace=str(tmp_path))("hi")
    assert "UnknownError" in str(exc.value)
    assert "OPENCODE_CONFIG" not in str(exc.value) and "hint" not in str(exc.value)


# -- plan -------------------------------------------------------------------


def _spec(tmp_path, agent):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"prompt": "q", "gold": "a"}\n' * 6, encoding="utf-8")
    skill = tmp_path / "skill"
    skill.mkdir(exist_ok=True)
    (skill / "rules.md").write_text("MODE: forward\n", encoding="utf-8")
    return EvolveSpec.from_dict({
        "kind": "skill_dir", "target": str(skill),
        "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
        "score": "exact", "agent": agent, "reflect": "echo",
        "prompt_template": "{prompt}",
        "evolve": {"rounds": 1, "n_workers": 1, "seed": 0}})


@pytest.fixture
def opencode_on_path(monkeypatch):
    real = cli.doctor_report

    def fake():
        rep = real()
        return {**rep, "agent_clis": {**rep["agent_clis"], "opencode": "/fake/opencode"}}

    monkeypatch.setattr(cli, "doctor_report", fake)


def _warnings(tmp_path, agent):
    payload = cli.plan_payload(_spec(tmp_path, agent))
    assert payload["ok"] is True, payload
    return [w for w in payload["warnings"] if "opencode" in w]


def test_plan_warns_that_an_isolated_model_pin_cannot_see_custom_providers(
        tmp_path, opencode_on_path):
    warns = _warnings(tmp_path, {"ref": "opencode", "extra_args": ["-m", "x/y"]})
    assert len(warns) == 1, warns
    w = warns[0]
    assert "`-m x/y`" in w and "custom provider" in w and "UnknownError" in w
    assert "OPENCODE_CONFIG" in w and '"isolate": false' in w


@pytest.mark.parametrize("args", [["--model", "x/y"], ["--model=x/y"], ["-m", "x/y", "--print-logs"]])
def test_plan_recognises_the_model_flag_spellings(tmp_path, opencode_on_path, args):
    warns = _warnings(tmp_path, {"ref": "opencode", "extra_args": args})
    assert any("custom provider" in w for w in warns), warns


@pytest.mark.parametrize("agent", [
    {"ref": "opencode"},                                              # no pin
    {"ref": "opencode", "extra_args": ["-m", "justamodel"]},          # no provider part
])
def test_plan_keeps_the_generic_warning_when_there_is_nothing_to_add(
        tmp_path, opencode_on_path, agent):
    warns = _warnings(tmp_path, agent)
    assert warns and not any("custom provider" in w for w in warns), warns


def test_plan_is_quiet_about_providers_when_not_isolated(tmp_path, opencode_on_path):
    assert _warnings(tmp_path, {"ref": "opencode", "extra_args": ["-m", "x/y"],
                                "isolate": False}) == []


# -- plan: provider key without a pinned model, and the configured recipe (#455)


def _with_keys(monkeypatch, **present):
    real = cli.doctor_report

    def fake():
        rep = real()
        keys = {k: False for k in rep["provider_keys_present"]}
        keys.update(present)
        return {**rep, "provider_keys_present": keys,
                "agent_clis": {**rep["agent_clis"], "opencode": "/fake/opencode"}}

    monkeypatch.setattr(cli, "doctor_report", fake)


def test_plan_says_a_gateway_key_alone_leaves_opencode_on_its_default_model(
        tmp_path, monkeypatch):
    _with_keys(monkeypatch, ARK_API_KEY=True)
    warns = _warnings(tmp_path, {"ref": "opencode"})
    assert len(warns) == 1, warns
    w = warns[0]
    assert "untested" not in w
    assert "ARK_API_KEY" in w and "default model" in w
    assert '"-m", "ark/deepseek-v4-flash"' in w and "OPENCODE_CONFIG" in w


def test_plan_keeps_the_generic_warning_without_a_gateway_key(tmp_path, monkeypatch):
    _with_keys(monkeypatch, OPENAI_API_KEY=True)
    warns = _warnings(tmp_path, {"ref": "opencode"})
    assert warns and not any("default model" in w for w in warns), warns


@pytest.mark.parametrize("env", [{"OPENCODE_CONFIG": "/p.json"},
                                 {"OPENCODE_CONFIG_CONTENT": "{}"}])
def test_plan_is_quiet_about_the_documented_isolated_recipe(tmp_path, monkeypatch, env):
    _with_keys(monkeypatch, ARK_API_KEY=True)
    agent = {"ref": "opencode", "extra_args": ["-m", "ark/deepseek-v4-flash"], "env": env}
    assert _warnings(tmp_path, agent) == []
