"""``init`` picks its default worker from the CLIs on PATH (#361).

Gated the same way as ``plan`` (#346): both read ``doctor_report()``.
"""

import json
import os

import pytest

from agentdescent import cli

_ALL = ("claude", "codex", "dsh", "opencode")


def _path_has(monkeypatch, *present):
    """Make exactly ``present`` worker CLIs look installed to doctor_report."""
    real = cli.doctor_report

    def fake():
        rep = real()
        clis = dict(rep["agent_clis"])
        for name in _ALL:
            clis[name] = f"/fake/{name}" if name in present else None
        return {**rep, "agent_clis": clis}

    monkeypatch.setattr(cli, "doctor_report", fake)


def _run_init(tmp_path, monkeypatch, capsys, *extra):
    # Hermetic: the starter reflector needs a key, and `init` warns (#395) when
    # it is missing -- these tests are about the *CLI* warnings, not that one.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    skill = tmp_path / "my-skill"
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("# x")
    assert cli.main(["init", str(skill), "--out", "spec.json", *extra]) == 0
    out = capsys.readouterr().out
    return json.loads((tmp_path / "spec.json").read_text()), out


def test_only_opencode_on_path_defaults_to_opencode(tmp_path, monkeypatch, capsys):
    _path_has(monkeypatch, "opencode")
    spec, out = _run_init(tmp_path, monkeypatch, capsys)
    assert spec["agent"] == {"ref": "opencode"}  # no Claude-only extra_args
    assert "warning" not in out
    assert "`opencode`" in out


@pytest.mark.parametrize("present,ref", [
    (("claude", "codex", "dsh", "opencode"), "claude_code"),
    (("codex", "dsh", "opencode"), "codex"),
    (("dsh", "opencode"), "dsh"),
    (("opencode",), "opencode"),
])
def test_priority_when_several_are_on_path(tmp_path, monkeypatch, capsys, present, ref):
    _path_has(monkeypatch, *present)
    spec, _ = _run_init(tmp_path, monkeypatch, capsys)
    assert spec["agent"]["ref"] == ref


def test_claude_keeps_its_permission_flag(tmp_path, monkeypatch, capsys):
    _path_has(monkeypatch, "claude", "opencode")
    spec, _ = _run_init(tmp_path, monkeypatch, capsys)
    assert spec["agent"] == {"ref": "claude_code",
                             "extra_args": ["--permission-mode", "acceptEdits"]}


def test_no_cli_on_path_keeps_placeholder_and_warns(tmp_path, monkeypatch, capsys):
    _path_has(monkeypatch)
    spec, out = _run_init(tmp_path, monkeypatch, capsys)
    assert spec["agent"]["ref"] == "claude_code"
    warning = [ln for ln in out.splitlines() if ln.startswith("warning:")]
    assert len(warning) == 1
    assert "doctor" in warning[0] and "--agent" in warning[0]


def test_explicit_agent_overrides_path_detection(tmp_path, monkeypatch, capsys):
    _path_has(monkeypatch, "opencode")
    spec, out = _run_init(tmp_path, monkeypatch, capsys, "--agent", "claude_code")
    assert spec["agent"]["ref"] == "claude_code"
    assert "warning" not in out  # the user chose; plan will say what is missing


def test_text_kind_never_warns_about_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _path_has(monkeypatch)
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "p.md"
    prompt.write_text("be terse")
    assert cli.main(["init", str(prompt), "--out", "spec.json"]) == 0
    assert "warning" not in capsys.readouterr().out
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["agent"]["ref"] == "openai_compatible"


def test_starter_spec_default_follows_path(tmp_path, monkeypatch):
    _path_has(monkeypatch, "codex")
    skill = tmp_path / "s"
    skill.mkdir()
    assert cli.starter_spec(str(skill))["agent"] == {"ref": "codex"}
    assert cli.starter_spec(str(skill), agent="dsh")["agent"] == {"ref": "dsh"}


def test_init_then_plan_agrees_on_the_cli_gate(tmp_path, monkeypatch, capsys):
    """Only opencode on PATH: the spec init writes is not refused for a missing claude."""
    _path_has(monkeypatch, "opencode")
    spec, _ = _run_init(tmp_path, monkeypatch, capsys)
    from agentdescent.evolvespec import load_spec

    assert cli._missing_cli_messages(load_spec(str(tmp_path / "spec.json"))) == []
    assert os.path.exists(tmp_path / "spec.json") and spec["agent"]["ref"] == "opencode"
