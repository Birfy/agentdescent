"""The plugin ``host`` CLI is gated on PATH separately from agent/reflect (#368)."""

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout

import pytest

from agentdescent import cli, runstore
from agentdescent.evolvespec import EvolveSpec, SpecError
from agentdescent.mcp import NESTED_ENV, Tools

_ALL = ("claude", "codex", "dsh", "opencode")


def _clis(monkeypatch, *present):
    real = cli.doctor_report

    def fake():
        rep = real()
        clis = dict(rep["agent_clis"])
        for name in _ALL:
            clis[name] = f"/fake/{name}" if name in present else None
        return {**rep, "agent_clis": clis}

    monkeypatch.setattr(cli, "doctor_report", fake)


def _plugin_spec(tmp_path, *, host="claude_code", agent="opencode",
                 reflect="echo") -> EvolveSpec:
    plugin = tmp_path / "plug"
    (plugin / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text('{"name": "plug"}')
    (plugin / "skills" / "s").mkdir(parents=True, exist_ok=True)
    (plugin / "skills" / "s" / "SKILL.md").write_text("# s\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps({"prompt": f"q{i}", "gold": "a"}) + "\n"
                             for i in range(6)))
    return EvolveSpec.from_dict({
        "kind": "plugin", "host": host, "target": str(plugin),
        "data": {"path": str(cases), "prompt": "prompt", "gold": "gold"},
        "score": "contains", "agent": {"ref": agent}, "reflect": reflect,
        "evolve": {"rounds": 1, "n_workers": 1, "seed": 0},
    })


def test_installed_agent_does_not_make_absent_host_pass(tmp_path, monkeypatch):
    _clis(monkeypatch, "opencode")
    spec = _plugin_spec(tmp_path)
    msgs = cli._missing_cli_messages(spec)
    assert len(msgs) == 1 and msgs[0].startswith("host: `claude_code`")
    assert "`claude`" in msgs[0] and "on PATH" in msgs[0]
    payload = cli.plan_payload(spec)
    assert payload["ok"] is False
    assert "host" in payload["error"] and "`claude`" in payload["error"]
    with pytest.raises(SpecError, match="host: `claude_code`"):
        cli.require_cli_on_path(spec)


def test_present_host_passes(tmp_path, monkeypatch):
    _clis(monkeypatch, "opencode", "claude")
    spec = _plugin_spec(tmp_path)
    assert cli._missing_cli_messages(spec) == []
    cli.require_cli_on_path(spec)


def test_roles_are_reported_separately(tmp_path, monkeypatch):
    """Agent, reflector and host each name their own missing binary; the
    existing agent/reflect checks are preserved alongside the new one."""
    _clis(monkeypatch)
    spec = _plugin_spec(tmp_path, host="dsh", agent="codex", reflect="opencode")
    msgs = cli._missing_cli_messages(spec)
    assert [m.split(":", 1)[0] for m in msgs] == ["agent", "reflect", "host"]
    assert "`codex`" in msgs[0] and "`opencode`" in msgs[1] and "`dsh`" in msgs[2]


def test_host_is_only_checked_for_plugins(tmp_path, monkeypatch):
    _clis(monkeypatch, "opencode")
    spec = _plugin_spec(tmp_path)
    from dataclasses import replace
    assert cli._missing_cli_messages(replace(spec, kind="skill_dir")) == []


def test_cli_evolve_refuses_absent_host_before_create(tmp_path, monkeypatch):
    _clis(monkeypatch, "opencode")
    store = str(tmp_path / "runs")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_plugin_spec(tmp_path).to_dict()))
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        code = cli.main(["--store", store, "--json", "evolve", str(path), "--detach"])
    assert code == 2
    assert "host" in err.getvalue() and "`claude`" in err.getvalue()
    assert not os.path.isdir(store) or os.listdir(store) == []


def _no_runs(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("create/launch must not run")

    monkeypatch.setattr(runstore, "create", boom)
    monkeypatch.setattr(runstore, "launch", boom)


def test_mcp_start_refuses_absent_host_before_create(tmp_path, monkeypatch):
    _clis(monkeypatch, "opencode")
    monkeypatch.delenv(NESTED_ENV, raising=False)
    _no_runs(monkeypatch)
    out = Tools(str(tmp_path / "runs")).start(_plugin_spec(tmp_path).to_dict())
    assert out["ok"] is False and "host: `claude_code`" in out["error"]


def test_nested_mcp_start_keeps_stub_with_absent_host(tmp_path, monkeypatch):
    """#352 contract: nested start returns the refused stub, not ok:false."""
    _clis(monkeypatch, "opencode")
    monkeypatch.setenv(NESTED_ENV, "1")
    _no_runs(monkeypatch)
    out = Tools(str(tmp_path / "runs")).start(_plugin_spec(tmp_path).to_dict())
    assert out["ok"] is True and out["run_id"] == "nested-stub"
    assert out["state"] == "refused"


def test_init_warns_when_plugin_host_is_absent(tmp_path, monkeypatch, capsys):
    _clis(monkeypatch, "opencode")
    plugin = tmp_path / "plug"
    (plugin / ".claude-plugin").mkdir(parents=True)
    out = tmp_path / "spec.json"
    assert cli.main(["init", str(plugin), "--out", str(out)]) == 0
    spec = json.loads(out.read_text())
    assert spec["host"] == "claude_code" and spec["agent"]["ref"] == "opencode"
    text = capsys.readouterr().out
    assert "host `claude_code` needs `claude`" in text
