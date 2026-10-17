"""Resume re-checks the stored plugin ``host`` CLI before relaunch (#370).

Fresh start was gated in #369; both resume entry points went straight to
``runstore.launch`` and could start a detached run whose host is absent.
"""

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout

import pytest

from agentdescent import cli, runstore
from agentdescent.evolvespec import SpecError
from agentdescent.mcp import Tools

from test_plugin_host_cli import _clis, _plugin_spec


def _stopped_run(tmp_path, **spec_kw):
    store = str(tmp_path / "runs")
    rd = runstore.create(_plugin_spec(tmp_path, **spec_kw).to_dict(), store=store)
    rd.update_status(state="cancelled", pid=2 ** 22 - 1, finished=1.0)
    return store, rd


def _snapshot(rd):
    files = {}
    for name in sorted(os.listdir(rd.path)):
        p = os.path.join(rd.path, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                files[name] = f.read()
    return files


@pytest.fixture
def launches(monkeypatch):
    calls = []

    def spy(rd, **kw):
        calls.append((rd.run_id, kw))
        return rd.update_status(state="running", pid=os.getpid())

    monkeypatch.setattr(runstore, "launch", spy)
    return calls


def test_cli_resume_refuses_absent_host_without_launch(tmp_path, monkeypatch, launches):
    _clis(monkeypatch, "opencode")
    store, rd = _stopped_run(tmp_path)
    before = _snapshot(rd)
    err = io.StringIO()
    with redirect_stdout(io.StringIO()), redirect_stderr(err):
        code = cli.main(["--store", store, "resume", rd.run_id])
    assert code == 2
    assert "host: `claude_code`" in err.getvalue() and "`claude`" in err.getvalue()
    assert launches == []
    assert _snapshot(rd) == before            # status, pid, spec, ledger untouched
    assert rd.status().state == "cancelled"


def test_cli_resume_with_present_host_reaches_launch(tmp_path, monkeypatch, launches):
    _clis(monkeypatch, "opencode", "claude")
    store, rd = _stopped_run(tmp_path)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        code = cli.main(["--store", store, "resume", rd.run_id])
    assert code == 0
    assert [c[0] for c in launches] == [rd.run_id]


def test_mcp_resume_refuses_absent_host_without_launch(tmp_path, monkeypatch, launches):
    _clis(monkeypatch, "opencode")
    store, rd = _stopped_run(tmp_path)
    before = _snapshot(rd)
    out = Tools(store).resume(rd.run_id)
    assert "host: `claude_code`" in (out.get("error") or "")
    assert "`claude`" in out["error"] and out.get("state") != "running"
    assert launches == []
    assert _snapshot(rd) == before


def test_mcp_resume_with_present_host_forwards_bridge_env(tmp_path, monkeypatch, launches):
    _clis(monkeypatch, "opencode", "claude")
    store, rd = _stopped_run(tmp_path)
    tools = Tools(store)
    monkeypatch.setattr(tools, "host_model_env", lambda: {"AGENTDESCENT_BRIDGE": "x"})
    out = tools.resume(rd.run_id)
    assert out["state"] == "running"
    assert launches[0][0] == rd.run_id
    assert launches[0][1]["env"] == {"AGENTDESCENT_BRIDGE": "x"}


def test_resume_gate_checks_only_the_host(tmp_path, monkeypatch, launches):
    """agent/reflect keep their soft resume warnings; only the host is hard."""
    _clis(monkeypatch, "claude")                 # agent `opencode` absent
    store, rd = _stopped_run(tmp_path)
    cli.require_resume_host_on_path(rd.run_id, store=store)
    assert Tools(store).resume(rd.run_id)["state"] == "running"
    assert [c[0] for c in launches] == [rd.run_id]


def test_already_running_rejection_wins_over_host_gate(tmp_path, monkeypatch, launches):
    _clis(monkeypatch, "opencode")
    store, rd = _stopped_run(tmp_path)
    rd.update_status(state="running", pid=os.getpid())
    cli.require_resume_host_on_path(rd.run_id, store=store)   # no SpecError
    out = Tools(store).resume(rd.run_id)
    assert "still running" in out["error"]
    assert launches == []


def test_budget_pair_is_still_checked_first(tmp_path, monkeypatch, launches):
    _clis(monkeypatch, "opencode")
    store, rd = _stopped_run(tmp_path)
    out = Tools(store).resume(rd.run_id, budget_usd=1.0)
    assert out.get("error") and "host" not in out["error"]
    assert launches == []


def test_require_resume_host_raises_spec_error(tmp_path, monkeypatch):
    _clis(monkeypatch, "opencode")
    store, rd = _stopped_run(tmp_path)
    with pytest.raises(SpecError, match=f"cannot resume {rd.run_id}: host"):
        cli.require_resume_host_on_path(rd.run_id, store=store)
