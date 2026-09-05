"""The MCP server: the tool bodies without the SDK, and the wiring with it."""

import json
import os
import time

import pytest

from agentdescent import runstore
from agentdescent.mcp import TOOL_DESCRIPTIONS, Tools, build_server
from agentdescent.cli import NESTED_ENV

from tests.test_evolvespec import _dir_spec  # noqa: F401


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    return str(tmp_path / "runs")


def _wait(rd, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = rd.status()
        if st.state in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.2)
    raise AssertionError(rd.status())


# ---------------------------------------------------------------------------
# tool bodies, no SDK
# ---------------------------------------------------------------------------


def test_plan_reports_a_bad_spec_by_field(store, tmp_path):
    t = Tools(store)
    spec = _dir_spec(str(tmp_path)).to_dict()
    ok = t.plan(spec)
    assert ok["ok"] and ok["tasks"] == 12 and ok["estimate"]["agent_calls_upper_bound"] > 0
    spec["score"] = "sideways"
    bad = t.plan(spec)
    assert bad["ok"] is False and "score" in bad["error"]
    assert "ok" in t.plan({"kind": "text"}) and t.plan({"kind": "text"})["ok"] is False


def test_start_status_show_apply_flow(store, tmp_path):
    t = Tools(store)
    spec = _dir_spec(str(tmp_path)).to_dict()
    started = t.start(spec)
    assert started["ok"] and started["run_id"] and started["state"] == "running"
    rd = runstore.get(started["run_id"], store=store)
    _wait(rd)
    st = t.status(started["run_id"])
    assert st["state"] == "done" and st["recent_rounds"]
    assert t.status()[0]["run_id"] == started["run_id"]
    shown = t.show(started["run_id"])
    assert shown["final_reward"] == 1.0 and "+MODE: reverse" in shown["diff"]
    assert shown["apply_plan"]["written"] == ["rules.md"]
    plan = t.apply(started["run_id"], dry_run=True)
    assert plan["written"] == ["rules.md"]
    done = t.apply(started["run_id"])
    assert done["backup"]
    with open(os.path.join(spec["target"], "rules.md")) as fh:
        assert "MODE: reverse" in fh.read()


def test_start_refuses_a_bad_spec_without_creating_a_run(store, tmp_path):
    t = Tools(store)
    spec = _dir_spec(str(tmp_path)).to_dict()
    spec["agent"] = "echo"
    out = t.start(spec)
    assert out["ok"] is False and "WorkspaceAgent" in out["error"]
    assert runstore.list_runs(store=store) == []


def test_start_is_a_stub_inside_a_worker(store, tmp_path, monkeypatch):
    monkeypatch.setenv(NESTED_ENV, "1")
    t = Tools(store)
    out = t.start(_dir_spec(str(tmp_path)).to_dict())
    assert out["ok"] and out["nested"] and out["run_id"] == "nested-stub"
    assert runstore.list_runs(store=store) == []


def test_unknown_run_ids_are_errors_not_exceptions(store):
    t = Tools(store)
    assert "error" in t.status("nope")
    assert "error" in t.show("nope")
    assert "error" in t.apply("nope")
    assert "error" in t.cancel("nope")
    assert "error" in t.resume("nope")


def test_resources_are_json(store, tmp_path):
    t = Tools(store)
    assert json.loads(t.runs_resource()) == []
    assert "error" in json.loads(t.rounds_resource("nope"))


def test_every_tool_has_a_description_written_for_the_model():
    for name, text in TOOL_DESCRIPTIONS.items():
        assert len(text) > 60, name
    assert "DESTRUCTIVE" in TOOL_DESCRIPTIONS["apply"]
    assert "FIRST" in TOOL_DESCRIPTIONS["doctor"]
    assert "WITHOUT running" in TOOL_DESCRIPTIONS["plan"]


# ---------------------------------------------------------------------------
# with the SDK
# ---------------------------------------------------------------------------


def test_server_registers_every_tool_and_both_resources(store):
    pytest.importorskip("mcp")
    import asyncio

    server = build_server(store)

    async def go():
        tools = {t.name for t in await server.list_tools()}
        assert tools == set(TOOL_DESCRIPTIONS)
        uris = {str(r.uri) for r in await server.list_resources()}
        assert "agentdescent://runs" in uris
        templates = await server.list_resource_templates()
        assert any("rounds" in str(t.uri_template if hasattr(t, "uri_template") else t.uriTemplate)
                   for t in templates)
        res = await server.call_tool("status", {})
        return res

    res = asyncio.run(go())
    assert res is not None


def test_serve_without_the_sdk_says_how_to_get_it(monkeypatch):
    import builtins
    import agentdescent.mcp as m

    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("mcp"):
            raise ImportError("no mcp")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"agentdescent\[mcp\]"):
        m._server_class()
