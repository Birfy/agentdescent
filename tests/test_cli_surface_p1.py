"""The P1 surfaces: platform, capability, ops and experience, CLI + MCP."""

import json
import os

import pytest

from agentdescent import cli
from agentdescent.durable_runs import SqliteRunStore
from agentdescent.loop_kernel import (
    LoopConfig, LoopKernel, LoopObservation,
)
from agentdescent.mcp import Tools


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _platform_spec(tmp_path, runs_db=None):
    return {
        "runs_db": str(runs_db or tmp_path / "platform.sqlite3"),
        "platform": {
            "tenant": "demo", "workspace": "default",
            "max_generations": 3, "target": 0.75,
        },
        "campaign": {"max_cycles": 2, "target": 0.75, "patience": 1},
        "compute": {
            "models": [{"name": "local", "cost": 0.0}],
            "options": [{"name": "direct", "action": "direct", "calls": 1}],
        },
        "step": "agentdescent.surface_platform:demo_step",
    }


def _capability_spec():
    return {
        "traces": [
            {"trace_id": f"t{i}", "task": f"{i} + {i + 1}", "reward": 0.0}
            for i in range(4)
        ],
        "goal": "add two integers",
        "descriptor": [0, 0],
        "pipeline": "agentdescent.surface_capability:demo_pipeline",
        "scout": "agentdescent.surface_capability:demo_scout",
        "archive": "agentdescent.surface_capability:demo_archive",
        "make_request": "agentdescent.surface_capability:demo_make_request",
        "tags": ["demo"],
    }


def _experience_record(run_id, features, reward=1.0):
    return {
        "run_id": run_id, "features": features,
        "rewards": {"edit": reward, "retry": 1.0 - reward},
        "accepted": True,
    }


def test_platform_init_plan_run_demo(tmp_path, capsys):
    spec = tmp_path / "platform.json"
    assert cli.main(["platform", "init", "--out", str(spec), "--demo",
                     "--json"]) == 0
    init = _json(capsys)
    assert init["ok"] is True and init["demo"] is True

    assert cli.main(["platform", "plan", "--spec", str(spec), "--json"]) == 0
    plan = _json(capsys)
    assert plan["ok"] is True
    assert not os.path.exists(plan["runs_db"])  # plan must not create the DB
    assert plan["campaign"]["max_cycles"] == 2
    assert plan["refs"]["step"].endswith("demo_step")

    out = tmp_path / "platform-report.json"
    assert cli.main(["platform", "run", "--spec", str(spec), "--json",
                     "--out", str(out)]) == 0
    payload = _json(capsys)
    assert payload["schema_version"] == 1
    assert payload["report"]["complete"] is True
    assert payload["report"]["cycles"]
    assert json.loads(out.read_text())["report"]["complete"] is True


def test_platform_bad_ref_is_json_error(tmp_path, capsys):
    spec = _platform_spec(tmp_path)
    spec["step"] = "nosuchmodule:step"
    path = _write(tmp_path / "bad.json", spec)
    assert cli.main(["platform", "plan", "--spec", path, "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "nosuchmodule" in payload["error"]


def test_capability_init_plan_run_demo(tmp_path, capsys):
    spec = tmp_path / "capability.json"
    assert cli.main(["capability", "init", "--out", str(spec), "--demo",
                     "--json"]) == 0
    init = _json(capsys)
    assert init["ok"] is True
    assert os.path.isfile(init["traces"])

    assert cli.main(["capability", "plan", "--spec", str(spec),
                     "--json"]) == 0
    plan = _json(capsys)
    assert plan["traces"] == 4

    assert cli.main(["capability", "run", "--spec", str(spec), "--json"]) == 0
    payload = _json(capsys)
    assert payload["schema_version"] == 1
    assert payload["report"]["admitted"] is True
    assert payload["report"]["verdict"] == "admitted"


def test_capability_missing_traces_is_json_error(tmp_path, capsys):
    spec = _capability_spec()
    spec["traces"] = str(tmp_path / "missing.jsonl")
    path = _write(tmp_path / "bad.json", spec)
    assert cli.main(["capability", "plan", "--spec", path, "--json"]) == 2
    assert _json(capsys)["ok"] is False


def _finished_loop(tmp_path):
    store_path = tmp_path / "runs.sqlite3"
    store = SqliteRunStore(str(store_path))
    kernel = LoopKernel(
        store, LoopConfig("t1", "ws1", max_generations=1))
    run_id = kernel.submit()
    kernel.run(run_id, "worker", lambda context: LoopObservation(
        0.75, cost=1.0, artifact_id="a0"))
    store.close()
    return str(store_path)


def test_ops_plan_and_publish(tmp_path, capsys):
    runs_db = _finished_loop(tmp_path)
    assert cli.main(["ops", "plan", "--runs-db", runs_db,
                     "--tenant", "t1", "--workspace", "ws1", "--json"]) == 0
    plan = _json(capsys)
    assert plan["terminal_runs"] == 1
    assert plan["generations"] == 1

    out = tmp_path / "ops.json"
    prom = tmp_path / "ops.prom"
    assert cli.main([
        "ops", "publish", "--runs-db", runs_db, "--tenant", "t1",
        "--workspace", "ws1", "--out", str(out), "--prometheus", str(prom),
        "--json"]) == 0
    payload = _json(capsys)
    assert payload["schema_version"] == 1
    assert payload["published"] == 1
    assert payload["metrics"]["decisions"] == 1
    assert json.loads(out.read_text())["published"] == 1
    assert "agentdescent_rsi_decisions_total 1" in prom.read_text()


def test_ops_missing_database_is_json_error(tmp_path, capsys):
    assert cli.main(["ops", "plan", "--runs-db", str(tmp_path / "missing.db"),
                     "--tenant", "t1", "--workspace", "ws1", "--json"]) == 2
    assert _json(capsys)["ok"] is False


def test_experience_report_ingest_recall_update_replay(tmp_path, capsys):
    store = tmp_path / "experience.jsonl"
    records = tmp_path / "records.jsonl"
    rows = [
        _experience_record("r1", [0.1, 0.2], 1.0),
        _experience_record("r2", [0.2, 0.1], 0.5),
        _experience_record("r3", [0.9, 0.1], 0.0),
    ]
    records.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                       encoding="utf-8")

    assert cli.main(["experience", "ingest", "--store", str(store),
                     "--records", str(records), "--json"]) == 0
    ingested = _json(capsys)
    assert ingested["ingested"] == ["r1", "r2", "r3"]

    assert cli.main(["experience", "report", "--store", str(store),
                     "--json"]) == 0
    report = _json(capsys)
    assert report["summary"]["rows"] == 3
    assert report["summary"]["actions"] == {"edit": 3, "retry": 3}

    assert cli.main(["experience", "recall", "--store", str(store),
                     "--features", "0.15,0.15", "--k", "2", "--json"]) == 0
    recall = _json(capsys)
    assert len(recall["hits"]) == 2

    assert cli.main([
        "experience", "update", "--store", str(store),
        "--validation", str(records), "--actions", "edit,retry",
        "--dim", "2", "--no-require-anchors", "--json"]) == 0
    update = _json(capsys)
    assert update["update"]["accepted"] is True
    assert update["update"]["version"] == 2

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps({"left": [0.1, 0.2], "right": [0.9, 0.1],
                    "preferred": "left"}) + "\n"
        + json.dumps({"left": [0.2, 0.1], "right": [0.9, 0.1],
                      "preferred": "left"}) + "\n",
        encoding="utf-8")
    replay_store = tmp_path / "replay.jsonl"
    assert cli.main([
        "experience", "replay", "--source", str(store),
        "--store", str(replay_store), "--pairs", str(pairs),
        "--actions", "edit,retry", "--dim", "2", "--k", "2", "--json"]) == 0
    replay = _json(capsys)
    assert replay["replay"]["ingested"] == ["r1", "r2"]


def test_mcp_refs_are_refused_without_the_operator_allowlist(tmp_path):
    spec = _platform_spec(tmp_path)
    tools = Tools(surface_allow=(), surface_write_dirs=(str(tmp_path),))
    payload = tools.platform(spec, dry_run=True)
    assert payload["ok"] is False
    assert "allowlist" in payload["error"]

    allowed = Tools(surface_allow=("agentdescent.",),
                    surface_write_dirs=(str(tmp_path),))
    evil = dict(spec, step="evilmod691:step")
    payload = allowed.platform(evil, dry_run=True)
    assert payload["ok"] is False
    assert "allowlist" in payload["error"]


def test_mcp_write_paths_are_confined(tmp_path):
    tools = Tools(surface_allow=("agentdescent.",),
                  surface_write_dirs=(str(tmp_path),))
    outside = str(tmp_path.parent / "agentdescent-outside.json")
    payload = tools.ops(str(tmp_path / "missing.db"), "t1", "ws1", out=outside)
    assert payload["ok"] is False
    assert "write allowlist" in payload["error"]


def test_experience_errors_are_json(tmp_path, capsys):
    store = tmp_path / "experience.jsonl"
    assert cli.main(["experience", "recall", "--store", str(store),
                     "--features", "0.1,0.2", "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "no experience record" in payload["error"]


def test_mcp_surfaces_delegate_to_the_same_payloads(tmp_path):
    tools = Tools(surface_allow=("agentdescent.",),
                  surface_write_dirs=(str(tmp_path),))
    platform_spec = _platform_spec(tmp_path)
    plan = tools.platform(platform_spec, dry_run=True)
    assert plan["ok"] is True
    run = tools.platform(platform_spec, dry_run=False)
    assert run["report"]["complete"] is True

    capability_spec = _capability_spec()
    assert tools.capability(capability_spec, dry_run=True)["ok"] is True
    assert tools.capability(capability_spec, dry_run=False)["report"]["admitted"]

    runs_db = _finished_loop(tmp_path)
    assert tools.ops(runs_db, "t1", "ws1", dry_run=True)["terminal_runs"] == 1
    published = tools.ops(runs_db, "t1", "ws1")
    assert published["published"] == 1

    store = tmp_path / "experience.jsonl"
    records = [
        _experience_record("r1", [0.1, 0.2], 1.0),
        _experience_record("r2", [0.2, 0.1], 0.5),
    ]
    ing = tools.experience_ingest(str(store), records)
    assert ing["ingested"] == ["r1", "r2"]
    assert tools.experience_report(str(store))["summary"]["rows"] == 2
    assert tools.experience_recall(str(store), [0.1, 0.2])["hits"]
    assert tools.experience_update(
        str(store), records, [], "edit,retry", 2,
        require_anchors=False)["update"]["accepted"] is True
    pairs = [{"left": [0.1, 0.2], "right": [0.9, 0.1], "preferred": "left"},
             {"left": [0.2, 0.1], "right": [0.9, 0.1], "preferred": "left"}]
    replay = tools.experience_replay(
        str(store), str(tmp_path / "replay.jsonl"), pairs, "edit,retry", 2,
        k=2)
    assert replay["replay"]["ingested"] == ["r1", "r2"]
