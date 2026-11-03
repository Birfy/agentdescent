"""The tenant surface: one quota-checked platform cycle."""

import json
import os

from agentdescent import cli
from agentdescent.mcp import Tools


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _run_cli(args):
    return cli.main(args)


def test_tenant_init_plan_run_demo(tmp_path, capsys):
    spec_path = tmp_path / "tenant.json"
    assert _run_cli(["tenant", "init", "--out", str(spec_path), "--demo",
                     "--json"]) == 0
    init = _json(capsys)
    assert init["ok"] is True and init["demo"] is True

    assert _run_cli(["tenant", "plan", "--spec", str(spec_path),
                     "--json"]) == 0
    plan = _json(capsys)
    assert plan["ok"] is True
    assert plan["tenant"] == "demo"
    assert plan["reserve"] == {"tokens": 50.0, "cost": 1.0}

    out = tmp_path / "tenant-report.json"
    assert _run_cli(["tenant", "run", "--spec", str(spec_path), "--json",
                     "--out", str(out)]) == 0
    payload = _json(capsys)
    assert payload["schema_version"] == 1
    assert payload["report"]["complete"] is True
    assert payload["report"]["usage"] == {"tokens": 50.0, "cost": 1.0,
                                          "active": 0}
    assert json.loads(out.read_text())["report"]["complete"] is True


def test_tenant_quota_exhaustion_is_an_error(tmp_path, capsys):
    spec_path = tmp_path / "tenant.json"
    _run_cli(["tenant", "init", "--out", str(spec_path), "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec["quota"] = {"max_tokens": 10, "max_cost": 1.0}
    spec["tokens"] = 50
    path = tmp_path / "quota.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert _run_cli(["tenant", "run", "--spec", str(path), "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "quota" in payload["error"]


def test_tenant_viewer_has_no_run_permission(tmp_path, capsys):
    spec_path = tmp_path / "tenant.json"
    _run_cli(["tenant", "init", "--out", str(spec_path), "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec["roles"] = ["viewer"]
    path = tmp_path / "viewer.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert _run_cli(["tenant", "plan", "--spec", str(path), "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "run:write" in payload["error"]


def test_tenant_unknown_top_level_key_is_refused(tmp_path, capsys):
    spec_path = tmp_path / "tenant.json"
    _run_cli(["tenant", "init", "--out", str(spec_path), "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec["token"] = spec.pop("tokens")
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert _run_cli(["tenant", "run", "--spec", str(path), "--json"]) == 2
    payload = _json(capsys)
    assert "unknown tenant key" in payload["error"]


def test_tenant_missing_reservation_is_refused(tmp_path, capsys):
    spec_path = tmp_path / "tenant.json"
    _run_cli(["tenant", "init", "--out", str(spec_path), "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec.pop("cost")
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert _run_cli(["tenant", "plan", "--spec", str(path), "--json"]) == 2
    assert "tokens and cost" in _json(capsys)["error"]


def test_tenant_bool_reservation_is_refused(tmp_path, capsys):
    spec_path = tmp_path / "tenant.json"
    _run_cli(["tenant", "init", "--out", str(spec_path), "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec["tokens"] = True
    path = tmp_path / "bool.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert _run_cli(["tenant", "plan", "--spec", str(path), "--json"]) == 2
    assert "bool" in _json(capsys)["error"]


def test_tenant_unknown_quota_key_is_refused(tmp_path, capsys):
    spec_path = tmp_path / "tenant.json"
    _run_cli(["tenant", "init", "--out", str(spec_path), "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec["quota"]["max_token"] = 10
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert _run_cli(["tenant", "plan", "--spec", str(path), "--json"]) == 2
    payload = _json(capsys)
    assert "quota" in payload["error"]


def test_mcp_tenant_and_write_confinement(tmp_path):
    spec_path = tmp_path / "tenant.json"
    _run_cli(["tenant", "init", "--out", str(spec_path), "--demo", "--json"])
    spec = json.loads(spec_path.read_text())
    tools = Tools(surface_allow=("agentdescent.",),
                  surface_write_dirs=(str(tmp_path),))
    assert tools.tenant(spec, dry_run=True)["ok"] is True
    run = tools.tenant(spec, dry_run=False)
    assert run["report"]["complete"] is True

    outside = str(tmp_path.parent / "tenant-outside.sqlite3")
    spec["platform"]["runs_db"] = outside
    payload = tools.tenant(spec, dry_run=True)
    assert payload["ok"] is False
    assert "write allowlist" in payload["error"]
