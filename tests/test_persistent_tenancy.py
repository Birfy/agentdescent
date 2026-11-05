"""Tenant cycles can accumulate quota and audit in a registry file."""

import json

from agentdescent import cli
from agentdescent.mcp import Tools
from agentdescent.workspace_store import PersistentWorkspaceRegistry


def _spec(tmp_path):
    spec_path = tmp_path / "tenant.json"
    cli.main(["tenant", "init", "--out", str(spec_path), "--demo", "--json"])
    spec = json.loads(spec_path.read_text())
    spec["registry"] = str(tmp_path / "registry.json")
    spec["quota"] = {"max_tokens": 10, "max_cost": 10.0,
                     "max_concurrent": 1}
    spec["tokens"] = 6
    spec["cost"] = 0.0
    return spec


def test_second_cycle_hits_persisted_quota(tmp_path, capsys):
    spec = _spec(tmp_path)
    path = tmp_path / "tenant.json"
    path.write_text(json.dumps(spec), encoding="utf-8")

    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "quota" in payload["error"]

    registry = PersistentWorkspaceRegistry(spec["registry"])
    assert registry.usage("operator", "demo", "default").tokens == 6
    assert registry.snapshot()["audit"]


def test_changed_quota_is_refused_not_ignored(tmp_path, capsys):
    spec = _spec(tmp_path)
    path = tmp_path / "tenant.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 0
    capsys.readouterr()

    spec["quota"]["max_tokens"] = 100
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "registry has quota" in payload["error"]


def test_demoted_role_is_refused(tmp_path, capsys):
    spec = _spec(tmp_path)
    path = tmp_path / "tenant.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 0
    capsys.readouterr()

    spec["roles"] = ["viewer"]
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli.main(["tenant", "plan", "--spec", str(path), "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "registry has roles" in payload["error"]


def test_mcp_registry_write_is_confined(tmp_path):
    spec = _spec(tmp_path)
    tools = Tools(surface_allow=("agentdescent.",),
                  surface_write_dirs=(str(tmp_path),))
    first = tools.tenant(spec, dry_run=False)
    assert first["report"]["complete"] is True
    second = tools.tenant(spec, dry_run=False)
    assert second["ok"] is False
    assert "quota" in second["error"]

    outside = str(tmp_path.parent / "registry-outside.json")
    spec["registry"] = outside
    denied = tools.tenant(spec, dry_run=True)
    assert denied["ok"] is False
    assert "write allowlist" in denied["error"]


def test_new_principal_on_existing_workspace_checks_the_persisted_quota(
        tmp_path, capsys):
    spec = _spec(tmp_path)
    path = tmp_path / "tenant.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 0
    capsys.readouterr()

    spec["principal"] = "operator2"
    spec["quota"]["max_tokens"] = 100
    path.write_text(json.dumps(spec), encoding="utf-8")
    for action in ("plan", "run"):
        assert cli.main(["tenant", action, "--spec", str(path),
                         "--json"]) == 2
        payload = json.loads(capsys.readouterr().out)
        assert "registry has quota" in payload["error"]


def test_new_principal_on_existing_workspace_sees_persisted_usage(
        tmp_path, capsys):
    spec = _spec(tmp_path)
    spec["quota"] = {"max_tokens": 100, "max_cost": 10.0,
                     "max_concurrent": 1}
    path = tmp_path / "tenant.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 0
    capsys.readouterr()

    spec["principal"] = "operator2"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli.main(["tenant", "plan", "--spec", str(path), "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["usage"] == {"tokens": 6.0, "cost": 0.0, "active": 0}

    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 0
    capsys.readouterr()
    registry = PersistentWorkspaceRegistry(spec["registry"])
    assert registry.usage("operator2", "demo", "default").tokens == 12.0
    assert registry.get_principal("operator2").roles_for("default") == (
        "operator",)


def test_existing_principal_on_a_new_workspace_checks_roles(tmp_path, capsys):
    spec = _spec(tmp_path)
    path = tmp_path / "tenant.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert cli.main(["tenant", "run", "--spec", str(path), "--json"]) == 0
    capsys.readouterr()

    spec["workspace"] = "other"
    spec["platform"]["platform"]["workspace"] = "other"
    path.write_text(json.dumps(spec), encoding="utf-8")
    for action in ("plan", "run"):
        assert cli.main(["tenant", action, "--spec", str(path),
                         "--json"]) == 2
        payload = json.loads(capsys.readouterr().out)
        assert "registry has roles" in payload["error"]
