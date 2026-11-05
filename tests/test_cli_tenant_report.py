"""``agentdescent tenant report``: inspect a persistent registry."""

import hashlib
import json

from agentdescent import cli
from agentdescent.workspace_store import PersistentWorkspaceRegistry
from agentdescent.workspaces import Principal


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _spec(tmp_path, capsys, *, runs=2):
    spec_path = tmp_path / "tenant.json"
    assert cli.main(["tenant", "init", "--out", str(spec_path),
                     "--demo", "--json"]) == 0
    spec = json.loads(spec_path.read_text())
    capsys.readouterr()
    spec["registry"] = str(tmp_path / "registry.json")
    spec["quota"] = {"max_tokens": 1000, "max_cost": 10.0,
                     "max_concurrent": 1}
    spec["tokens"] = 6
    spec["cost"] = 0.5
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    for _ in range(runs):
        assert cli.main(["tenant", "run", "--spec", str(spec_path),
                         "--json"]) == 0
        capsys.readouterr()
    return spec["registry"]


def test_report_shows_accumulated_usage(tmp_path, capsys):
    registry = _spec(tmp_path, capsys)
    assert cli.main(["tenant", "report", "--registry", registry,
                     "--json"]) == 0
    report = _json(capsys)
    assert report["schema_version"] == 1 and report["ok"] is True
    assert report["counts"]["workspaces"] == 1
    row = report["workspaces"][0]
    assert row["tenant"] == "demo" and row["workspace"] == "default"
    assert row["usage"] == {"tokens": 12.0, "cost": 1.0, "active": 0}
    assert row["quota"]["max_tokens"] == 1000
    assert report["principals"][0]["principal"] == "operator"
    assert report["audit"] is None
    assert "audit:read" in report["audit_note"]


def test_report_filters_tenant_and_workspace(tmp_path, capsys):
    registry = _spec(tmp_path, capsys, runs=1)
    assert cli.main(["tenant", "report", "--registry", registry,
                     "--tenant", "other", "--json"]) == 0
    report = _json(capsys)
    assert report["workspaces"] == []
    assert report["principals"] == []

    assert cli.main(["tenant", "report", "--registry", registry,
                     "--workspace", "default", "--json"]) == 0
    assert len(_json(capsys)["workspaces"]) == 1


def test_audit_requires_audit_read(tmp_path, capsys):
    registry = _spec(tmp_path, capsys, runs=1)
    assert cli.main(["tenant", "report", "--registry", registry,
                     "--principal", "operator", "--json"]) == 0
    denied = _json(capsys)
    assert denied["audit"] is None
    assert "audit not available" in denied["audit_note"]

    store = PersistentWorkspaceRegistry(registry)
    store.add_principal(Principal(
        "auditor", "demo", {"default": ("approver",)}))
    store.add_principal(Principal(
        "viewer", "demo", {"default": ("viewer",)}))

    assert cli.main(["tenant", "report", "--registry", registry,
                     "--principal", "auditor", "--json"]) == 0
    audited = _json(capsys)
    assert audited["audit"]
    assert {event["tenant_id"] for event in audited["audit"]} == {"demo"}

    assert cli.main(["tenant", "report", "--registry", registry,
                     "--principal", "auditor", "--audit-limit", "1",
                     "--json"]) == 0
    assert len(_json(capsys)["audit"]) == 1

    assert cli.main(["tenant", "report", "--registry", registry,
                     "--principal", "viewer", "--json"]) == 0
    assert _json(capsys)["audit"] is None


def test_missing_registry_is_an_input_error(tmp_path, capsys):
    assert cli.main(["tenant", "report", "--registry",
                     str(tmp_path / "missing.json"), "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "does not exist" in payload["error"]


def test_report_does_not_modify_the_registry(tmp_path, capsys):
    registry = _spec(tmp_path, capsys, runs=1)
    with open(registry, "rb") as handle:
        before = hashlib.sha256(handle.read()).hexdigest()

    assert cli.main(["tenant", "report", "--registry", registry,
                     "--principal", "operator", "--json"]) == 0
    capsys.readouterr()
    with open(registry, "rb") as handle:
        after = hashlib.sha256(handle.read()).hexdigest()
    assert after == before
