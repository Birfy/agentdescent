"""The governed-harness surface: safety, proof, approval, canary and promotion."""

import json
import os

import pytest

from agentdescent import cli
from agentdescent.approval import ApprovalStore
from agentdescent.mcp import Tools
from agentdescent.proofs import ProofInvariant, ProofRunner


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def failing_prover():
    return ProofRunner([ProofInvariant("never", lambda artifact: False)])


def test_governed_init_plan_run_demo(tmp_path, capsys):
    spec = tmp_path / "governed.json"
    assert cli.main(["governed-harness", "init", "--out", str(spec),
                     "--demo", "--json"]) == 0
    init = _json(capsys)
    assert init["ok"] is True
    assert os.path.isfile(init["before"]) and os.path.isfile(init["after"])

    assert cli.main(["governed-harness", "plan", "--spec", str(spec),
                     "--json"]) == 0
    plan = _json(capsys)
    assert plan["ok"] is True
    assert plan["tasks"] == 20
    assert plan["gate"] == "SafeSelfModifier"

    out = tmp_path / "governed-report.json"
    assert cli.main(["governed-harness", "run", "--spec", str(spec),
                     "--json", "--out", str(out)]) == 0
    payload = _json(capsys)
    assert payload["schema_version"] == 1
    assert payload["promoted"] is True
    assert payload["report"]["verdict"] == "promote"
    assert payload["report"]["rollback"]["write"]
    assert json.loads(out.read_text())["promoted"] is True


def test_failed_proof_never_reaches_promotion(tmp_path, capsys):
    spec_path = tmp_path / "governed.json"
    cli.main(["governed-harness", "init", "--out", str(spec_path),
              "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec["prover"] = "tests.test_cli_governed:failing_prover"
    path = _write(tmp_path / "failing.json", spec)
    assert cli.main(["governed-harness", "run", "--spec", path,
                     "--json"]) == 1
    payload = _json(capsys)
    assert payload["promoted"] is False
    assert payload["report"]["verdict"] == "reject"
    assert payload["report"]["evidence"] is None


def test_pending_approval_defers_and_approved_release_promotes(tmp_path, capsys):
    spec_path = tmp_path / "governed.json"
    cli.main(["governed-harness", "init", "--out", str(spec_path),
              "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    approvals = str(tmp_path / "approvals.jsonl")
    spec["approval_store"] = approvals
    path = _write(tmp_path / "approval.json", spec)

    assert cli.main(["governed-harness", "run", "--spec", path,
                     "--json"]) == 1
    pending = _json(capsys)
    assert pending["promoted"] is False
    assert pending["report"]["verdict"] == "needs-approval"
    key = pending["report"]["approval"]["key"]
    ApprovalStore(approvals).decide(key, approve=True)

    assert cli.main(["governed-harness", "run", "--spec", path,
                     "--json"]) == 0
    approved = _json(capsys)
    assert approved["promoted"] is True


def test_conformal_reject_is_not_promoted_by_default(tmp_path, capsys):
    spec_path = tmp_path / "governed.json"
    cli.main(["governed-harness", "init", "--out", str(spec_path),
              "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec["before"] = _write(tmp_path / "before.json",
                            {f"t{i}": 1.0 for i in range(30)})
    spec["after"] = _write(tmp_path / "after.json",
                           {f"t{i}": (2.0 if i < 25 else 0.0)
                            for i in range(30)})
    path = _write(tmp_path / "conformal.json", spec)

    assert cli.main(["governed-harness", "run", "--spec", path,
                     "--json"]) == 1
    payload = _json(capsys)
    assert payload["promoted"] is False
    assert "conformal" in " ".join(payload["report"]["decision"]["reasons"])


def test_mcp_governed_harness_and_allowlist(tmp_path):
    spec_path = tmp_path / "governed.json"
    cli.main(["governed-harness", "init", "--out", str(spec_path),
              "--demo", "--json"])
    spec = json.loads(spec_path.read_text())
    tools = Tools(surface_allow=("agentdescent.", "tests."),
                  surface_write_dirs=(str(tmp_path),))
    plan = tools.governed_harness(spec, dry_run=True)
    assert plan["ok"] is True
    run = tools.governed_harness(spec, dry_run=False)
    assert run["promoted"] is True

    denied = Tools(surface_allow=())
    payload = denied.governed_harness(
        dict(spec, prover="tests.test_cli_governed:failing_prover"),
        dry_run=True)
    assert payload["ok"] is False
    assert "allowlist" in payload["error"]


def test_mcp_governed_harness_write_is_confined(tmp_path):
    spec_path = tmp_path / "governed.json"
    cli.main(["governed-harness", "init", "--out", str(spec_path),
              "--demo", "--json"])
    spec = json.loads(spec_path.read_text())
    tools = Tools(surface_allow=("agentdescent.",),
                  surface_write_dirs=(str(tmp_path),))
    outside = str(tmp_path.parent / "governed-outside.json")
    payload = tools.governed_harness(spec, out=outside)
    assert payload["ok"] is False
    assert "write allowlist" in payload["error"]

    spec["approval_store"] = str(tmp_path.parent / "approvals-outside.jsonl")
    payload = tools.governed_harness(spec, dry_run=True)
    assert payload["ok"] is False
    assert "write allowlist" in payload["error"]
