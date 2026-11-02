"""A harness change must use one governed path, not hand-wired gates."""

from __future__ import annotations

import json

import pytest

from agentdescent.approval import ApprovalStore
from agentdescent.governed_harness import (
    GovernedHarnessReport,
    govern_harness_change,
)
from agentdescent.promotion import PromotionRules
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.safe_selfmod import ModificationSpec, SafeSelfModifier


def _scores(gain=0.5, n=50, base=0.5):
    before = {f"t{i}": base for i in range(n)}
    after = {f"t{i}": base + gain for i in range(n)}
    return before, after


def _spec(source="def f(x):\n    return x + 1\n", *, risk=0.1, mid="m1"):
    return ModificationSpec(mid, source, "harness/f.py", (), risk)


def _gate(**kwargs):
    values = {"max_risk": 0.5, "canary_risk": 0.2}
    values.update(kwargs)
    return SafeSelfModifier(set(), **values)


def _prover(*, passes=True):
    return ProofRunner([
        ProofInvariant("always", lambda _mod: passes),
    ], issuer="test")


def _approval(tmp_path, key="h1", *, decision=None):
    store = ApprovalStore(str(tmp_path / "approvals.jsonl"))
    store.request(key, "harness-1", "review")
    if decision is not None:
        store.decide(key, approve=decision)
    return store


def test_low_risk_proof_and_approval_promote_with_rollback_plan(tmp_path):
    before, after = _scores()
    store = _approval(tmp_path, decision=True)
    report = govern_harness_change(
        "harness-1", _spec(), before, after, gate=_gate(),
        prover=_prover(), approval=store, approval_key="h1",
        rollback_current={"harness/f.py": "new", "old.py": "gone"},
        rollback_previous={"harness/f.py": "old"},
        contaminated=False)

    assert isinstance(report, GovernedHarnessReport)
    assert report.promoted
    assert report.verdict == "promote"
    assert report.safety.allowed
    assert report.approval is not None and report.approval.status == "approved"
    assert report.proof is not None and report.proof.passed
    assert report.evidence is not None
    assert report.rollback == {
        "write": {"harness/f.py": "old"},
        "delete": ("old.py",),
        "noop": False,
    }


def test_denied_import_rejects_before_proof_or_approval(tmp_path):
    before, after = _scores()
    store = ApprovalStore(str(tmp_path / "approvals.jsonl"))
    report = govern_harness_change(
        "harness-1", _spec("import os\n", risk=0.0), before, after,
        gate=_gate(), prover=_prover(), approval=store, contaminated=False)

    assert report.decision.action == "reject"
    assert not report.promoted
    assert report.evidence is None
    assert report.proof is None
    assert report.approval is None
    assert store.pending() == []
    assert any(item == "import:os" for item in report.safety.violations)


def test_failed_proof_is_attached_to_a_reject(tmp_path):
    before, after = _scores()
    report = govern_harness_change(
        "harness-1", _spec(), before, after, gate=_gate(),
        prover=_prover(passes=False), contaminated=False)

    assert report.decision.action == "reject"
    assert not report.promoted
    assert report.proof is not None and not report.proof.passed
    assert report.evidence is None
    assert "proof failed" in report.detail


def test_pending_approval_defers_without_evidence_or_promotion(tmp_path):
    before, after = _scores()
    store = _approval(tmp_path)
    report = govern_harness_change(
        "harness-1", _spec(), before, after, gate=_gate(),
        approval=store, approval_key="h1", contaminated=False)

    assert report.verdict == "needs-approval"
    assert report.decision.action == "defer"
    assert not report.promoted
    assert report.evidence is None
    assert report.approval is not None and report.approval.status == "pending"
    assert len(store.pending()) == 1


def test_rejected_approval_rejects_without_evidence(tmp_path):
    before, after = _scores()
    store = _approval(tmp_path, decision=False)
    report = govern_harness_change(
        "harness-1", _spec(), before, after, gate=_gate(),
        approval=store, approval_key="h1", contaminated=False)

    assert report.decision.action == "reject"
    assert not report.promoted
    assert report.evidence is None
    assert report.approval is not None and report.approval.status == "rejected"


def test_canary_rollback_wins_over_a_failed_eval(tmp_path):
    before, after = _scores(gain=-0.2, n=50)
    report = govern_harness_change(
        "harness-1", _spec(risk=0.0), before, after, gate=_gate(),
        contaminated=False)

    assert report.decision.action == "rollback"
    assert not report.promoted
    assert report.evidence is not None
    assert report.evidence.canary.verdict == "rollback"
    assert "canary rolled back" in report.detail


def test_rollback_plan_and_to_dict_are_json_safe(tmp_path):
    before, after = _scores()
    report = govern_harness_change(
        "harness-1", _spec(risk=0.0), before, after, gate=_gate(),
        rollback_current={"a.py": "new", "b.py": "extra"},
        rollback_previous={"a.py": "old"}, contaminated=False)

    assert report.rollback["write"] == {"a.py": "old"}
    assert report.rollback["delete"] == ("b.py",)
    assert report.rollback["noop"] is False

    payload = report.to_dict()
    encoded = json.dumps(payload)
    assert json.loads(encoded)["promoted"] is True
    assert payload["rollback"]["delete"] == ["b.py"]
    assert payload["proof"] is None
    assert payload["approval"] is None


def test_invalid_types_and_maps_are_refused(tmp_path):
    before, after = _scores(n=2)
    gate = _gate()
    base = dict(artifact_id="harness-1", modification=_spec(),
                before=before, after=after, gate=gate)

    with pytest.raises(TypeError):
        govern_harness_change(1, **{k: v for k, v in base.items()
                                    if k != "artifact_id"})
    with pytest.raises(ValueError):
        govern_harness_change("", **{k: v for k, v in base.items()
                                     if k != "artifact_id"})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "modification": "not-a-spec"})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "before": [("t0", 0.5)]})
    with pytest.raises(ValueError):
        govern_harness_change(**{**base, "before": {}})
    with pytest.raises(ValueError):
        govern_harness_change(**{**base, "after": {}})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "gate": object()})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "rules": object()})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "prover": object()})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "approval": object()})
    with pytest.raises(ValueError):
        govern_harness_change(**{**base, "rollback_current": {"a": "x"}})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "rollback_current": {"a": 1},
                                 "rollback_previous": {}})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "contaminated": "no"})
    with pytest.raises(ValueError):
        govern_harness_change(**{**base, "risk": 1.5})
    with pytest.raises(TypeError):
        govern_harness_change(**{**base, "kind": 1})
    with pytest.raises(TypeError):
        govern_harness_change(**base, approved=True)
