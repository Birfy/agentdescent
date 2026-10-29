"""A capability is admitted only after validation, review, safety, proof and gain."""

import pytest

from agentdescent.artifact_graph import ArtifactGraph
from agentdescent.capability_foundry import (
    CapabilityFoundry,
    CapabilityRequest,
    CapabilityStatus,
)
from agentdescent.deliberation import DeliberationReport
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.safe_selfmod import ModificationSpec, SafeSelfModifier
from agentdescent.tool_synthesis import ToolSpec, ToolValidation

_SOURCE = "def parse_csv(text):\n    return text.split(',')\n"


def _tool(source=_SOURCE):
    return ToolSpec("csv-tool", "parse a CSV line", source,
                    capabilities=("files:read",))


def _modification(tool=None, risk=0.0):
    tool = tool or _tool()
    return ModificationSpec(tool.name, tool.source, "tool",
                            requested_capabilities=tool.capabilities, risk=risk)


def _review(winner="approve", agreement=0.8):
    return DeliberationReport(winner, agreement, 0.9, (), "consensus")


def _prover(*, passes=True):
    return ProofRunner([
        ProofInvariant("check", lambda art: passes and bool(art.source)),
    ])


def _foundry(*, graph=None, prover=None, review=True, proof=True):
    return CapabilityFoundry(
        SafeSelfModifier({"files:read"}),
        graph=graph if graph is not None else ArtifactGraph(),
        prover=prover if prover is not None else _prover(),
        require_review=review, require_proof=proof)


def _request(**overrides):
    values = dict(
        gap_id="csv-failures", tool=_tool(),
        validation=ToolValidation(True, 0.9),
        modification=_modification(), before=0.5, after=0.7, canary=0.65,
        review=_review())
    values.update(overrides)
    return CapabilityRequest(**values)


def test_green_capability_is_promoted_and_registered():
    graph = ArtifactGraph()
    outcome = _foundry(graph=graph).admit(_request())
    assert outcome.status == CapabilityStatus.PROMOTED
    assert outcome.accepted
    assert outcome.artifact_id == "csv-tool"
    assert "csv-tool" in graph
    assert outcome.proof is not None and outcome.proof.passed
    assert outcome.holdout_gain == pytest.approx(0.2)


def test_invalid_or_low_scoring_validation_rejects():
    rejected = _foundry().admit(_request(
        validation=ToolValidation(False, 0.9, "broken")))
    assert rejected.status == CapabilityStatus.REJECTED
    weak = _foundry().admit(_request(validation=ToolValidation(True, 0.2)))
    assert weak.status == CapabilityStatus.REJECTED


def test_missing_and_disagreeing_review_fail_closed():
    missing = _foundry().admit(_request(review=None))
    assert missing.status == CapabilityStatus.DEFERRED
    assert "review" in missing.missing

    rejected = _foundry().admit(_request(review=_review(winner="reject")))
    assert rejected.status == CapabilityStatus.REJECTED

    deferred = _foundry().admit(_request(review=_review(agreement=0.2)))
    assert deferred.status == CapabilityStatus.DEFERRED


def test_safety_deny_hold_and_rollback():
    denied_source = "import os\n\ndef f():\n    return os.getcwd()\n"
    denied_tool = _tool(denied_source)
    denied = _foundry().admit(_request(
        tool=denied_tool, modification=_modification(denied_tool)))
    assert denied.status == CapabilityStatus.REJECTED

    held = _foundry().admit(_request(
        modification=_modification(risk=0.2), canary=None))
    assert held.status == CapabilityStatus.DEFERRED

    promoted = _foundry().admit(_request(
        modification=_modification(risk=0.2), canary=0.65))
    assert promoted.status == CapabilityStatus.PROMOTED

    rolled = _foundry().admit(_request(after=0.4))
    assert rolled.status == CapabilityStatus.ROLLED_BACK


def test_proof_missing_or_failed_fails_closed():
    no_prover = CapabilityFoundry(
        SafeSelfModifier({"files:read"}), prover=None, require_review=True,
        require_proof=True)
    deferred = no_prover.admit(_request())
    assert deferred.status == CapabilityStatus.DEFERRED
    assert "prover" in deferred.missing

    failed = _foundry(prover=_prover(passes=False)).admit(_request())
    assert failed.status == CapabilityStatus.REJECTED
    assert failed.proof is not None and not failed.proof.passed


def test_holdout_gain_and_duplicate_registration_reject():
    rejected = CapabilityFoundry(
        SafeSelfModifier({"files:read"}), prover=_prover(),
        min_holdout_gain=0.5).admit(_request())
    assert rejected.status == CapabilityStatus.REJECTED

    graph = ArtifactGraph()
    foundry = _foundry(graph=graph)
    assert foundry.admit(_request()).status == CapabilityStatus.PROMOTED
    duplicate = foundry.admit(_request())
    assert duplicate.status == CapabilityStatus.REJECTED
    assert "already registered" in " ".join(duplicate.reasons)


def test_gate_evidence_must_bind_to_the_admitted_tool():
    safe = _tool()
    wrong_id = ModificationSpec("other-id", safe.source, "tool",
                                requested_capabilities=safe.capabilities)
    outcome = _foundry().admit(_request(modification=wrong_id))
    assert outcome.status == CapabilityStatus.REJECTED
    assert "does not match" in " ".join(outcome.reasons)

    dangerous = "import os\n\ndef f():\n    return os.getcwd()\n"
    safe_twin = ModificationSpec(safe.name, safe.source, "tool",
                                 requested_capabilities=safe.capabilities)
    dangerous_tool = _tool(dangerous)
    mismatched = _foundry().admit(_request(
        tool=dangerous_tool, modification=safe_twin))
    assert mismatched.status == CapabilityStatus.REJECTED
    assert "source does not match" in " ".join(mismatched.reasons)

    unreviewed = _request(tool=ToolSpec(
        "csv-tool", "parse", safe.source,
        capabilities=("network:outbound",)))
    rejected = _foundry().admit(unreviewed)
    assert rejected.status == CapabilityStatus.REJECTED
    assert "not reviewed" in " ".join(rejected.reasons)


def test_registration_is_required_by_default_but_can_be_optional():
    mandatory = CapabilityFoundry(SafeSelfModifier({"files:read"}), graph=None,
                                  prover=_prover())
    deferred = mandatory.admit(_request())
    assert deferred.status == CapabilityStatus.DEFERRED
    assert "graph" in deferred.missing

    optional = CapabilityFoundry(SafeSelfModifier({"files:read"}), graph=None,
                                 prover=_prover(), require_registration=False)
    promoted = optional.admit(_request())
    assert promoted.status == CapabilityStatus.PROMOTED
    assert promoted.artifact_id == ""


def test_registered_artifact_carries_the_proof_digest():
    graph = ArtifactGraph()
    outcome = _foundry(graph=graph).admit(_request())
    assert outcome.proof is not None
    artifact = graph["csv-tool"]
    assert artifact.metadata["proof_digest"] == outcome.proof.digest
    assert artifact.metadata["artifact_hash"] == outcome.proof.artifact_hash


def test_invalid_records_are_refused():
    with pytest.raises(ValueError):
        CapabilityRequest("", _tool(), ToolValidation(True), _modification(),
                          0.5, 0.7)
    with pytest.raises(TypeError):
        CapabilityRequest("gap", "tool", ToolValidation(True),
                          _modification(), 0.5, 0.7)
    with pytest.raises(ValueError):
        CapabilityRequest("gap", _tool(), ToolValidation(True),
                          _modification(), 0.5, float("nan"))
    with pytest.raises(TypeError):
        CapabilityFoundry(object())
    with pytest.raises(ValueError):
        CapabilityFoundry(SafeSelfModifier({"files:read"}), min_valid_score=2)
    with pytest.raises(ValueError):
        CapabilityFoundry(SafeSelfModifier({"files:read"}), min_review_agreement=-1)
