"""The trusted release pipeline composes evidence, promotion and ledger."""

import json

import pytest

from agentdescent.lineage import LineageGraph
from agentdescent.policy_release import verify_manifest
from agentdescent.promotion_ledger import PromotionLedger
from agentdescent.release_pipeline import (
    ReleaseReport,
    release_candidate,
    release_evidence,
)
from agentdescent.trust_evidence import build_trust_evidence


def _scores(gain=0.5, n=50):
    before = {f"t{i}": 0.5 for i in range(n)}
    after = {f"t{i}": 0.5 + gain for i in range(n)}
    return before, after


def _evidence(before=None, after=None):
    if before is None:
        before, after = _scores()
    return build_trust_evidence(before, after, contaminated=False, risk=0.1)


def test_successful_release_is_signed_and_lineage_linked():
    before, after = _scores()
    lineage = LineageGraph()
    ledger = PromotionLedger(lineage, issuer="ci")
    report = release_candidate(
        before, after, artifact_id="skill-1", contaminated=False, risk=0.1,
        ledger=ledger, key_id="ci", secret=b"secret", run_id="run-1")

    assert isinstance(report, ReleaseReport)
    assert report.released
    assert report.decision.action == "promote" and report.decision.accepted
    release = report.release
    assert release is not None
    assert release.verification.valid
    assert release.manifest.artifact_id == "skill-1"
    assert release.manifest.metadata["issuer"] == "ci"
    assert (release.manifest.evidence["decision_hash"]
            == report.decision.decision_hash)
    assert (release.manifest.evidence["evidence_hash"]
            == report.decision.evidence_hash)
    assert "skill-1" in lineage
    assert lineage["skill-1"].run_id == "run-1"
    assert release.manifest.manifest_hash() in report.detail
    assert verify_manifest(release.envelope, {"ci": b"secret"}).valid


def test_report_to_dict_is_json_serializable():
    before, after = _scores()
    report = release_candidate(
        before, after, artifact_id="skill-1", contaminated=False, risk=0.1,
        ledger=PromotionLedger(), key_id="ci", secret=b"secret")

    data = report.to_dict()
    assert data["artifact_id"] == "skill-1"
    assert data["released"] is True
    assert data["evidence"]["eval"]["content_hash"] == (
        report.evidence.eval_result.content_hash)
    assert data["decision"]["decision_hash"] == report.decision.decision_hash
    assert data["release"]["verification"]["valid"] is True
    json.dumps(data)


def test_rejected_candidate_is_not_released_or_recorded():
    before, _after = _scores()
    lineage = LineageGraph()
    ledger = PromotionLedger(lineage)
    report = release_candidate(
        before, dict(before), artifact_id="skill-bad", contaminated=False,
        risk=0.1, ledger=ledger, key_id="ci", secret=b"secret")

    assert not report.released
    assert report.release is None
    assert not report.decision.accepted
    assert report.decision.action == "reject"
    assert len(lineage) == 0
    assert "not released" in report.detail


def test_contaminated_dataset_is_not_released():
    before, after = _scores()
    lineage = LineageGraph()
    ledger = PromotionLedger(lineage)
    report = release_candidate(
        before, after, artifact_id="skill-bad", contaminated=True, risk=0.1,
        ledger=ledger, key_id="ci", secret=b"secret")

    assert not report.released
    assert report.release is None
    assert report.decision.action == "reject"
    assert "contaminated" in " ".join(report.decision.reasons)
    assert len(lineage) == 0


@pytest.mark.parametrize("overrides", [
    {"ledger": None},
    {"key_id": None},
    {"key_id": "   "},
    {"secret": None},
    {"secret": b""},
])
def test_missing_release_material_on_promotable_raises(overrides):
    evidence = _evidence()
    ledger = PromotionLedger()
    kwargs = dict(
        artifact_id="skill-1", ledger=ledger, key_id="ci", secret=b"secret")
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        release_evidence(evidence, **kwargs)
    assert len(ledger.lineage) == 0


def test_release_evidence_uses_prebuilt_evidence_without_rebuilding():
    evidence = _evidence()
    ledger = PromotionLedger()
    report = release_evidence(
        evidence, artifact_id="skill-1", ledger=ledger, key_id="ci",
        secret=b"secret")

    assert report.evidence is evidence
    assert report.released
    assert report.release is not None
    assert report.release.verification.valid


def test_second_release_records_parent_edge():
    before, after = _scores()
    lineage = LineageGraph()
    ledger = PromotionLedger(lineage)
    first = release_candidate(
        before, after, artifact_id="skill-1", contaminated=False, risk=0.1,
        ledger=ledger, key_id="ci", secret=b"secret")
    second = release_candidate(
        before, after, artifact_id="skill-2", contaminated=False, risk=0.1,
        ledger=ledger, key_id="ci", secret=b"secret",
        parents=("skill-1",))

    assert first.released and second.released
    assert second.release is not None
    assert second.release.lineage.parent_ids == ("skill-1",)
    assert lineage.parents("skill-2") == ("skill-1",)
    assert lineage.path("skill-1", "skill-2") == ("skill-1", "skill-2")


def test_deferred_result_is_reported_without_raising_or_releasing():
    before, after = _scores()
    ledger = PromotionLedger()
    report = release_candidate(
        before, after, artifact_id="skill-1", contaminated=False,
        ledger=ledger, key_id="ci", secret=b"secret")

    assert report.decision.action == "defer"
    assert "risk" in report.decision.missing
    assert not report.released
    assert report.release is None
    assert len(ledger.lineage) == 0


def test_invalid_artifacts_and_types_are_refused():
    before, after = _scores()
    evidence = _evidence(before, after)
    ledger = PromotionLedger()

    with pytest.raises(TypeError):
        release_evidence(object(), artifact_id="skill-1")
    with pytest.raises(TypeError):
        release_evidence(evidence, artifact_id=123)
    with pytest.raises(ValueError):
        release_evidence(evidence, artifact_id="   ")
    with pytest.raises(TypeError):
        release_evidence(evidence, artifact_id="skill-1", content_hash=123)
    with pytest.raises(TypeError):
        release_evidence(evidence, artifact_id="skill-1", key_id=123)
    with pytest.raises(TypeError):
        release_evidence(evidence, artifact_id="skill-1", secret="secret")
    with pytest.raises(TypeError):
        release_evidence(evidence, artifact_id="skill-1", ledger=object())
    with pytest.raises(TypeError):
        release_evidence(evidence, artifact_id="skill-1", rules=object())
    with pytest.raises(TypeError):
        release_evidence(evidence, artifact_id="skill-1", parents="skill-0")
    with pytest.raises(ValueError):
        release_evidence(evidence, artifact_id="skill-1", parents=("",))
    with pytest.raises(TypeError):
        release_evidence(evidence, artifact_id="skill-1", metadata=[])
    with pytest.raises(ValueError):
        release_candidate({"a": 0.1}, {"b": 0.2}, artifact_id="skill-1")
    assert len(ledger.lineage) == 0
