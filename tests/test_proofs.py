"""A certificate must bind the artifact to the invariants it passed."""

from types import SimpleNamespace

import pytest

from agentdescent.proofs import (
    ProofCarryingArtifact,
    ProofInvariant,
    ProofRunner,
)


def _artifact(source="print(1)", tests=("t",)):
    return SimpleNamespace(artifact_id="a1", source=source, tests=tests)


def _runner():
    return ProofRunner([
        ProofInvariant("no-network", lambda art: "requests" not in art.source),
        ProofInvariant("has-tests", lambda art: bool(art.tests),
                       description="ships at least one test", severity="warning"),
    ], issuer="ci")


def test_prove_and_verify_roundtrip():
    artifact = _artifact()
    certificate = _runner().prove(artifact, issued_at=100.0)
    assert certificate.passed
    assert certificate.artifact_hash
    assert certificate.digest
    assert _runner().verify(certificate, artifact).valid
    assert certificate.to_dict()["issuer"] == "ci"
    assert certificate.to_dict()["results"][1]["description"] == "ships at least one test"


def test_proof_carrying_artifact_checks_the_certificate_not_just_the_hash():
    artifact = _artifact()
    certificate = _runner().prove(artifact, issued_at=100.0)
    wrapper = ProofCarryingArtifact(certificate.artifact_id,
                                    certificate.artifact_hash, certificate)
    assert wrapper.verify(artifact).valid
    assert not wrapper.verify(_artifact("print(2)")).valid

    bad = _artifact("import requests")
    failed = _runner().prove(bad, issued_at=100.0)
    assert not failed.passed
    wrapper = ProofCarryingArtifact(failed.artifact_id, failed.artifact_hash,
                                    failed)
    verification = wrapper.verify(bad)
    assert not verification.valid
    assert "no-network" in verification.reason


def test_failing_error_obligation_denies_and_verifies_false():
    artifact = _artifact("import requests")
    certificate = _runner().prove(artifact, issued_at=100.0)
    assert not certificate.passed
    verification = _runner().verify(certificate, artifact)
    assert not verification.valid
    assert "no-network" in verification.reason


def test_warning_does_not_fail_the_certificate():
    artifact = _artifact(tests=())
    certificate = _runner().prove(artifact, issued_at=100.0)
    assert certificate.passed
    assert _runner().verify(certificate, artifact).valid


def test_crashing_check_is_a_failure_not_a_crash():
    runner = ProofRunner([ProofInvariant("boom", lambda _art: 1 / 0)])
    certificate = runner.prove(_artifact(), issued_at=100.0)
    assert not certificate.passed
    assert "ZeroDivisionError" in certificate.results[0].reason


def test_tampered_artifact_or_certificate_is_rejected():
    artifact = _artifact()
    certificate = _runner().prove(artifact, issued_at=100.0)
    assert not _runner().verify(certificate, _artifact("print(2)")).valid
    tampered = type(certificate)(
        certificate.artifact_id, certificate.artifact_hash, certificate.passed,
        certificate.results, certificate.issuer, certificate.issued_at, "00" * 32)
    assert not _runner().verify(tampered, artifact).valid
    wrapper = ProofCarryingArtifact(tampered.artifact_id, tampered.artifact_hash,
                                    tampered)
    assert not wrapper.verify(artifact).valid


def test_invalid_invariants_and_runner_arguments_are_refused():
    with pytest.raises(ValueError):
        ProofInvariant("", lambda _art: True)
    with pytest.raises(TypeError):
        ProofInvariant("x", None)
    with pytest.raises(ValueError):
        ProofInvariant("x", lambda _art: True, severity="fatal")
    with pytest.raises(ValueError):
        ProofRunner([])
    with pytest.raises(ValueError):
        ProofRunner([ProofInvariant("x", lambda _art: True),
                     ProofInvariant("x", lambda _art: True)])
    with pytest.raises(ValueError):
        ProofRunner([ProofInvariant("x", lambda _art: True)], issuer="")
    with pytest.raises(ValueError):
        _runner().prove(_artifact(), issued_at=float("nan"))
    with pytest.raises(TypeError):
        _runner().verify("not a cert", _artifact())
