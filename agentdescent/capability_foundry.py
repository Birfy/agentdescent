"""Turn a synthesized tool into an admitted capability, not a loose artifact.

`tool_synthesis` can propose a tool, `deliberation` can review it, `safe_selfmod`
can gate its authority, `proofs` can certify its invariants and `artifact_graph`
can store it. None of them owns the order or the decision to admit. This module
is that narrow coordinator:

1. reject a tool whose validator did not accept it or whose score is too low;
2. require a debate/review winner and enough agreement when configured;
3. run the capability/risk gate and its canary/rollback rule;
4. prove the tool's invariants and verify the certificate;
5. require a holdout gain;
6. register the admitted tool in the artifact graph.

Any missing required evidence keeps the result ``deferred``; a failed gate is
``rejected``; a measured canary regression is ``rolled_back``. Only a fully
green candidate is ``promoted``.

    foundry = CapabilityFoundry(
        gate=SafeSelfModifier({"files:write"}),
        prover=ProofRunner([ProofInvariant("no-network", no_network)]),
        graph=ArtifactGraph(),
    )
    outcome = foundry.admit(CapabilityRequest(
        gap_id="csv-failures",
        tool=tool,
        validation=validation,
        review=debate_report,
        modification=ModificationSpec(tool.name, tool.source, "tool"),
        before=0.5, after=0.7, canary=0.65,
    ))
    outcome.status, outcome.artifact_id, outcome.reasons
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .artifact_graph import Artifact, ArtifactGraph
from .deliberation import DeliberationReport
from .proofs import ProofCertificate, ProofRunner
from .safe_selfmod import ModificationSpec, SafeSelfModifier, SafetyDecision
from .tool_synthesis import ToolSpec, ToolValidation

__all__ = [
    "CapabilityStatus",
    "CapabilityRequest",
    "CapabilityOutcome",
    "CapabilityFoundry",
]


class CapabilityStatus:
    """The closed outcome vocabulary for one capability candidate."""

    PROMOTED = "promoted"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"

    ALL = (PROMOTED, DEFERRED, REJECTED, ROLLED_BACK)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class CapabilityRequest:
    """One synthesized tool and the evidence gathered for it."""

    gap_id: str
    tool: ToolSpec
    validation: ToolValidation
    modification: ModificationSpec
    before: float
    after: float
    canary: Optional[float] = None
    review: Optional[DeliberationReport] = None
    review_winner: str = "approve"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.gap_id, str) or not self.gap_id.strip():
            raise ValueError("gap_id is required")
        if not isinstance(self.tool, ToolSpec):
            raise TypeError("tool must be a ToolSpec")
        if not isinstance(self.validation, ToolValidation):
            raise TypeError("validation must be a ToolValidation")
        if not isinstance(self.modification, ModificationSpec):
            raise TypeError("modification must be a ModificationSpec")
        _finite(self.before, "before")
        _finite(self.after, "after")
        if self.canary is not None:
            _finite(self.canary, "canary")
        if not isinstance(self.review_winner, str) or not self.review_winner.strip():
            raise ValueError("review_winner is required")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class CapabilityOutcome:
    """The admission decision, the proof and the registered artifact id."""

    gap_id: str
    tool_name: str
    status: str
    reasons: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    artifact_id: str = ""
    proof: Optional[ProofCertificate] = None
    safety: Optional[SafetyDecision] = None
    review: Optional[DeliberationReport] = None
    validation: Optional[ToolValidation] = None
    holdout_gain: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in CapabilityStatus.ALL:
            raise ValueError(f"unknown capability status {self.status!r}")

    @property
    def accepted(self) -> bool:
        return self.status == CapabilityStatus.PROMOTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "missing": list(self.missing),
            "artifact_id": self.artifact_id,
            "proof": None if self.proof is None else self.proof.to_dict(),
            "safety": None if self.safety is None else {
                "action": self.safety.action,
                "allowed": self.safety.allowed,
                "reason": self.safety.reason,
                "risk": self.safety.risk,
                "violations": list(self.safety.violations),
                "capabilities": list(self.safety.capabilities),
            },
            "review": None if self.review is None else {
                "winner": self.review.winner,
                "agreement": self.review.agreement,
                "confidence": self.review.confidence,
                "stopped_reason": self.review.stopped_reason,
            },
            "validation": None if self.validation is None else {
                "valid": self.validation.valid,
                "score": self.validation.score,
                "reason": self.validation.reason,
            },
            "holdout_gain": self.holdout_gain,
        }


class CapabilityFoundry:
    """Admit a synthesized tool only when its review, safety, proof and gain hold.

    ``require_registration`` (default on) makes a missing artifact graph a
    deferral: an admission loop that cannot register the result has not
    admitted it. Set it to ``False`` only when the caller registers elsewhere.
    """

    def __init__(self, gate: SafeSelfModifier, *,
                 graph: Optional[ArtifactGraph] = None,
                 prover: Optional[ProofRunner] = None,
                 require_review: bool = True,
                 require_proof: bool = True,
                 require_registration: bool = True,
                 min_valid_score: float = 0.5,
                 min_review_agreement: float = 0.6,
                 min_holdout_gain: float = 0.0) -> None:
        if not isinstance(gate, SafeSelfModifier):
            raise TypeError("gate must be a SafeSelfModifier")
        if graph is not None and not isinstance(graph, ArtifactGraph):
            raise TypeError("graph must be an ArtifactGraph or None")
        if prover is not None and not isinstance(prover, ProofRunner):
            raise TypeError("prover must be a ProofRunner or None")
        for name, value in (("require_review", require_review),
                            ("require_proof", require_proof),
                            ("require_registration", require_registration)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        score = _finite(min_valid_score, "min_valid_score")
        agreement = _finite(min_review_agreement, "min_review_agreement")
        gain = _finite(min_holdout_gain, "min_holdout_gain")
        if not 0.0 <= score <= 1.0:
            raise ValueError("min_valid_score must be in [0, 1]")
        if not 0.0 <= agreement <= 1.0:
            raise ValueError("min_review_agreement must be in [0, 1]")
        if gain < 0:
            raise ValueError("min_holdout_gain must be >= 0")
        self.gate = gate
        self.graph = graph
        self.prover = prover
        self.require_review = require_review
        self.require_proof = require_proof
        self.require_registration = require_registration
        self.min_valid_score = score
        self.min_review_agreement = agreement
        self.min_holdout_gain = gain

    def admit(self, request: CapabilityRequest) -> CapabilityOutcome:
        if not isinstance(request, CapabilityRequest):
            raise TypeError("request must be a CapabilityRequest")
        reasons: list[str] = []
        missing: list[str] = []
        deferred: list[str] = []
        hard_fail = False
        rolled_back = False

        if request.modification.modification_id != request.tool.name:
            hard_fail = True
            reasons.append(
                f"modification {request.modification.modification_id!r} does not "
                f"match tool {request.tool.name!r}")
        if request.modification.source != request.tool.source:
            hard_fail = True
            reasons.append("modification source does not match the tool source")
        unreviewed = sorted(set(request.tool.capabilities) -
                            set(request.modification.requested_capabilities))
        if unreviewed:
            hard_fail = True
            reasons.append(
                "tool capabilities not reviewed by the gate: "
                + ",".join(unreviewed))

        validation = request.validation
        if not validation.valid:
            hard_fail = True
            reasons.append(
                f"validation rejected the tool: {validation.reason or 'invalid'}")
        elif validation.score < self.min_valid_score:
            hard_fail = True
            reasons.append(
                f"validation score {validation.score:.3f} < "
                f"min_valid_score {self.min_valid_score:.3f}")

        review = request.review
        if review is None:
            if self.require_review:
                missing.append("review")
        else:
            if review.winner != request.review_winner:
                hard_fail = True
                reasons.append(
                    f"review winner {review.winner!r} != "
                    f"{request.review_winner!r}")
            if review.agreement < self.min_review_agreement:
                deferred.append(
                    f"review agreement {review.agreement:.3f} < "
                    f"{self.min_review_agreement:.3f}")

        # The gate and prover are read-only here: gather every reason instead of
        # stopping at the first. A future stateful gate must stay side-effect
        # free, or this method must stop after the first hard failure.
        safety = self.gate.review(request.modification)
        if not safety.allowed:
            hard_fail = True
            reasons.append(f"safety denied: {safety.reason} "
                           f"({','.join(safety.violations) or 'no violation tag'})")
        else:
            safety = self.gate.promote(
                request.modification, before=request.before,
                after=request.after, canary=request.canary)
            if safety.action == "deny":
                hard_fail = True
                reasons.append(f"safety denied: {safety.reason}")
            elif safety.action == "rollback":
                rolled_back = True
                reasons.append(f"safety rolled back: {safety.reason}")
            elif safety.action == "hold":
                deferred.append(f"safety is holding: {safety.reason}")

        proof: Optional[ProofCertificate] = None
        if self.require_proof:
            if self.prover is None:
                missing.append("prover")
            else:
                proof = self.prover.prove(
                    request.tool, artifact_id=request.tool.name)
                verification = self.prover.verify(proof, request.tool)
                if not proof.passed:
                    hard_fail = True
                    reasons.append("proof certificate is marked failed")
                if not verification.valid:
                    hard_fail = True
                    reasons.append(
                        f"proof verification failed: {verification.reason}")

        holdout_gain = request.after - request.before
        if holdout_gain < self.min_holdout_gain:
            hard_fail = True
            reasons.append(
                f"holdout gain {holdout_gain:+.6g} < "
                f"min_holdout_gain {self.min_holdout_gain:.6g}")

        if rolled_back:
            status = CapabilityStatus.ROLLED_BACK
        elif hard_fail:
            status = CapabilityStatus.REJECTED
        elif missing or deferred:
            status = CapabilityStatus.DEFERRED
        else:
            status = CapabilityStatus.PROMOTED
        if status == CapabilityStatus.PROMOTED and self.graph is None \
                and self.require_registration:
            status = CapabilityStatus.DEFERRED
            missing.append("graph")
            reasons.append("artifact registration is required")

        artifact_id = ""
        if status == CapabilityStatus.PROMOTED and self.graph is not None:
            if request.tool.name in self.graph:
                status = CapabilityStatus.REJECTED
                reasons.append(
                    f"artifact {request.tool.name!r} is already registered")
            else:
                self.graph.add(Artifact(
                    artifact_id=request.tool.name,
                    text=request.tool.description + "\n" + request.tool.source,
                    kind="tool",
                    tags=request.tool.capabilities,
                    metadata={
                        "gap_id": request.gap_id,
                        "validation_score": validation.score,
                        "holdout_gain": holdout_gain,
                        "proof_digest": (None if proof is None else proof.digest),
                        "artifact_hash": (None if proof is None
                                          else proof.artifact_hash),
                    }))
                artifact_id = request.tool.name
                reasons.append("admitted and registered")

        reasons.extend(deferred)
        return CapabilityOutcome(
            gap_id=request.gap_id, tool_name=request.tool.name, status=status,
            reasons=tuple(reasons), missing=tuple(missing),
            artifact_id=artifact_id, proof=proof, safety=safety,
            review=review, validation=validation, holdout_gain=holdout_gain)
