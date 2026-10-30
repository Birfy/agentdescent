"""Build one promotion bundle from the independent evidence modules.

`promotion` knows how to judge evidence, but a caller still has to assemble the
evaluation, canary, conformal, sequential, Goodhart, metamorphic, verifier and
calibration reports. This module is that assembly line. It turns paired
incumbent/candidate task scores into every report the promotion seam can use,
so the independent modules are connected instead of being entry points nobody
calls.

    evidence = build_trust_evidence(
        before_scores, after_scores,
        verifier=ensemble, candidate=candidate,
        calibration=(confidences, outcomes),
        proxy_gains=proxy, oracle_gains=oracle,
        metamorphic=suite, metamorphic_run=solver,
        metamorphic_tasks=tasks,
        contaminated=False, risk=0.2,
    )
    decision = promote(evidence.to_promotion_request("skill-1"),
                       PromotionRules(require_goodhart=True,
                                      require_verifier=True,
                                      require_calibration=True))

The builder does not decide promotion and does not re-run a gate. It only
computes the reports from the data the caller already has.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from .conformal import ConformalGate, ConformalReport
from .evalops import EvalResult
from .goodhart import GoodhartReport, detect_goodhart
from .metamorphic import MetamorphicReport, MetamorphicSuite
from .promotion import PromotionRequest
from .safety_controller import CanaryResult, canary
from .self_model import CalibrationReport, SelfModel
from .sequential import SequentialReport, sequential_test
from .stats_eval import bootstrap_ci, is_significant, paired_gain
from .verifier_ensemble import EnsembleReport, VerifierEnsemble

__all__ = ["TrustEvidence", "build_trust_evidence"]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _content_hash(before: Mapping[str, float], after: Mapping[str, float]) -> str:
    body = json.dumps(
        {"before": sorted((str(k), float(v)) for k, v in before.items()),
         "after": sorted((str(k), float(v)) for k, v in after.items())},
        sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class TrustEvidence:
    """The reports one paired evaluation produced, ready for `promote()`."""

    eval_result: EvalResult
    canary: CanaryResult
    conformal: ConformalReport
    sequential: SequentialReport
    goodhart: Optional[GoodhartReport] = None
    metamorphic: Optional[MetamorphicReport] = None
    verifier: Optional[EnsembleReport] = None
    calibration: Optional[CalibrationReport] = None
    contaminated: Optional[bool] = None
    risk: Optional[float] = None
    approved: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.contaminated is not None and not isinstance(self.contaminated, bool):
            raise TypeError("contaminated must be a bool or None")
        if self.approved is not None and not isinstance(self.approved, bool):
            raise TypeError("approved must be a bool or None")
        if self.risk is not None:
            risk = _finite(self.risk, "risk")
            if not 0.0 <= risk <= 1.0:
                raise ValueError("risk must be in [0, 1]")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_promotion_request(self, artifact_id: str, *,
                             artifact_hash: Optional[str] = None,
                             proof: Optional[Any] = None,
                             admission: Optional[Any] = None) -> PromotionRequest:
        """A `PromotionRequest` carrying every report this bundle has."""
        metadata = dict(self.metadata)
        metadata.update({
            "goodhart": None if self.goodhart is None else asdict(self.goodhart),
            "metamorphic": (None if self.metamorphic is None
                            else asdict(self.metamorphic)),
            "verifier": (None if self.verifier is None
                         else self.verifier.to_dict()),
            "calibration": (None if self.calibration is None
                            else asdict(self.calibration)),
        })
        return PromotionRequest(
            artifact_id=artifact_id, artifact_hash=artifact_hash,
            eval_result=self.eval_result, contaminated=self.contaminated,
            conformal=self.conformal, sequential=self.sequential,
            canary=self.canary, proof=proof, admission=admission,
            risk=self.risk, approved=self.approved, goodhart=self.goodhart,
            metamorphic=self.metamorphic, verifier=self.verifier,
            calibration=self.calibration, metadata=metadata)

    def to_dict(self) -> dict:
        return {
            "eval": self.eval_result.to_dict(),
            "canary": asdict(self.canary),
            "conformal": asdict(self.conformal),
            "sequential": asdict(self.sequential),
            "goodhart": None if self.goodhart is None else asdict(self.goodhart),
            "metamorphic": (None if self.metamorphic is None
                            else asdict(self.metamorphic)),
            "verifier": (None if self.verifier is None
                         else self.verifier.to_dict()),
            "calibration": (None if self.calibration is None
                            else asdict(self.calibration)),
            "contaminated": self.contaminated,
            "risk": self.risk,
            "approved": self.approved,
            "metadata": dict(self.metadata),
        }


def build_trust_evidence(
        before: Mapping[str, float], after: Mapping[str, float], *,
        eval_id: str = "paired", dataset_id: str = "paired", version: int = 1,
        verifier: Optional[VerifierEnsemble] = None,
        candidate: Any = None,
        calibration: Optional[Tuple[Iterable[float], Iterable[float]]] = None,
        sequential_deltas: Optional[Iterable[float]] = None,
        sequential_max_n: Optional[int] = None,
        proxy_gains: Optional[Sequence[float]] = None,
        oracle_gains: Optional[Sequence[float]] = None,
        metamorphic: Optional[MetamorphicSuite] = None,
        metamorphic_run: Optional[Callable[[Any], Any]] = None,
        metamorphic_tasks: Iterable[Any] = (),
        contaminated: Optional[bool] = None,
        risk: Optional[float] = None,
        approved: Optional[bool] = None,
        alpha: float = 0.05,
        seed: int = 0,
        min_effect: float = 0.0,
) -> TrustEvidence:
    """Compute every report from paired task scores and optional side data."""
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError("before and after must be mappings")
    if not before or not after:
        raise ValueError("before and after must not be empty")
    alpha = _finite(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if _finite(min_effect, "min_effect") < 0:
        raise ValueError("min_effect must be >= 0")
    if (proxy_gains is None) != (oracle_gains is None):
        raise ValueError("proxy_gains and oracle_gains must be supplied together")
    if metamorphic is not None and metamorphic_run is None:
        raise ValueError("metamorphic needs a run callable")
    if verifier is not None and candidate is None:
        raise ValueError("verifier needs a candidate")

    deltas = paired_gain(before, after)
    if not deltas:
        raise ValueError("before and after have no task keys in common")
    bootstrap = bootstrap_ci(before, after, alpha=alpha, seed=seed)
    common = sorted(before.keys() & after.keys(), key=str)
    before_values = [float(before[key]) for key in common]
    after_values = [float(after[key]) for key in common]
    per_task = {str(key): float(after[key]) - float(before[key])
                for key in common}
    eval_result = EvalResult(
        eval_id=eval_id, dataset_id=dataset_id, version=version,
        content_hash=_content_hash(before, after), n=bootstrap.n,
        score=sum(after_values) / len(after_values),
        baseline=sum(before_values) / len(before_values),
        delta=bootstrap.mean, ci=(bootstrap.lo, bootstrap.hi),
        passed=is_significant(bootstrap, min_effect=min_effect),
        cost=0.0, per_task=per_task,
        detail=(f"bootstrap n={bootstrap.n}, "
                f"prob_positive={bootstrap.prob_positive:.3f}"))

    canary_result = canary(before, after, min_effect=min_effect, alpha=alpha)
    conformal = ConformalGate(alpha=alpha).fit(deltas).decide()
    stream = sequential_test(
        deltas if sequential_deltas is None else sequential_deltas,
        alpha=alpha, max_n=sequential_max_n)

    goodhart = None
    if proxy_gains is not None and oracle_gains is not None:
        goodhart = detect_goodhart(proxy_gains, oracle_gains)

    metamorphic_report = None
    if metamorphic is not None:
        metamorphic_report = metamorphic.evaluate(
            metamorphic_run, metamorphic_tasks)

    verifier_report = None
    if verifier is not None:
        verifier_report = verifier.evaluate(candidate)

    calibration_report = None
    if calibration is not None:
        confidences, outcomes = calibration
        calibration_report = SelfModel().fit(confidences, outcomes)

    return TrustEvidence(
        eval_result=eval_result, canary=canary_result, conformal=conformal,
        sequential=stream, goodhart=goodhart, metamorphic=metamorphic_report,
        verifier=verifier_report, calibration=calibration_report,
        contaminated=contaminated, risk=risk, approved=approved,
        metadata={"before_tasks": len(before), "after_tasks": len(after),
                  "common_tasks": len(deltas)})
