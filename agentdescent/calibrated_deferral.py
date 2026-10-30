"""Calibrate confidence first, then decide answer / defer / deny.

`deferral` learns a review threshold from confidence and risk, but it trusts
the confidence it is given. `self_model` can calibrate that confidence against
real outcomes. Used separately, a miscalibrated confidence produces a confident
auto-approve region with the wrong risk; this module connects them:

    calibrated = SelfModel().fit(confidences, outcomes)
    policy = DeferralPolicy(DeferralConfig(target_risk=0.05))
    router = CalibratedDeferral(calibrated, policy)
    router.fit(confidences, outcomes, risks, safe)
    decision = router.decide(raw_confidence, risk)
    decision.action, decision.calibrated_confidence

The fit is deterministic. The deferral policy is trained on the **calibrated**
confidence (not the raw one), and the hard-risk deny ceiling still applies to
every candidate. This is the routing layer; the conformal/e-value gate remains
the formal guarantee.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from .deferral import DeferralPolicy, DeferralReport, DeferralSample
from .self_model import CalibrationReport, SelfModel

__all__ = [
    "CalibratedDecision",
    "CalibratedDeferralReport",
    "CalibratedDeferral",
]


def _prob(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return number


@dataclass(frozen=True)
class CalibratedDecision:
    """One routed candidate, with both the raw and calibrated confidence."""

    raw_confidence: float
    calibrated_confidence: float
    risk: float
    action: str
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibratedDeferralReport:
    """Calibration quality and the deferral policy it produced."""

    calibration: CalibrationReport
    raw_calibration: CalibrationReport
    deferral: DeferralReport
    detail: str = ""

    @property
    def fitted(self) -> bool:
        return self.deferral.verdict not in ("insufficient", "defer-all",
                                             "infeasible")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration": asdict(self.calibration),
            "raw_calibration": asdict(self.raw_calibration),
            "deferral": asdict(self.deferral),
            "detail": self.detail,
        }


class CalibratedDeferral:
    """A `SelfModel` feeding a `DeferralPolicy` on calibrated confidence."""

    def __init__(self, model: Optional[SelfModel] = None,
                 policy: Optional[DeferralPolicy] = None) -> None:
        if model is not None and not isinstance(model, SelfModel):
            raise TypeError("model must be a SelfModel or None")
        if policy is not None and not isinstance(policy, DeferralPolicy):
            raise TypeError("policy must be a DeferralPolicy or None")
        self.model = model or SelfModel()
        self.policy = policy or DeferralPolicy()
        self.report: Optional[CalibratedDeferralReport] = None

    @property
    def fitted(self) -> bool:
        return (self.report is not None and self.report.fitted
                and self.policy.threshold is not None
                and math.isfinite(float(self.policy.threshold)))

    def fit(self, confidences: Iterable[float], outcomes: Iterable[float],
            risks: Iterable[float], safe: Optional[Iterable[bool]] = None, *,
            bins: int = 10, min_n: int = 5) -> CalibratedDeferralReport:
        """Fit calibration and the deferral policy.

        ``outcomes`` are task successes; ``safe`` is the reviewer's safety
        label. When ``safe`` is omitted it defaults to ``outcome == 1``.
        Both stages are fitted on the same rows (in-sample calibration); pass
        a held-out set to ``raw_report`` yourself if you need an unbiased ECE.
        """
        conf = tuple(_prob(value, "confidence") for value in confidences)
        out = tuple(float(value) for value in outcomes)
        risk = tuple(_prob(value, "risk") for value in risks)
        if safe is None:
            safe_rows = tuple(value == 1.0 for value in out)
        else:
            safe_rows = tuple(safe)
        if not (len(conf) == len(out) == len(risk) == len(safe_rows)):
            raise ValueError("confidence/outcome/risk/safe lengths must match")
        if not conf:
            raise ValueError("fit needs at least one sample")
        if any(value not in (0.0, 1.0) for value in out):
            raise ValueError("outcomes must be 0 or 1")
        if any(not isinstance(value, bool) for value in safe_rows):
            raise TypeError("safe must contain bools")
        raw = self.model.raw_report(conf, out, bins=bins, min_n=min_n)
        calibration = self.model.fit(conf, out, bins=bins, min_n=min_n)
        calibrated = self.model.predict_many(conf)
        deferral = self.policy.fit(
            DeferralSample(c, r, s)
            for c, r, s in zip(calibrated, risk, safe_rows))
        detail = (f"raw {raw.verdict} ECE={raw.ece:.3f}; "
                  f"calibrated {calibration.verdict} ECE={calibration.ece:.3f}; "
                  f"deferral {deferral.verdict}")
        self.report = CalibratedDeferralReport(
            calibration=calibration, raw_calibration=raw,
            deferral=deferral, detail=detail)
        return self.report

    def decide(self, raw_confidence: float, risk: float) -> CalibratedDecision:
        if not self.fitted:
            raise RuntimeError("CalibratedDeferral.decide requires a fitted "
                               "calibration and a finite deferral threshold")
        raw = _prob(raw_confidence, "confidence")
        risk = _prob(risk, "risk")
        calibrated = self.model.predict(raw)
        threshold = float(self.policy.threshold)  # type: ignore[arg-type]
        # Both the empirical threshold and the calibrated risk estimate must
        # permit auto-approval; calibration therefore changes the decision, not
        # just the scale of the threshold.
        if risk > self.policy.config.hard_risk:
            action = "deny"
        elif ((1.0 - calibrated) > self.policy.config.target_risk
              or calibrated < threshold):
            action = "defer"
        else:
            action = "auto-approve"
        return CalibratedDecision(raw, calibrated, risk, action, threshold)

    def route(self, rows: Iterable[Tuple[float, float]]) -> Tuple[CalibratedDecision, ...]:
        if not self.fitted:
            raise RuntimeError("CalibratedDeferral.route requires fit() first")
        return tuple(self.decide(confidence, risk) for confidence, risk in rows)
