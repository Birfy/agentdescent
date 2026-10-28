"""Learn when to approve automatically, when to ask a human, and when to deny.

A static approval threshold is either too conservative (a human reviews every
low-risk change) or too permissive (a confident-but-wrong candidate lands). The
frontier answer is **learning to defer**: fit a confidence threshold from past
review outcomes, control the risk of the auto-approve region, and send the
uncertain middle to a human.

This module is the small, deterministic version of that policy. It takes
labelled decisions already available in an approval store -- confidence, risk,
and whether the reviewer ultimately accepted -- and chooses the lowest threshold
whose auto-approve region has empirical risk at most ``target_risk`` and
coverage at least ``min_coverage``. A separate ``hard_risk`` ceiling denies
catastrophic candidates regardless of confidence.

    policy = DeferralPolicy(DeferralConfig(target_risk=0.05))
    policy.fit(history)              # DeferralSample(confidence, risk, safe)
    policy.decide(confidence=0.82, risk=0.1)
    # -> "auto-approve" | "defer" | "deny"

The fit is empirical, not a formal guarantee. Pair it with a conformal or
e-value gate for the assurance claim; this policy is the routing layer that
decides *where* that guarantee is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

__all__ = [
    "DeferralSample",
    "DeferralConfig",
    "DeferralReport",
    "DeferralPolicy",
]


def _prob(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return number


@dataclass(frozen=True)
class DeferralSample:
    """One past decision and its reviewer outcome."""

    confidence: float
    risk: float
    safe: bool

    def __post_init__(self) -> None:
        _prob(self.confidence, "confidence")
        _prob(self.risk, "risk")
        if not isinstance(self.safe, bool):
            raise ValueError("DeferralSample.safe must be a bool")


@dataclass(frozen=True)
class DeferralConfig:
    """Targets for the auto-approve region."""

    target_risk: float = 0.05
    min_coverage: float = 0.0
    hard_risk: float = 0.8
    min_samples: int = 5

    def __post_init__(self) -> None:
        _prob(self.target_risk, "target_risk")
        _prob(self.min_coverage, "min_coverage")
        _prob(self.hard_risk, "hard_risk")
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int)                 or self.min_samples < 1:
            raise ValueError("min_samples must be an int >= 1")


@dataclass(frozen=True)
class DeferralReport:
    """What the fitted policy does on its calibration sample."""

    threshold: float
    coverage: float
    review_rate: float
    deny_rate: float
    auto_risk: float
    n: int
    verdict: str
    detail: str = ""



class DeferralPolicy:
    """A confidence-threshold policy with a hard-risk deny ceiling."""

    def __init__(self, config: Optional[DeferralConfig] = None) -> None:
        self.config = config or DeferralConfig()
        self.threshold: Optional[float] = None
        self.report: Optional[DeferralReport] = None

    @property
    def fitted(self) -> bool:
        return self.threshold is not None

    def fit(self, samples: Iterable[DeferralSample]) -> DeferralReport:
        """Choose the lowest safe auto-approve threshold."""
        rows = list(samples)
        if not rows:
            raise ValueError("DeferralPolicy.fit needs at least one sample")
        if len(rows) < self.config.min_samples:
            return self._defer_all(rows, "insufficient",
                                   f"{len(rows)} < min_samples={self.config.min_samples}")
        candidates = sorted({row.confidence for row in rows})
        selected: Optional[float] = None
        best = None
        coverage_filtered = False
        for threshold in candidates:
            auto = [row for row in rows if row.confidence >= threshold
                    and row.risk <= self.config.hard_risk]
            if not auto:
                continue
            coverage = len(auto) / len(rows)
            risk = sum(0.0 if row.safe else 1.0 for row in auto) / len(auto)
            if coverage + 1e-12 < self.config.min_coverage:
                coverage_filtered = True
                continue
            if risk <= self.config.target_risk + 1e-12:
                selected = threshold
                best = (coverage, risk)
                break
        if selected is None:
            verdict = "infeasible" if coverage_filtered else "defer-all"
            return self._defer_all(rows, verdict,
                                   "no threshold meets target_risk and min_coverage")
        coverage, risk = best  # type: ignore[misc]
        auto = [row for row in rows if row.confidence >= selected
                and row.risk <= self.config.hard_risk]
        deny = [row for row in rows if row.risk > self.config.hard_risk]
        review = len(rows) - len(auto) - len(deny)
        if len(auto) == len(rows):
            verdict = "auto-all"
        else:
            verdict = "fitted"
        self.threshold = selected
        detail = (f"threshold={selected:.3f}, coverage={coverage:.3f}, "
                  f"auto-risk={risk:.3f}, review={review}/{len(rows)}")
        self.report = DeferralReport(
            threshold=selected, coverage=coverage,
            review_rate=review / len(rows), deny_rate=len(deny) / len(rows),
            auto_risk=risk, n=len(rows), verdict=verdict, detail=detail)
        return self.report

    def _defer_all(self, rows: Sequence[DeferralSample], verdict: str,
                   detail: str) -> DeferralReport:
        deny = [row for row in rows if row.risk > self.config.hard_risk]
        review = len(rows) - len(deny)
        self.threshold = math.inf
        self.report = DeferralReport(
            threshold=math.inf, coverage=0.0,
            review_rate=review / len(rows), deny_rate=len(deny) / len(rows),
            auto_risk=0.0, n=len(rows), verdict=verdict,
            detail=(detail + f"; review={review}/{len(rows)}"))
        return self.report

    def decide(self, confidence: float, risk: float) -> str:
        """Route one candidate: auto-approve, defer to a human, or deny."""
        if self.threshold is None:
            raise RuntimeError("DeferralPolicy.decide requires fit() first")
        confidence = _prob(confidence, "confidence")
        risk = _prob(risk, "risk")
        if risk > self.config.hard_risk:
            return "deny"
        if confidence >= self.threshold:
            return "auto-approve"
        return "defer"

    def route(self, samples: Iterable[DeferralSample]) -> Tuple[Tuple[str, ...], DeferralReport]:
        """Apply the fitted policy to samples and return decisions + report."""
        if self.threshold is None:
            raise RuntimeError("DeferralPolicy.route requires fit() first")
        rows = list(samples)
        decisions = tuple(
            self.decide(sample.confidence, sample.risk) for sample in rows)
        counts = {name: decisions.count(name) for name in
                  ("auto-approve", "defer", "deny")}
        denial_rate = counts["deny"] / len(decisions) if decisions else 0.0
        review_rate = counts["defer"] / len(decisions) if decisions else 0.0
        auto = [sample for sample, decision in zip(rows, decisions)
                if decision == "auto-approve"]
        auto_risk = (sum(0.0 if sample.safe else 1.0 for sample in auto) / len(auto)
                     if auto else 0.0)
        coverage = len(auto) / len(decisions) if decisions else 0.0
        report = DeferralReport(
            threshold=float(self.threshold), coverage=coverage,
            review_rate=review_rate, deny_rate=denial_rate, auto_risk=auto_risk,
            n=len(decisions), verdict="routed",
            detail=(f"{counts['auto-approve']} auto / {counts['defer']} review / "
                    f"{counts['deny']} deny"))
        return decisions, report
