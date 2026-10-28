"""Distribution-free lower prediction bounds for the acceptance gate.

A point estimate says "the candidate gained 0.04"; it does not say what a new
task's improvement is likely to be. Bootstrap CIs in
:mod:`agentdescent.stats_eval` answer a fixed-sample question. Split conformal
answers the *exchangeable* version without a model: hold out calibration tasks,
measure paired improvements, and read the finite-sample lower prediction bound.

With ``n`` calibration improvements, the ``floor((n + 1) * alpha)``-th smallest
value is the split-conformal lower bound for **one new task's** paired
improvement. Under exchangeability -- calibration and the new task are draws
from the same distribution, and the calibration set was not selected using the
new observation -- a new improvement falls below that bound with probability at
most ``alpha``. When ``(n + 1) * alpha < 1`` no finite bound is possible at that
level; this module returns ``-inf`` and the gate reports ``insufficient`` rather
than clamping a rank into existence.

This is a **per-task prediction guarantee**, not a false-acceptance rate across
many candidates. If the archive reuses one calibration set to select among many
candidates, or changes the calibration set after seeing outcomes, exchangeability
is broken and the bound no longer covers. Use a fresh calibration split for a
selection rule that needs a candidate-level guarantee.

    gate = ConformalGate(alpha=0.05)
    gate.fit([candidate_i - incumbent_i for i in calibration_tasks])
    report = gate.decide(observed_improvement)
    if report.accepted:
        commit()

The calibrator never runs a task; the caller supplies paired improvements from
the same distribution as the decision. It is standalone for now -- it does not
replace ``Policies(acceptance=...)`` -- so an integration can call ``decide()``
at the merge seam without changing the existing acceptance path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

__all__ = [
    "ConformalGate",
    "ConformalReport",
    "conformal_lower_bound",
    "conformal_p_value",
]


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def _samples(values: Iterable[float]) -> Tuple[float, ...]:
    out = []
    for value in values:
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"calibration values must be finite, got {value!r}")
        out.append(v)
    if not out:
        raise ValueError("at least one calibration value is required")
    return tuple(out)


def conformal_lower_bound(values: Sequence[float], alpha: float = 0.05) -> float:
    """The split-conformal ``alpha`` lower bound of a sample.

    Uses the finite-sample rank ``floor((n + 1) * alpha)`` (1-indexed). When
    that rank is zero -- equivalently ``n < 1/alpha - 1`` -- no finite bound
    exists and ``-inf`` is returned. Ties make the bound conservative (the
    floor rank can only certify at most ``alpha``, never more).
    """
    alpha = _validate_alpha(alpha)
    items = _samples(values)
    rank = min(len(items), math.floor((len(items) + 1) * alpha))
    if rank < 1:
        return float("-inf")
    return sorted(items)[rank - 1]


def conformal_p_value(values: Sequence[float], observed: float) -> float:
    """Conformal p-value for one observation against a calibration sample.

    This is the smeared empirical upper tail ``(1 + #{s_i >= observed}) / (n + 1)``.
    It is a **diagnostic**, not the gate: under exchangeability a typical
    observation from the same distribution has a p-value around ``0.5`` even
    when the lower bound is positive. Use :meth:`ConformalGate.decide` for the
    decision.
    """
    items = _samples(values)
    value = float(observed)
    if not math.isfinite(value):
        raise ValueError(f"observed must be finite, got {observed!r}")
    at_least = sum(1 for item in items if item >= value)
    return (1 + at_least) / (len(items) + 1)


@dataclass(frozen=True)
class ConformalReport:
    """The calibration and one decision, with the bound it used."""

    alpha: float
    n: int
    threshold: float
    observed: Optional[float]
    p_value: Optional[float]
    decision: str
    calibration_below_fraction: float
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"


class ConformalGate:
    """A fitted split-conformal lower prediction bound used as an acceptance rule.

    Parameters
    ----------
    alpha:
        Target per-task coverage level. The finite-sample default requires at
        least ``ceil(1/alpha) - 1`` calibration values for a finite bound
        (19 at ``alpha=0.05``).
    min_samples:
        Refuse to decide below this many calibration values. The default is
        ``max(1, ceil(1/alpha) - 1)``, the smallest sample that can produce a
        finite bound at this alpha.
    min_gain:
        The lower bound must exceed this, not merely zero. Use a positive value
        when a change has to pay for its own evaluation cost.
    """

    def __init__(self, alpha: float = 0.05, *, min_samples: Optional[int] = None,
                 min_gain: float = 0.0) -> None:
        self.alpha = _validate_alpha(alpha)
        derived = max(1, math.ceil(1.0 / self.alpha) - 1)
        if min_samples is None:
            min_samples = derived
        if int(min_samples) < 1:
            raise ValueError("min_samples must be >= 1")
        if not math.isfinite(float(min_gain)):
            raise ValueError("min_gain must be finite")
        self.min_samples = int(min_samples)
        self.min_gain = float(min_gain)
        self._values: Tuple[float, ...] = ()

    @property
    def n(self) -> int:
        return len(self._values)

    @property
    def values(self) -> Tuple[float, ...]:
        return self._values

    def fit(self, improvements: Iterable[float]) -> "ConformalGate":
        """Replace the calibration sample and return ``self``."""
        self._values = _samples(improvements)
        return self

    def threshold(self) -> float:
        if not self._values:
            raise ValueError("ConformalGate is not fitted")
        return conformal_lower_bound(self._values, self.alpha)

    def p_value(self, observed: float) -> float:
        if not self._values:
            raise ValueError("ConformalGate is not fitted")
        return conformal_p_value(self._values, observed)

    def decide(self, observed: Optional[float] = None) -> ConformalReport:
        """Decide from the fitted lower bound.

        ``observed`` is accepted for diagnostics and reports its p-value, but it
        does **not** gate the decision: a typical draw from the calibration
        distribution has a mid-range p-value even when its lower bound is
        positive. The decision is the conformal bound itself.
        """
        if not self._values:
            raise ValueError("ConformalGate is not fitted")
        threshold = self.threshold()
        below = sum(1 for value in self._values if value < threshold) / len(self._values)
        p_value = None if observed is None else self.p_value(observed)
        if not math.isfinite(threshold) or len(self._values) < self.min_samples:
            decision = "insufficient"
            if not math.isfinite(threshold):
                detail = (f"{len(self._values)} calibration value(s) cannot certify "
                          f"alpha={self.alpha:g}; need at least {self.min_samples}")
            else:
                detail = f"{len(self._values)} < min_samples={self.min_samples}"
        else:
            decision = "accept" if threshold > self.min_gain else "reject"
            detail = (f"lower bound {threshold:.6g} vs min_gain {self.min_gain:.6g}"
                      + ("" if p_value is None else f"; p={p_value:.4g} (diagnostic)"))
        return ConformalReport(alpha=self.alpha, n=len(self._values),
                               threshold=threshold, observed=observed,
                               p_value=p_value, decision=decision,
                               calibration_below_fraction=below, detail=detail)

    def accept(self, observed: Optional[float] = None) -> bool:
        return self.decide(observed).accepted
