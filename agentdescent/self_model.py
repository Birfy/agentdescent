"""A calibrated model of what the agent itself can do.

The cheapest way to make a self-improving system safer is not another guard on
the candidate; it is a **self-model** that knows when the current artifact is
likely to fail. An uncalibrated confidence score is worse than none: it makes
the router confidently choose the tasks the agent is worst at.

This module fits a one-dimensional **Platt scaling** map in logit space --
``sigmoid(a * logit(confidence) + b)`` -- reports ECE and Brier, and exposes the
two decisions that matter:

* ``route(confidence, threshold)`` -- answer or abstain;
* ``select_by_confidence`` -- split a batch into the accepted and abstained
  items, so a controller can spend the abstentions on a stronger model.

The fit is deterministic (fixed-step gradient descent, no random init), so a
report can be reproduced from the same confidence/outcome rows. ``fit`` also
returns the report on the training rows; a held-out report is the caller's job,
exactly as with any calibrator. A fit with ``a < 0`` means raw confidence is
anti-correlated with success; ``predict`` then deliberately reverses the order,
and the report marks it ``anti_correlated``. An unfitted model refuses to
predict rather than silently mapping everything to ``[0.5, 0.73]``.

    model = SelfModel().fit(confidences, outcomes)
    print(model.report.ece, model.report.brier)
    if model.route(0.83, threshold=0.7) == "answer":
        ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CalibrationReport",
    "SelfModel",
    "brier_score",
    "expected_calibration_error",
    "select_by_confidence",
]


_EPS = 1e-6


def _logit(value: float) -> float:
    clipped = min(1.0 - _EPS, max(_EPS, float(value)))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _rows(confidences: Iterable[float], outcomes: Iterable[float]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    conf = tuple(float(x) for x in confidences)
    out = tuple(float(x) for x in outcomes)
    if len(conf) != len(out):
        raise ValueError(f"confidence/outcome mismatch: {len(conf)} != {len(out)}")
    if not conf:
        raise ValueError("at least one calibration row is required")
    for value in conf:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {value!r}")
    for value in out:
        if value not in (0.0, 1.0):
            raise ValueError(f"outcome must be 0 or 1, got {value!r}")
    return conf, out


def brier_score(confidences: Iterable[float], outcomes: Iterable[float]) -> float:
    """Mean squared difference between confidence and outcome (lower is better)."""
    conf, out = _rows(confidences, outcomes)
    return sum((c - y) ** 2 for c, y in zip(conf, out)) / len(conf)


def expected_calibration_error(confidences: Iterable[float], outcomes: Iterable[float],
                               *, bins: int = 10) -> float:
    """Weighted gap between confidence and accuracy over equal-width bins.

    Empty bins are skipped. ``bins=1`` is the absolute mean gap; the default
    matches the standard ECE definition.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    conf, out = _rows(confidences, outcomes)
    buckets: List[List[Tuple[float, float]]] = [[] for _ in range(bins)]
    for c, y in zip(conf, out):
        index = min(bins - 1, int(c * bins))
        buckets[index].append((c, y))
    total = len(conf)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        mean_out = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_conf - mean_out)
    return ece


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration and ranking quality for one set of rows."""

    n: int
    ece: float
    brier: float
    mean_confidence: float
    accuracy: float
    verdict: str
    anti_correlated: bool = False
    detail: str = ""

    @property
    def gap(self) -> float:
        return self.mean_confidence - self.accuracy

    @property
    def calibrated(self) -> bool:
        return self.verdict == "calibrated"


def _report(confidences: Sequence[float], outcomes: Sequence[float],
            *, bins: int = 10, min_n: int = 5,
            anti_correlated: bool = False) -> CalibrationReport:
    ece = expected_calibration_error(confidences, outcomes, bins=bins)
    brier = brier_score(confidences, outcomes)
    mean_conf = sum(confidences) / len(confidences)
    accuracy = sum(outcomes) / len(outcomes)
    if len(confidences) < min_n:
        verdict = "insufficient"
    elif ece <= 0.05:
        verdict = "calibrated"
    elif mean_conf > accuracy + 0.05:
        verdict = "overconfident"
    elif mean_conf < accuracy - 0.05:
        verdict = "underconfident"
    else:
        verdict = "miscalibrated"
    detail = (f"ECE={ece:.4f}, Brier={brier:.4f}, "
              f"confidence={mean_conf:.3f}, accuracy={accuracy:.3f}")
    if anti_correlated:
        detail += "; anti-correlated (fit reversed raw confidence)"
    return CalibrationReport(n=len(confidences), ece=ece, brier=brier,
                             mean_confidence=mean_conf, accuracy=accuracy,
                             verdict=verdict,
                             anti_correlated=anti_correlated, detail=detail)


class SelfModel:
    """A logit-space Platt model plus the two routing decisions.

    ``fit`` runs deterministic Newton/IRLS on the log loss of
    ``sigmoid(a * logit(confidence) + b)`` with a backtracking line search;
    ``a=1, b=0`` is the identity map. Newton avoids the divergence of fixed-step
    gradient descent on saturated 0/1 confidences. The map is monotone but not
    necessarily order-preserving: when raw confidence is anti-correlated with
    success, the fit can make ``a < 0`` and routing deliberately reverses the
    order. That case is reported as ``anti_correlated`` instead of being hidden.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0
        self.report: Optional[CalibrationReport] = None
        self._fitted = False

    @staticmethod
    def _log_loss(xs: Sequence[float], out: Sequence[float], a: float, b: float,
                  l2: float) -> float:
        total = 0.0
        for x, y in zip(xs, out):
            p = min(1.0 - 1e-12, max(1e-12, _sigmoid(a * x + b)))
            total -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
        return total / len(xs) + 0.5 * l2 * a * a

    def fit(self, confidences: Iterable[float], outcomes: Iterable[float], *,
            steps: int = 100, learning_rate: float = 1.0,
            l2: float = 1e-4, tol: float = 1e-10,
            bins: int = 10, min_n: int = 5) -> CalibrationReport:
        """Fit the logit-space scaling map with Newton/IRLS and return the report."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if learning_rate <= 0 or not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if l2 < 0 or not math.isfinite(l2):
            raise ValueError("l2 must be finite and >= 0")
        if min_n < 1:
            raise ValueError("min_n must be >= 1")
        conf, out = _rows(confidences, outcomes)
        xs = [_logit(value) for value in conf]
        n = len(conf)
        a, b = 1.0, 0.0
        for _ in range(steps):
            grad_a = grad_b = 0.0
            h_aa = h_ab = h_bb = 0.0
            for x, y in zip(xs, out):
                p = _sigmoid(a * x + b)
                error = p - y
                weight = max(1e-12, p * (1.0 - p))
                grad_a += error * x
                grad_b += error
                h_aa += weight * x * x
                h_ab += weight * x
                h_bb += weight
            grad_a = grad_a / n + l2 * a
            grad_b = grad_b / n
            h_aa = h_aa / n + l2
            h_ab = h_ab / n
            h_bb = h_bb / n
            if abs(grad_a) + abs(grad_b) < tol:
                break
            determinant = h_aa * h_bb - h_ab * h_ab
            if determinant <= 1e-12:
                step_a, step_b = grad_a, grad_b
                step_scale = 0.01
            else:
                step_a = (h_bb * grad_a - h_ab * grad_b) / determinant
                step_b = (h_aa * grad_b - h_ab * grad_a) / determinant
                step_scale = learning_rate
            current = self._log_loss(xs, out, a, b, l2)
            accepted = False
            for _ in range(30):
                next_a = a - step_scale * step_a
                next_b = b - step_scale * step_b
                if math.isfinite(next_a) and math.isfinite(next_b) and \
                        self._log_loss(xs, out, next_a, next_b, l2) <= current + 1e-15:
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                break
            a, b = next_a, next_b
        self.a, self.b = a, b
        self._fitted = True
        calibrated = [self.predict(x) for x in conf]
        self.report = _report(calibrated, out, bins=bins, min_n=min_n,
                              anti_correlated=a < 0)
        return self.report

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SelfModel.predict/route requires fit() first")

    def predict(self, confidence: float) -> float:
        """The calibrated success probability for a raw confidence."""
        self._require_fitted()
        value = float(confidence)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")
        return _sigmoid(self.a * _logit(value) + self.b)

    def predict_many(self, confidences: Iterable[float]) -> Tuple[float, ...]:
        return tuple(self.predict(value) for value in confidences)

    def route(self, confidence: float, *, threshold: float = 0.5) -> str:
        """``answer`` when the calibrated probability clears ``threshold``."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return "answer" if self.predict(confidence) >= threshold else "abstain"

    def coverage(self, confidences: Iterable[float], *,
                 threshold: float = 0.5) -> float:
        values = tuple(float(x) for x in confidences)
        if not values:
            raise ValueError("coverage needs at least one confidence")
        return sum(1 for value in values if self.route(value, threshold=threshold) == "answer") / len(values)

    def raw_report(self, confidences: Iterable[float], outcomes: Iterable[float],
                   *, bins: int = 10, min_n: int = 5) -> CalibrationReport:
        """The same ECE/Brier on uncalibrated confidences, for comparison."""
        conf, out = _rows(confidences, outcomes)
        return _report(conf, out, bins=bins, min_n=min_n)


def select_by_confidence(items: Iterable[object],
                         confidence: Iterable[float], *,
                         threshold: float = 0.5,
                         model: Optional[SelfModel] = None) -> Tuple[Tuple[object, ...], Tuple[object, ...]]:
    """Split items into ``(answered, abstained)`` at a calibrated threshold.

    ``confidence`` is aligned with ``items``. Passing a fitted ``model`` routes
    on its calibrated probability; otherwise the raw confidence is thresholded.
    """
    rows = tuple(items)
    values = tuple(float(x) for x in confidence)
    if len(rows) != len(values):
        raise ValueError(f"items/confidence mismatch: {len(rows)} != {len(values)}")
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {value!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    answered: List[object] = []
    abstained: List[object] = []
    for item, value in zip(rows, values):
        probability = model.predict(value) if model is not None else value
        (answered if probability >= threshold else abstained).append(item)
    return tuple(answered), tuple(abstained)
