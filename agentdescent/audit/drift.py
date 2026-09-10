"""Watching the correction over generations, without alarming every generation.

One rectification says how biased the verifier is now. A sequence of them says
something the single number cannot: whether the loop is *finding* the verifier's
blind spots. A ``delta_hat`` that walks steadily upward is the signature of a
population drifting into whatever the proxy likes -- the failure this package
exists to catch, and the one no single measurement can show.

**The wrong way to watch it is a test per generation.** At alpha = 0.05 that
alarms once every twenty generations *when nothing is wrong*, by construction;
over a hundred generations the expected count of false alarms is five, and an
operator who has seen five false alarms does not act on the sixth. The plan says
alpha spending or EWMA. This is EWMA, which is the one that also smooths.

    z_t = lam * x_t + (1 - lam) * z_{t-1}

``L`` is not a per-generation significance level: it sets the **average run
length** between false alarms. Measured over two thousand runs of a hundred
in-control generations each:

===========================  =========================  =====================
                             alarms per 100 generations  clean runs that alarm
===========================  =========================  =====================
a two-sided test per gen      4.95                       **99.3%**
this chart (lam=.2, L=3)      0.27                       16.2%
===========================  =========================  =====================

`test_a_naive_test_per_generation_cries_wolf` is that measurement, kept as a
test so the constants above cannot be tuned without it noticing.

Two things this does differently from the textbook chart, both because the
inputs are estimates rather than measurements:

**The limits are recursive, not closed-form.** The usual
``sigma sqrt(lam/(2-lam) (1-(1-lam)^2t))`` assumes every point has the same
standard error. Here each ``delta_hat`` arrives with its own ``se``, which grows
and shrinks with how many labels that generation bought, so the variance is
carried forward exactly instead::

    Var(z_t) = lam**2 se_t**2 + (1 - lam)**2 Var(z_{t-1})

**Overlapping label sets invalidate the chart, and are the normal case.**
:class:`~agentdescent.audit.calibrator.Calibrator` recomputes from the whole
store, so two consecutive rectifications share most of their labels and are
strongly positively correlated. That makes the true variance of ``z`` *larger*
than the recursion above, so the limits are too tight and the chart alarms on a
verifier that never moved. :class:`DriftMonitor` reads ``Rectification.covers``,
notices, and says so rather than charting anyway. Feed it rectifications
computed on disjoint windows -- one generation's labels each -- and the chart is
valid.

``gain_factor`` is watched too, and differently: it has no standard error, so
there is nothing to put limits around. It is smoothed and compared against a
threshold, and the threshold is near 1 because that is where the verifier stops
carrying usable information about the truth. The right response there is a
different verifier, not a bigger audit, and :class:`DriftSignal` says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .calibrator import Rectification

__all__ = ["DEFAULT_L", "DEFAULT_LAMBDA", "MIN_GAIN", "DriftKind",
           "DriftMonitor", "DriftPoint", "DriftReport", "DriftSignal", "EWMA"]

#: Smoothing weight. 0.2 is the usual choice for detecting a shift of about one
#: standard error: small enough to average away a single noisy generation, large
#: enough that a real shift is not smoothed into invisibility.
DEFAULT_LAMBDA = 0.2
#: Control limit in standard errors of the EWMA. Not a per-generation alpha --
#: see the module docstring.
DEFAULT_L = 3.0
#: Below this smoothed ``gain_factor`` the verifier is no longer carrying usable
#: information about the truth. 1.0 means the unlabelled scores contributed
#: nothing at all; the margin is there because an estimate of 1.02 is not
#: distinguishable from 1.00.
MIN_GAIN = 1.05


class DriftKind(str, Enum):
    """What a signal is telling the operator to do."""

    #: ``delta_hat`` is walking upward: the verifier is being over-generous more
    #: often than it was, which is what a population drifting into a gaming
    #: signature looks like from here.
    BIAS_UP = "bias-up"
    #: Downward. Less dangerous -- a harsher verifier costs progress, not
    #: soundness -- but the instrument still moved and the history is mixed.
    BIAS_DOWN = "bias-down"
    #: The verifier no longer predicts the truth well enough to be worth
    #: borrowing from. Replace the verifier; a bigger audit does not fix this.
    SIGNAL_LOST = "signal-lost"
    #: The chart cannot be read, because consecutive points share labels.
    NOT_INDEPENDENT = "not-independent"


@dataclass(frozen=True)
class DriftPoint:
    """One generation's rectification, reduced to what a chart needs."""

    label: str
    delta_hat: float
    se: float
    gain_factor: float
    n: int
    covers: Tuple[float, float] = (0.0, 0.0)

    @classmethod
    def of(cls, rect: Rectification, label: str = "") -> "DriftPoint":
        return cls(label=label or f"{rect.verifier_version}@{rect.computed_at:.0f}",
                   delta_hat=rect.delta_hat, se=rect.se,
                   gain_factor=rect.gain_factor, n=rect.n, covers=rect.covers)


@dataclass(frozen=True)
class DriftSignal:
    """One alarm, with the number that raised it and what to do."""

    kind: DriftKind
    at: int
    label: str
    value: float
    #: The smoothed statistic, and the limit it crossed. ``limit`` is a
    #: half-width around the centre for the bias charts and an absolute floor
    #: for :attr:`DriftKind.SIGNAL_LOST`.
    z: float
    limit: float
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind.value}] {self.label}: {self.detail}"


class EWMA:
    """``z_t`` and its exact variance when every point brings its own ``se``.

    Stateful and single-purpose: :meth:`observe` returns the smoothed value and
    the half-width of the control band at that point. The band widens with a
    noisy generation and narrows with a well-audited one, which a closed-form
    chart cannot express.
    """

    __slots__ = ("lam", "L", "centre", "z", "var", "t")

    def __init__(self, *, lam: float = DEFAULT_LAMBDA, L: float = DEFAULT_L,
                 centre: float = 0.0) -> None:
        if not 0.0 < lam <= 1.0:
            raise ValueError("lam must be in (0, 1]")
        if L <= 0.0:
            raise ValueError("L must be positive")
        self.lam, self.L, self.centre = lam, L, centre
        self.z, self.var, self.t = centre, 0.0, 0

    def observe(self, value: float, se: float) -> Tuple[float, float]:
        """Fold in one point. Returns ``(z, halfwidth)``."""
        self.t += 1
        self.z = self.lam * value + (1.0 - self.lam) * self.z
        self.var = (self.lam ** 2) * (se ** 2) \
            + ((1.0 - self.lam) ** 2) * self.var
        return self.z, self.L * math.sqrt(self.var)


@dataclass
class DriftReport:
    """Every point charted, every signal raised, and whether the chart is valid."""

    points: List[DriftPoint] = field(default_factory=list)
    z_bias: List[float] = field(default_factory=list)
    band: List[float] = field(default_factory=list)
    z_gain: List[float] = field(default_factory=list)
    signals: List[DriftSignal] = field(default_factory=list)
    centre: float = 0.0
    #: ``True`` once consecutive points have been seen to share labels. Every
    #: number above is still reported -- withholding them would hide the drift
    #: as thoroughly as a false alarm would bury it -- but none of the bias
    #: signals mean what they say.
    overlapping: bool = False

    @property
    def alarming(self) -> bool:
        return any(s.kind is not DriftKind.NOT_INDEPENDENT for s in self.signals)

    def of_kind(self, kind: DriftKind) -> List[DriftSignal]:
        return [s for s in self.signals if s.kind is kind]

    def to_markdown(self) -> str:
        rows = [f"# Drift -- {len(self.points)} generations", ""]
        if self.overlapping:
            rows += [
                "!!! warning \"The control limits do not apply\"",
                "    Consecutive rectifications share labels, so the points are",
                "    positively correlated and the true spread of `z` is wider",
                "    than the band below. Recompute each generation on its own",
                "    window before reading an alarm as one.",
                "",
            ]
        rows += ["| generation | n | delta_hat | se | z | band | gain | z(gain) |",
                 "|---|---|---|---|---|---|---|---|"]
        for i, p in enumerate(self.points):
            rows.append(
                f"| {p.label} | {p.n} | {p.delta_hat:+.4f} | {p.se:.4f} | "
                f"{self.z_bias[i]:+.4f} | ±{self.band[i]:.4f} | "
                f"{p.gain_factor:.3f} | {self.z_gain[i]:.3f} |")
        if self.signals:
            rows += ["", "## Signals", ""]
            rows += [f"- {s}" for s in self.signals]
        else:
            rows += ["", "No signal. The verifier's bias has not moved further "
                     "than its own measurement error over these generations."]
        return "\n".join(rows)


class DriftMonitor:
    """EWMA charts on ``delta_hat`` and ``gain_factor``, generation by generation.

    ``centre`` is where ``delta_hat`` is expected to sit. Left ``None`` it is
    taken from the first point, which makes the chart ask "has the bias moved
    since we started watching" rather than "is the bias zero" -- the second is a
    question the calibrator already answers and a control chart answers badly.
    """

    def __init__(self, *, lam: float = DEFAULT_LAMBDA, L: float = DEFAULT_L,
                 centre: Optional[float] = None, min_gain: float = MIN_GAIN,
                 gain_lam: float = DEFAULT_LAMBDA) -> None:
        self.lam, self.L, self.min_gain = lam, L, min_gain
        self.centre = centre
        self._bias: Optional[EWMA] = None
        self._gain: Optional[EWMA] = None
        self.report = DriftReport()

    def observe(self, rect: Rectification, label: str = "") -> List[DriftSignal]:
        """Chart one generation. Returns only the signals *this* point raised.

        A stale rectification is skipped rather than charted: it carries the
        previous generation's numbers, so charting it would plot the same point
        twice and read as a verifier that had gone quiet.
        """
        if rect.is_stale:
            return []
        point = DriftPoint.of(rect, label)
        if point.delta_hat != point.delta_hat:            # NaN
            return []
        return self._observe_point(point)

    def observe_point(self, point: DriftPoint) -> List[DriftSignal]:
        """Chart a point assembled by hand. For a caller that is not using
        :class:`~agentdescent.audit.calibrator.Calibrator`."""
        return self._observe_point(point)

    def _observe_point(self, point: DriftPoint) -> List[DriftSignal]:
        report, fresh = self.report, []
        previous = report.points[-1] if report.points else None
        if self.centre is None:
            self.centre = point.delta_hat
        if self._bias is None:
            self._bias = EWMA(lam=self.lam, L=self.L, centre=self.centre)
            self._gain = EWMA(lam=self.lam, L=self.L, centre=point.gain_factor)
            report.centre = self.centre

        index = len(report.points)
        if previous is not None and _overlaps(previous.covers, point.covers):
            if not report.overlapping:
                report.overlapping = True
                fresh.append(DriftSignal(
                    DriftKind.NOT_INDEPENDENT, index, point.label, point.delta_hat,
                    0.0, 0.0,
                    f"this generation's labels span {point.covers} and the "
                    f"previous one's {previous.covers}; the points share labels, "
                    f"so the control limits below are too tight and an alarm "
                    f"here is not evidence of drift"))

        z, band = self._bias.observe(point.delta_hat, point.se)
        gain = point.gain_factor
        z_gain, _ = self._gain.observe(
            gain if gain == gain else self._gain.z, 0.0)

        report.points.append(point)
        report.z_bias.append(z)
        report.band.append(band)
        report.z_gain.append(z_gain)

        if band > 0.0 and z > self.centre + band:
            fresh.append(DriftSignal(
                DriftKind.BIAS_UP, index, point.label, point.delta_hat, z, band,
                f"the verifier's bias has drifted to {z:+.4f}, past "
                f"{self.centre + band:+.4f}; the loop is finding answers this "
                f"verifier likes and the truth does not"))
        elif band > 0.0 and z < self.centre - band:
            fresh.append(DriftSignal(
                DriftKind.BIAS_DOWN, index, point.label, point.delta_hat, z, band,
                f"the verifier's bias has drifted to {z:+.4f}, below "
                f"{self.centre - band:+.4f}; it is harsher than it was, which "
                f"costs progress rather than soundness -- but the recorded "
                f"history now mixes two instruments"))

        if z_gain == z_gain and z_gain < self.min_gain and not any(
                s.kind is DriftKind.SIGNAL_LOST for s in report.signals):
            fresh.append(DriftSignal(
                DriftKind.SIGNAL_LOST, index, point.label, gain, z_gain,
                self.min_gain,
                f"gain_factor has fallen to {z_gain:.3f}; the verifier's scores "
                f"no longer predict the truth well enough to borrow from. "
                f"Replace the verifier -- more labels do not fix this, they "
                f"only pay for what the verifier stopped contributing"))

        report.signals.extend(fresh)
        return fresh

    def observe_all(self, rects: Sequence[Rectification]) -> DriftReport:
        for rect in rects:
            self.observe(rect)
        return self.report


def _overlaps(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Do two ``covers`` ranges share any time at all?

    ``(0.0, 0.0)`` means "not recorded", and two unrecorded ranges are not
    evidence of overlap -- a caller charting hand-built points should not be
    told its chart is invalid because it left a field blank.
    """
    if a == (0.0, 0.0) or b == (0.0, 0.0):
        return False
    return a[0] <= b[1] and b[0] <= a[1]
