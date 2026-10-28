"""Anytime-valid evidence for a stream of self-improvement deltas.

A fixed-sample p-value is only valid at the sample size it was computed for.
Dashboards, controllers, and daily update loops look at the same evidence
repeatedly, and a 5% test peeked at twenty times is not a 5% test. The
algorithmic answer is an **e-process**: a non-negative martingale whose value is
a running likelihood ratio against "the candidate does not improve". By
Ville's inequality, ``P(sup_t e_t >= 1/alpha) <= alpha`` under the null, so a
threshold can be watched continuously without alpha inflation.

This module implements a simple hedged betting e-process for bounded paired
deltas. Each delta is mapped from ``[-1, 1]`` to ``[0, 1]``; under the null the
mapped mean is at most ``0.5``. A grid of betting fractions ``lambda`` each
accumulates ``1 + lambda * (x - 0.5)``; the e-value is the average over the
grid. The average of valid e-processes is still an e-process, and the grid makes
the test robust to not knowing the true effect size in advance.

    stream = EProcess(alpha=0.05)
    for delta in observed_improvements:
        report = stream.update(delta)
        if report.decision == "accept":
            break

``delta`` is the paired improvement of candidate over incumbent on one task,
clipped to ``[-1, 1]``. For unbounded scores, normalise before calling; the
bounded assumption is what gives the guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

__all__ = [
    "EProcess",
    "SequentialReport",
    "sequential_test",
]

_DEFAULT_GRID: Tuple[float, ...] = tuple(i / 100 for i in range(1, 50))


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def _map_delta(delta: float) -> float:
    value = float(delta)
    if not math.isfinite(value):
        raise ValueError(f"delta must be finite, got {delta!r}")
    return 0.5 + 0.5 * max(-1.0, min(1.0, value))


def _log_mean_exp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty log grid")
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values) / len(values))


@dataclass(frozen=True)
class SequentialReport:
    """The state of an e-process after one observation."""

    e_value: float
    log_e: float
    p_value: float
    alpha: float
    n: int
    mean: float
    decision: str
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"


class EProcess:
    """A hedged betting e-process for paired improvement deltas.

    Parameters
    ----------
    alpha:
        Target anytime-valid false-positive rate.
    betting_grid:
        Betting fractions in ``(0, 0.5)``. The default is ``0.01..0.49``.
    max_n:
        When set, an unaccepted stream returns ``decision="reject"`` at this
        many observations. Without it the process stays ``"continue"`` forever,
        which is honest: an e-process has no fixed sample size.
    """

    def __init__(self, alpha: float = 0.05, *,
                 betting_grid: Optional[Sequence[float]] = None,
                 max_n: Optional[int] = None) -> None:
        self.alpha = _validate_alpha(alpha)
        grid = _DEFAULT_GRID if betting_grid is None else tuple(float(x) for x in betting_grid)
        if not grid:
            raise ValueError("betting_grid must not be empty")
        for value in grid:
            if not 0.0 < value < 0.5:
                raise ValueError(f"betting fractions must be in (0, 0.5), got {value}")
        if max_n is not None and max_n < 1:
            raise ValueError("max_n must be >= 1")
        self.betting_grid = grid
        self.max_n = max_n
        self._logs = [0.0 for _ in grid]
        self._n = 0
        self._sum = 0.0

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    @property
    def log_e(self) -> float:
        return _log_mean_exp(self._logs)

    @property
    def e_value(self) -> float:
        # exp(710) overflows; saturating keeps a long winning stream reportable
        # (p-value is already 0 by then) without an OverflowError in a dashboard.
        return math.exp(min(self.log_e, 709.0))

    def reset(self) -> "EProcess":
        self._logs = [0.0 for _ in self.betting_grid]
        self._n = 0
        self._sum = 0.0
        return self

    def update(self, delta: float) -> SequentialReport:
        """Consume one paired improvement and report the running evidence."""
        x = _map_delta(delta)
        actual = float(delta)
        for index, lam in enumerate(self.betting_grid):
            self._logs[index] += math.log1p(lam * (x - 0.5))
        self._n += 1
        self._sum += actual
        return self.report()

    def report(self) -> SequentialReport:
        log_e = self.log_e
        e_value = self.e_value
        p_value = min(1.0, 1.0 / e_value)
        if self._n and p_value <= self.alpha:
            decision = "accept"
            detail = f"e={e_value:.4g} >= 1/alpha={1 / self.alpha:.4g}"
        elif self.max_n is not None and self._n >= self.max_n:
            decision = "reject"
            detail = f"n={self._n} reached max_n without e >= 1/alpha"
        else:
            decision = "continue"
            detail = f"e={e_value:.4g}, threshold={1 / self.alpha:.4g}"
        return SequentialReport(e_value=e_value, log_e=log_e,
                                p_value=p_value, alpha=self.alpha, n=self._n,
                                mean=self.mean, decision=decision,
                                detail=detail)

    def accept(self) -> bool:
        return self.report().accepted


def sequential_test(deltas: Iterable[float], *, alpha: float = 0.05,
                    max_n: Optional[int] = None) -> SequentialReport:
    """Feed an iterable into a fresh :class:`EProcess` and return the report."""
    stream = EProcess(alpha=alpha, max_n=max_n)
    for delta in deltas:
        report = stream.update(delta)
        if report.decision != "continue":
            return report
    return stream.report()
