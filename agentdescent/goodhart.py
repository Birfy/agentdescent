"""Detect when the proxy reward stops tracking the thing we care about.

A self-improving loop optimises the reward it can measure. If that reward is a
proxy -- a cheap verifier, a unit-test pass rate, an LLM judge -- the candidates
will eventually find the gap between the proxy and the oracle. Goodhart's law is
not "the metric is bad"; it is "the correlation you calibrated on is no longer
the correlation at the frontier you reached".

This module makes the gap a first-class, cheap read model. It takes **paired
gains** -- the proxy gain and the oracle gain of the same candidate -- and
reports three things a single mean hides:

* **inflation** -- how much higher the proxy gain is than the oracle gain;
* **rank correlation** -- whether the proxy still orders candidates the way the
  oracle does (Spearman, tie-aware);
* **shortcut rate** -- the fraction of candidates the proxy calls an improvement
  while the oracle says they did not improve.

A candidate flagged as a shortcut is the concrete artifact to inspect; the
report is not a verdict on the reward model in the abstract.

    report = detect_goodhart(proxy_gains, oracle_gains)
    if report.verdict == "gaming":
        for row in report.shortcuts:
            print(row.proxy_gain, row.oracle_gain, row.task)

The monitor form accumulates a daily stream and can report a trailing window
(``report(last=30)``), so a dashboard can alert on *drift in the gap* rather
than on all history or a single run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "GoodhartPair",
    "GoodhartReport",
    "GoodhartMonitor",
    "detect_goodhart",
    "spearman",
]


@dataclass(frozen=True)
class GoodhartPair:
    """One candidate's proxy gain and oracle gain."""

    proxy_gain: float
    oracle_gain: float
    task: str = ""


@dataclass(frozen=True)
class GoodhartReport:
    """The proxy-oracle gap on one batch of paired gains."""

    n: int
    proxy_mean: float
    oracle_mean: float
    inflation: float
    rank_correlation: Optional[float]
    shortcut_rate: float
    verdict: str
    shortcuts: Tuple[GoodhartPair, ...] = ()
    detail: str = ""

    @property
    def aligned(self) -> bool:
        return self.verdict == "aligned"

    @property
    def gaming(self) -> bool:
        return self.verdict == "gaming"


def _pairs(proxy_gains: Iterable[float], oracle_gains: Iterable[float],
           tasks: Optional[Sequence[str]] = None) -> List[GoodhartPair]:
    proxies = [float(x) for x in proxy_gains]
    oracles = [float(x) for x in oracle_gains]
    if len(proxies) != len(oracles):
        raise ValueError(f"proxy/oracle length mismatch: {len(proxies)} != {len(oracles)}")
    if tasks is not None and len(tasks) != len(proxies):
        raise ValueError(f"tasks length mismatch: {len(tasks)} != {len(proxies)}")
    if not proxies:
        raise ValueError("at least one paired gain is required")
    for value in proxies + oracles:
        if not math.isfinite(value):
            raise ValueError(f"gains must be finite, got {value!r}")
    labels = list(tasks) if tasks is not None else [""] * len(proxies)
    return [GoodhartPair(p, o, str(label or "")) for p, o, label in zip(proxies, oracles, labels)]


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, so ties do not depend on input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    n = len(left)
    mean_l = sum(left) / n
    mean_r = sum(right) / n
    cov = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
    var_l = sum((a - mean_l) ** 2 for a in left)
    var_r = sum((b - mean_r) ** 2 for b in right)
    if var_l == 0 or var_r == 0:
        return 0.0
    return cov / math.sqrt(var_l * var_r)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Tie-aware Spearman rank correlation; 0.0 when either side is constant."""
    if len(left) != len(right):
        raise ValueError("spearman inputs must have equal length")
    if not left:
        raise ValueError("spearman needs at least one pair")
    return _pearson(_ranks(left), _ranks(right))


def detect_goodhart(proxy_gains: Sequence[float], oracle_gains: Sequence[float], *,
                    tasks: Optional[Sequence[str]] = None,
                    inflation_tolerance: float = 0.05,
                    min_rank_correlation: float = 0.5,
                    max_shortcut_rate: float = 0.2,
                    max_shortcuts: int = 5,
                    min_n: int = 3) -> GoodhartReport:
    """Measure whether the proxy still tracks the oracle at this frontier.

    A pair is a **shortcut** when the proxy reports a positive gain and the
    oracle reports no gain. ``verdict`` is ``gaming`` when inflation exceeds
    ``inflation_tolerance`` and either the rank correlation or the shortcut rate
    is past its bar; ``suspicious`` when any one signal is past its bar;
    ``aligned`` otherwise.
    """
    if min_rank_correlation < -1.0 or min_rank_correlation > 1.0:
        raise ValueError("min_rank_correlation must be in [-1, 1]")
    if not 0.0 <= max_shortcut_rate <= 1.0:
        raise ValueError("max_shortcut_rate must be in [0, 1]")
    if inflation_tolerance < 0:
        raise ValueError("inflation_tolerance must be >= 0")
    if max_shortcuts < 0:
        raise ValueError("max_shortcuts must be >= 0")
    if min_n < 1:
        raise ValueError("min_n must be >= 1")
    pairs = _pairs(proxy_gains, oracle_gains, tasks)
    proxies = [pair.proxy_gain for pair in pairs]
    oracles = [pair.oracle_gain for pair in pairs]
    proxy_mean = sum(proxies) / len(pairs)
    oracle_mean = sum(oracles) / len(pairs)
    inflation = proxy_mean - oracle_mean
    constant = len(set(proxies)) == 1 or len(set(oracles)) == 1
    rank: Optional[float] = None if len(pairs) < min_n or constant else spearman(proxies, oracles)
    shortcuts = [pair for pair in pairs if pair.proxy_gain > 0 and pair.oracle_gain <= 0]
    shortcut_rate = len(shortcuts) / len(pairs)
    inflation_bad = inflation > inflation_tolerance
    rank_bad = rank is not None and rank < min_rank_correlation
    shortcut_bad = shortcut_rate > max_shortcut_rate
    if len(pairs) < min_n:
        verdict = "insufficient"
    elif inflation_bad and (rank_bad or shortcut_bad):
        verdict = "gaming"
    elif inflation_bad or rank_bad or shortcut_bad:
        verdict = "suspicious"
    else:
        verdict = "aligned"
    shortcuts.sort(key=lambda pair: (-pair.proxy_gain, pair.oracle_gain))
    rank_text = "n/a" if rank is None else f"{rank:.3f}"
    detail = (f"inflation={inflation:+.4f}, rank={rank_text}, "
              f"shortcuts={len(shortcuts)}/{len(pairs)}")
    return GoodhartReport(n=len(pairs), proxy_mean=proxy_mean,
                          oracle_mean=oracle_mean, inflation=inflation,
                          rank_correlation=rank, shortcut_rate=shortcut_rate,
                          verdict=verdict,
                          shortcuts=tuple(shortcuts[:max_shortcuts]),
                          detail=detail)


@dataclass
class GoodhartMonitor:
    """Accumulate daily paired gains and report the running gap.

    The monitor keeps every pair so ``report`` can be recomputed with a
    different threshold without losing history. ``report(last=N)`` reports only
    the trailing window; a caller that needs bounded memory can keep its own
    deque and call :func:`detect_goodhart` directly.
    """

    pairs: List[GoodhartPair] = field(default_factory=list)

    def observe(self, proxy_gain: float, oracle_gain: float, task: str = "") -> None:
        proxy = float(proxy_gain)
        oracle = float(oracle_gain)
        if not math.isfinite(proxy) or not math.isfinite(oracle):
            raise ValueError("proxy/oracle gains must be finite")
        self.pairs.append(GoodhartPair(proxy, oracle, str(task)))

    def clear(self) -> None:
        self.pairs.clear()

    def report(self, *, last: Optional[int] = None, **kwargs) -> GoodhartReport:
        pairs = self.pairs[-last:] if last is not None and last > 0 else self.pairs
        if last is not None and last < 1:
            raise ValueError("last must be >= 1")
        return detect_goodhart([p.proxy_gain for p in pairs],
                               [p.oracle_gain for p in pairs],
                               tasks=[p.task for p in pairs], **kwargs)
