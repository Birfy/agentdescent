"""Where the *improvement* labels should go, which is not where the calibration ones do.

:mod:`~agentdescent.audit.sampler` allocates the calibration pool by Neyman:
``n_h ∝ W_h · sd_h`` on the residual, which minimises the variance of the
correction. The improvement pool has a different job -- find as many distinct
things wrong with the verifier as possible, so they can be fixed -- and the same
rule is wrong for it. Once thirty labels have shown the same formatting bug, the
thirty-first teaches nothing, and Neyman keeps sending labels there because that
is where the residual is largest.

What the improvement pool wants is **coverage**, and the statistic for it is
Good-Turing's unseen mass: the share of observed items seen exactly once
estimates the probability that the next draw shows something new.

    unseen = singletons / labels

Checked against the Phase 0 audit, where all 31 disagreements are in hand so the
true probability of a new mode can be computed rather than assumed:

    labels drawn    modes found    Good-Turing    true P(new)
         5             3.13           0.360          0.312
        10             4.41           0.192          0.177
        15             5.15           0.143          0.138
        20             5.82           0.123          0.122
        25             6.37           0.109          0.108
        30             6.89           0.099          0.110

It over-estimates slightly at small ``n``, which is the known behaviour, and
tracks closely from about fifteen labels on.

That table is also the argument for the whole module. Six times the labels
bought 2.2 times the modes: **five labels find three of seven error modes,
thirty find seven**. An allocation that spends the improvement budget in
proportion to how *often* a layer is wrong will keep buying the flat part of
that curve.

And frequency is not value. The most common mode in that audit is
``echoes-question`` at 12 of 31 -- and the obvious hard rule for it is the one
measured in :mod:`~agentdescent.audit.diagnose` that cuts the bias 74% and makes
the verifier *worse*. A rule that allocated by frequency would have spent the
budget confirming it.

Two asymmetries, both deliberate:

* **A key with no labels has unseen mass 1.0.** Everything about it is
  undiscovered, so it gets the largest share -- and unlike the calibration
  sampler's fill-in for a missing ``resid_sd``, this one needs no argument about
  which direction is safe: it is what the estimator says, and it self-corrects
  on the first labels that arrive. A key with *many* labels and no errors is a
  different thing entirely and gets ``unseen = 0``; the denominator that makes
  those two cases differ is in :func:`unseen_mass`, and getting it wrong sent
  the whole budget to the layer where the verifier had never once been wrong.
* **``unseen`` is undefined, not zero, with no labels at all.** Reporting zero
  would say "there is nothing left to find here", which is the opposite of what
  no evidence means.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

__all__ = ["Coverage", "CoveragePlan", "MIN_UNSEEN", "coverage_of",
           "exhausted", "plan_coverage", "rarefaction", "unseen_mass",
           "unseen_mass_overall"]

#: Below this estimated probability of a new mode, more improvement labels are
#: buying the flat part of the rarefaction curve. Not a law -- the point at which
#: a one-in-twenty chance of learning something stops being worth an oracle call
#: is a budget question -- but it is where the Phase 0 curve had flattened.
MIN_UNSEEN = 0.05


def unseen_mass(modes: Sequence[Optional[str]]) -> float:
    """Good-Turing: the probability that the next label shows an unseen mode.

    ``singletons / n``, where **n counts every label** and ``None`` is the
    species "the two scorers agreed". That denominator is the whole correctness
    of this function and the first version got it wrong: counting only the
    disagreements made a layer with ninety-seven agreeing labels and no errors
    look *unsampled*, so it scored ``1.0`` and drew the entire budget. Ninety-
    seven labels that found nothing are strong evidence there is little to find,
    not an absence of evidence.

    Good-Turing is over draws from a species distribution, and "no error" is a
    species like any other: it has been seen many times, so it is not new. The
    estimate is then exactly ``P(the next label shows an error mode never seen)``.

    NaN on an empty sample -- "no evidence" and "nothing left to find" are
    different claims and only one of them should stop a caller sampling.
    """
    if not modes:
        return float("nan")
    counts = Counter(m for m in modes if m is not None)
    return sum(1 for v in counts.values() if v == 1) / len(modes)


@dataclass(frozen=True)
class Coverage:
    """What one key has taught so far, and how much it still has to teach."""

    key: str
    #: Resolved improvement labels with this key.
    labels: int
    #: Distinct error modes among them.
    modes: int
    #: Modes seen exactly once -- the numerator of :attr:`unseen`.
    singletons: int
    #: Good-Turing estimate of ``P(next label shows a new mode)``. ``1.0`` for a
    #: key with no labels: all of its mass is undiscovered.
    unseen: float

    @property
    def exhausted(self) -> bool:
        return self.unseen == self.unseen and self.unseen < MIN_UNSEEN


@dataclass(frozen=True)
class CoveragePlan:
    """Per-key inclusion probabilities for the improvement pool.

    The same shape as :class:`~agentdescent.audit.sampler.SamplePlan`, and
    deliberately a different class: they answer different questions and mixing
    them up produces a plan that is merely suboptimal rather than an error, which
    is the kind that survives review.
    """

    #: key -> inclusion probability in ``[0, 1]``.
    rates: Dict[str, float]
    #: key -> labels the allocation asked for.
    target_n: Dict[str, int]
    #: key -> population share used to plan.
    weights: Dict[str, float]
    #: key -> Good-Turing unseen mass used to plan.
    unseen: Dict[str, float]
    #: Rate for a key the plan never saw. Zero, for the same reason
    #: :class:`~agentdescent.audit.sampler.SamplePlan` uses zero: sampling under
    #: a probability nobody recorded is worse than not sampling.
    default_rate: float = 0.0
    total_n: int = 0
    expected_units: int = 0
    #: Overall ``P(a new label shows a new mode)`` across every key.
    unseen_overall: float = float("nan")
    warnings: List[str] = field(default_factory=list)

    def rate_for(self, key: str) -> float:
        return self.rates.get(key, self.default_rate)

    @property
    def done(self) -> bool:
        """Is the improvement pool finished learning?

        True when the next label is unlikely to show anything new *anywhere*. The
        budget then belongs in the calibration pool, which never saturates --
        its interval keeps narrowing.
        """
        u = self.unseen_overall
        return u == u and u < MIN_UNSEEN


def coverage_of(records: Iterable[Any],
                key: Callable[[Any], str],
                mode: Callable[[Any], Optional[str]],
                *, keys: Sequence[str] = ()) -> Dict[str, Coverage]:
    """Group resolved records by ``key`` and measure the variety inside each.

    ``mode(record) -> str`` is the error signature: what a person would have to
    fix. It runs only on records where the two scorers disagreed, and may return
    ``None`` for one it cannot classify, which is then not counted as a mode --
    an unclassified error is not evidence that a key is exhausted.

    An **agreeing** label still counts toward ``labels``, and that is what makes
    the estimate mean anything: a key whose hundred labels all agreed has strong
    evidence that there is little left to find there, and counting only its
    disagreements would make it look unsampled.

    ``keys`` names keys that exist in the population but have no labels yet.
    They come back with ``unseen = 1.0``, which is what puts the budget on them.
    """
    grouped: Dict[str, List[Optional[str]]] = {k: [] for k in keys}
    seen_keys = set(keys)
    for rec in records:
        if getattr(rec, "oracle_score", None) is None:
            continue
        k = key(rec)
        seen_keys.add(k)
        grouped.setdefault(k, [])
        # `None` is a draw on the species "they agreed" -- see `unseen_mass`.
        grouped[k].append(None if rec.verifier_score == rec.oracle_score
                          else mode(rec))

    out: Dict[str, Coverage] = {}
    for k in sorted(seen_keys):
        drawn = grouped.get(k, [])
        counts = Counter(m for m in drawn if m is not None)
        out[k] = Coverage(
            key=k, labels=len(drawn), modes=len(counts),
            singletons=sum(1 for v in counts.values() if v == 1),
            unseen=unseen_mass(drawn) if drawn else 1.0)
    return out


def exhausted(coverage: Mapping[str, Coverage],
              min_unseen: float = MIN_UNSEEN) -> List[str]:
    """Keys where the next label is unlikely to show anything new."""
    return sorted(k for k, c in coverage.items()
                  if c.unseen == c.unseen and c.unseen < min_unseen)


def plan_coverage(*, weights: Mapping[str, float], expected_units: int,
                  coverage: Mapping[str, Coverage],
                  target_n: int = 100, min_per_key: int = 5,
                  max_rate: float = 1.0) -> CoveragePlan:
    """Allocate ``target_n`` improvement labels by how much each key can still teach.

    ``n_k ∝ W_k · unseen_k`` -- the same shape as the calibration sampler's
    ``n_h ∝ W_h · sd_h`` and a different quantity, which is the whole point.
    Neyman minimises the variance of a number; this maximises the count of
    distinct things discovered.

    Floors are applied **after** the allocation, for the same reason they are
    there: a key allocated one label contributes a mode count from one point.
    Rates are then ``n_k / (W_k · expected_units)``, capped at ``max_rate``.
    """
    warnings: List[str] = []
    weights = {k: float(v) for k, v in weights.items() if v > 0.0}
    if not weights:
        return CoveragePlan({}, {}, {}, {}, warnings=["no keys to plan for"])
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-6:
        warnings.append(
            f"weights sum to {total_w:.6f}, not 1; normalising, but a population "
            f"share that does not add up usually means a key was dropped")
        weights = {k: v / total_w for k, v in weights.items()}

    unseen = {k: (coverage[k].unseen if k in coverage else 1.0)
              for k in weights}
    for k, u in list(unseen.items()):
        if u != u:                                  # NaN
            unseen[k] = 1.0
    share = {k: weights[k] * unseen[k] for k in weights}
    denom = sum(share.values())
    if denom <= 0.0:
        warnings.append(
            "every key is exhausted; the improvement pool has learnt what it "
            "can and the budget belongs in the calibration pool, whose interval "
            "keeps narrowing")
        share = {k: weights[k] for k in weights}
        denom = sum(share.values())

    allocated = {k: int(math.ceil(target_n * share[k] / denom)) for k in weights}
    for k in allocated:
        allocated[k] = max(allocated[k], min_per_key)

    rates: Dict[str, float] = {}
    for k, n_k in allocated.items():
        available = weights[k] * max(0, expected_units)
        if available <= 0:
            rates[k] = 0.0
            warnings.append(f"key {k!r} is expected to see no units")
            continue
        rate = n_k / available
        if rate > max_rate:
            warnings.append(
                f"key {k!r} wants {n_k} of an expected {available:.0f} units; "
                f"capped at {max_rate:.2f}, so it will come up short")
        rates[k] = min(max_rate, rate)

    overall = unseen_mass_overall(coverage)
    if overall == overall and overall < MIN_UNSEEN:
        warnings.append(
            f"P(a new label shows a new error mode) is {overall:.3f}; more "
            f"improvement labels are buying the flat part of the curve")

    return CoveragePlan(
        rates=rates, target_n=allocated, weights=dict(weights), unseen=unseen,
        total_n=sum(allocated.values()), expected_units=expected_units,
        unseen_overall=overall, warnings=warnings)


def unseen_mass_overall(coverage: Mapping[str, Coverage]) -> float:
    """Good-Turing across every key, pooled by label count.

    Pooled rather than averaged: a key with two labels and a key with two
    hundred are not two equal opinions about how much is left to find.
    """
    labels = sum(c.labels for c in coverage.values())
    if not labels:
        return float("nan")
    return sum(c.singletons for c in coverage.values()) / labels


def rarefaction(modes: Sequence[str], sizes: Sequence[int], *,
                reps: int = 200, seed: int = 0) -> List[Tuple[int, float]]:
    """``[(m, mean distinct modes in a sample of m)]`` -- the diminishing return.

    The curve is the argument the module is built on, so it is computable rather
    than quoted: run it on your own labels before deciding a bigger improvement
    budget will find anything.
    """
    rng = random.Random(seed)
    out = []
    for m in sizes:
        if m <= 0 or m > len(modes):
            continue
        found = [len(set(rng.sample(list(modes), m))) for _ in range(reps)]
        out.append((m, sum(found) / len(found)))
    return out
