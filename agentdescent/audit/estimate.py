"""Design-based estimation of the verifier's bias from a probability sample.

What this is: the **Hajek** (inclusion-probability-weighted) mean of the residual
``f - Y`` over audited units, with a percentile bootstrap interval. It makes no
assumption about the workload -- only that every unit's inclusion probability was
recorded, which :class:`~agentdescent.audit.tap.AuditedReward` guarantees.

What this is **not**: prediction-powered inference. PPI additionally borrows
strength from the *unlabelled* verifier scores -- the units that were scored by
`f` but never sent to the oracle -- to shrink the interval, using a cross-fitted
coefficient, per-stratum weights and a `t` quantile. That machinery earns its
keep when labels are scarce and unlabelled scores are plentiful, it has several
ways to be subtly wrong that a coverage test will not catch, and it belongs in a
module with golden vectors and mutation tests.

The two are not alternatives so much as a baseline and a refinement. **This is
the baseline**, and it is the number a PPI estimate has to beat: same point
estimate in expectation, wider interval. Phase 0 of the audit plan only has to
decide whether the bias exists and whether it is large next to the noise the
acceptance gate already carries, and for that the baseline is enough -- so it
ships first, and any estimator added later is measured against it rather than
trusted over it.
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["hajek_mean", "bootstrap_ci", "standard_error", "residual_bias"]


def hajek_mean(values: Sequence[float], probs: Sequence[float]) -> float:
    """Inclusion-probability-weighted mean -- the Hajek ratio estimator.

    ``sum(v_i / p_i) / sum(1 / p_i)``. With one probability shared by every unit
    this reduces to the plain mean; the weighting only bites under stratified or
    score-dependent sampling, where the audited units are deliberately **not**
    representative of the population and averaging them raw estimates the wrong
    thing.

    Written weighted from the start for that reason: an unweighted mean is
    correct today, silently wrong the day someone raises the sampling rate on the
    boundary stratum, and produces a plausible number either way.

    The Hajek form rather than Horvitz-Thompson (``sum(v_i / p_i) / N``) because
    the population size ``N`` is not known here -- the tap sees units as they are
    scored, and how many there will be depends on how the run goes. Dividing by
    the estimated size instead is the standard remedy and is also less variable
    when the probabilities differ a lot.
    """
    if len(values) != len(probs):
        raise ValueError(f"{len(values)} values but {len(probs)} probabilities")
    if not values:
        return float("nan")
    if any(p <= 0.0 for p in probs):
        raise ValueError("inclusion probabilities must be strictly positive")
    den = sum(1.0 / p for p in probs)
    return sum(v / p for v, p in zip(values, probs)) / den


def _bootstrap_means(values: Sequence[float], probs: Sequence[float],
                     draws: int, seed: int,
                     clusters: Optional[Sequence] = None) -> List[float]:
    """Resampled Hajek means -- by unit, or by cluster when units are not independent.

    The unit bootstrap assumes the audited units are independent draws. In a run
    they are not: the same task is scored again for every artifact version, and
    those scores share whatever makes that task easy or hard for the judge. Pairs
    from one task therefore carry less information than the count suggests, and a
    unit bootstrap reports an interval that is too narrow -- by roughly the square
    root of the average units per cluster, which in a real run is 2-3x.

    Pass ``clusters`` (task ids) to resample **clusters with replacement**, taking
    every unit in each drawn cluster. That is the standard remedy and it is
    conservative in the right direction: an interval that is too wide delays a
    decision, one that is too narrow makes it wrongly.
    """
    rng = random.Random(seed)
    out = []
    if clusters is None:
        n = len(values)
        for _ in range(draws):
            idx = [rng.randrange(n) for _ in range(n)]
            out.append(hajek_mean([values[i] for i in idx], [probs[i] for i in idx]))
        return out

    groups: Dict[object, List[int]] = {}
    for i, key in enumerate(clusters):
        groups.setdefault(key, []).append(i)
    keys = list(groups)
    k = len(keys)
    for _ in range(draws):
        idx: List[int] = []
        for _ in range(k):
            idx.extend(groups[keys[rng.randrange(k)]])
        out.append(hajek_mean([values[i] for i in idx], [probs[i] for i in idx]))
    return out


def bootstrap_ci(values: Sequence[float], probs: Sequence[float], *,
                 draws: int = 5000, alpha: float = 0.05, seed: int = 0,
                 clusters: Optional[Sequence] = None) -> Tuple[float, float]:
    """Percentile bootstrap interval for :func:`hajek_mean`.

    A bootstrap rather than a `t` interval, for two reasons that both come from
    the shape of the data. The residual of a pair of **binary** scores takes only
    the values -1, 0 and 1, with most of its mass at 0 -- nothing like a normal at
    the sample sizes an audit produces. And the estimator is a *ratio*, which a
    closed-form interval would have to linearise before it could say anything.

    Returns ``(nan, nan)`` below two units rather than raising: an audit that has
    collected one label is a normal early state, not an error, and the caller's
    report should be able to say "not yet" without a special case at every site.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means = sorted(_bootstrap_means(values, probs, draws, seed, clusters))
    lo = means[int((alpha / 2) * draws)]
    hi = means[min(draws - 1, int((1 - alpha / 2) * draws))]
    return (lo, hi)


def standard_error(values: Sequence[float], probs: Sequence[float], *,
                   draws: int = 2000, seed: int = 1,
                   clusters: Optional[Sequence] = None) -> float:
    """Bootstrap standard error of :func:`hajek_mean`.

    Reported alongside the interval because the acceptance gate compares against
    a *variance*, not against an interval: the audit-limited test in the plan's
    Phase 4 is ``se(Delta)**2 > var_p``.
    """
    if len(values) < 2:
        return float("nan")
    return statistics.pstdev(_bootstrap_means(values, probs, draws, seed, clusters))


def residual_bias(records: Iterable, *, draws: int = 5000, alpha: float = 0.05,
                  seed: int = 0) -> Dict[str, object]:
    """Estimate ``Delta = E[f - Y]`` from resolved :class:`AuditRecord`s.

    Unresolved records are skipped rather than counted as zero -- "the oracle has
    not answered yet" and "the oracle agreed" are the two readings a missing
    value could have, and only one of them is a measurement.

    Returns ``n``, ``n_tasks``, ``delta``, ``f_mean``, ``y_mean``, and:

    ``ci`` / ``se``
        The unit bootstrap -- what a reader expects, and too narrow whenever the
        same task appears under several artifact versions.
    ``ci_clustered`` / ``se_clustered``
        Resampling **tasks**, which is the honest interval for a run. Reported
        beside the naive one rather than instead of it: the gap between them is
        itself the finding that the units were not independent draws.
    ``disagree``
        The share of audited units where the two scorers differed at all. Read it
        beside ``delta`` -- a small bias on every unit and a large bias on a few
        give the same mean and call for different fixes.
    """
    rows = [r for r in records if getattr(r, "oracle_score", None) is not None]
    if not rows:
        nan = float("nan")
        return {"n": 0, "n_tasks": 0, "delta": nan, "ci": (nan, nan),
                "ci_clustered": (nan, nan), "se": nan, "se_clustered": nan,
                "f_mean": nan, "y_mean": nan, "disagree": nan}
    res = [r.residual for r in rows]
    probs = [r.inclusion_prob for r in rows]
    tasks = [r.task_id for r in rows]
    return {
        "n": len(rows),
        "n_tasks": len(set(tasks)),
        "delta": hajek_mean(res, probs),
        "ci": bootstrap_ci(res, probs, draws=draws, alpha=alpha, seed=seed),
        "se": standard_error(res, probs, seed=seed),
        # The honest one when a run scores the same task under several artifact
        # versions, which every run does. Reported beside the naive interval
        # rather than instead of it: the difference between them is itself the
        # finding that the units were not independent.
        "ci_clustered": bootstrap_ci(res, probs, draws=draws, alpha=alpha,
                                     seed=seed, clusters=tasks),
        "se_clustered": standard_error(res, probs, seed=seed, clusters=tasks),
        "f_mean": statistics.fmean(r.verifier_score for r in rows),
        "y_mean": statistics.fmean(r.oracle_score for r in rows),
        "disagree": sum(1 for r in rows if r.residual != 0.0) / len(rows),
    }
