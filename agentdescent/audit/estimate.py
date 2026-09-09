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
                     draws: int, seed: int) -> List[float]:
    n = len(values)
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(hajek_mean([values[i] for i in idx], [probs[i] for i in idx]))
    return out


def bootstrap_ci(values: Sequence[float], probs: Sequence[float], *,
                 draws: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
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
    means = sorted(_bootstrap_means(values, probs, draws, seed))
    lo = means[int((alpha / 2) * draws)]
    hi = means[min(draws - 1, int((1 - alpha / 2) * draws))]
    return (lo, hi)


def standard_error(values: Sequence[float], probs: Sequence[float], *,
                   draws: int = 2000, seed: int = 1) -> float:
    """Bootstrap standard error of :func:`hajek_mean`.

    Reported alongside the interval because the acceptance gate compares against
    a *variance*, not against an interval: the audit-limited test in the plan's
    Phase 4 is ``se(Delta)**2 > var_p``.
    """
    if len(values) < 2:
        return float("nan")
    return statistics.pstdev(_bootstrap_means(values, probs, draws, seed))


def residual_bias(records: Iterable, *, draws: int = 5000, alpha: float = 0.05,
                  seed: int = 0) -> Dict[str, object]:
    """Estimate ``Delta = E[f - Y]`` from resolved :class:`AuditRecord`s.

    Unresolved records are skipped rather than counted as zero -- "the oracle has
    not answered yet" and "the oracle agreed" are the two readings a missing
    value could have, and only one of them is a measurement.

    Returns a dict with ``n``, ``delta``, ``ci``, ``se``, ``f_mean``, ``y_mean``
    and ``disagree`` (the share of audited units where the two scorers differed
    at all, which separates "a small bias everywhere" from "a large bias
    occasionally" -- they call for different fixes).
    """
    rows = [r for r in records if getattr(r, "oracle_score", None) is not None]
    if not rows:
        return {"n": 0, "delta": float("nan"), "ci": (float("nan"), float("nan")),
                "se": float("nan"), "f_mean": float("nan"),
                "y_mean": float("nan"), "disagree": float("nan")}
    res = [r.residual for r in rows]
    probs = [r.inclusion_prob for r in rows]
    return {
        "n": len(rows),
        "delta": hajek_mean(res, probs),
        "ci": bootstrap_ci(res, probs, draws=draws, alpha=alpha, seed=seed),
        "se": standard_error(res, probs, seed=seed),
        "f_mean": statistics.fmean(r.verifier_score for r in rows),
        "y_mean": statistics.fmean(r.oracle_score for r in rows),
        "disagree": sum(1 for r in rows if r.residual != 0.0) / len(rows),
    }
