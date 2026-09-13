"""How much of the oracle budget each layer gets, and why.

A flat sampling rate spends the budget where the units are. What decides the
width of the correction is where the verifier is *unreliable*, and those are not
the same place: a layer the verifier gets right every time contributes nothing
to the interval no matter how many of its units you label.

Neyman allocation says so exactly. To minimise the variance of a stratified mean
under a fixed total sample, take

    n_h  proportional to  W_h * sd_h

-- the layer's share of the population times the standard deviation *within* it.
Here the relevant deviation is of the **residual** ``f - Y``, which is what
:attr:`PPIResult.per_stratum` reports as ``resid_sd``. Reading ``sd(Y)`` instead
is the silent version of this mistake: it sends the budget to whichever layer has
the most variable outcome rather than the one where the verifier is least
trustworthy, and nothing errors.

## Rates, not a chosen set of units

The plan this implements had the sampler take a generation's units, stratify
them, and hand back which ones to send. That shape does not fit the tap: it sees
one ``(task, output)`` at a time as the loop scores it, and decides on the spot
with a draw seeded from the unit itself -- which is what makes inclusion
independent of thread scheduling and reproducible from a seed.

Allocation survives the translation intact. ``n_h`` units out of an expected
``W_h * N`` is an inclusion probability of ``n_h / (W_h * N)``, and a per-stratum
rate is exactly what :class:`~agentdescent.audit.tap.AuditedReward` already
takes. So this module plans **rates**, the tap keeps deciding per unit, and the
allocation is the same one.

The cost of the translation is honest and worth naming: rates are set from an
*expected* population, so the realised counts land near the plan rather than on
it. At the sizes an audit uses that is a few units either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .ppi import MIN_N_DOMINANT

__all__ = [
    "AuditPolicy",
    "SamplePlan",
    "boundary_stratifier",
    "observed_weights",
    "resid_sd_from",
]


def boundary_stratifier(threshold: float, width: float = 0.05
                        ) -> Callable[[Any, str, float], str]:
    """Split units into ``accepted`` / ``boundary`` / ``rejected`` around a threshold.

    The three layers are not arbitrary. A verifier's disagreement with the truth
    is concentrated where it is closest to indifferent -- an answer it scores 0.5
    is one it could not decide about, and those are the units where a label buys
    the most. Layers well clear of the threshold are cheap to be confident about
    and get a correspondingly small share.

    ``width`` is the half-width of the boundary band, in score units. The edge is
    **inclusive**, and the tolerance below is what makes that true rather than
    almost true: ``abs(0.55 - 0.5)`` is ``0.050000000000000044`` in binary, so a
    bare ``<= width`` puts a unit exactly on the edge outside the band -- or
    inside, depending on which numbers happen to be involved. Either convention
    is defensible; one that varies with the floating-point representation of the
    particular scores is not.
    """
    edge = width + 1e-9

    def stratify(task: Any, output: str, score: float) -> str:
        if abs(score - threshold) <= edge:
            return "boundary"
        return "accepted" if score > threshold else "rejected"

    return stratify


@dataclass(frozen=True)
class AuditPolicy:
    """What the audit is trying to achieve, and what it refuses to do to get there.

    Parameters
    ----------
    enabled:
        Off by default. The whole layer is opt-in, and a policy that is not
        enabled plans a rate of zero everywhere rather than a small one -- "we
        are not auditing" and "we are auditing a little" produce different
        records and only one of them is honest.
    target_halfwidth:
        How narrow the correction's 95% interval should be. Drives the total
        label budget through the Neyman-optimal sample size; see :func:`plan`.
    boundary_width:
        Half-width of the band around the acceptance threshold that counts as
        ``boundary``.
    calibration_fraction:
        Share of audited units that go to the calibration pool rather than the
        improvement pool. Passed through to the tap.
    min_per_stratum:
        No layer gets fewer than this many labels, whatever Neyman says. A layer
        allocated two labels contributes a variance estimate from two points,
        which is worse than not stratifying at all.
    min_dominant:
        The heaviest layer gets at least this many. Defaults to
        :data:`~agentdescent.audit.ppi.MIN_N_DOMINANT`, below which the reported
        coverage is about 0.92 rather than 0.95 -- so this floor and that warning
        are the same number for the same reason, and moving one without the other
        is how a floor stops meaning anything.
    max_labels:
        A hard cap. Oracle labels cost money or a person's afternoon, and a
        target half-width small enough to be unreachable should produce a warning
        and a bounded plan rather than an unbounded bill.
    """

    enabled: bool = False
    target_halfwidth: float = 0.05
    boundary_width: float = 0.05
    calibration_fraction: float = 0.7
    min_per_stratum: int = 20
    min_dominant: int = MIN_N_DOMINANT
    max_labels: int = 400
    #: Nominal coverage the half-width is quoted at.
    alpha: float = 0.05


@dataclass(frozen=True)
class SamplePlan:
    """Per-stratum inclusion probabilities, and the reasoning that produced them.

    :attr:`rates` goes straight into
    ``AuditedReward(rates=plan.rates, sample_rate=plan.default_rate)``.
    """

    #: stratum -> inclusion probability in ``[0, 1]``.
    rates: Dict[str, float]
    #: stratum -> labels the allocation asked for.
    target_n: Dict[str, int]
    #: stratum -> population share used to plan.
    weights: Dict[str, float]
    #: stratum -> residual sd used to plan.
    resid_sd: Dict[str, float]
    #: Rate for a stratum the plan never saw. Zero: an unplanned layer is one the
    #: population estimate did not know about, and quietly sampling it at some
    #: default would put units into the estimate under a probability nobody
    #: recorded.
    default_rate: float = 0.0
    total_n: int = 0
    expected_units: int = 0
    warnings: List[str] = field(default_factory=list)

    def rate_for(self, stratum: str) -> float:
        return self.rates.get(stratum, self.default_rate)


def observed_weights(store: Any, verifier_version: str) -> Dict[str, float]:
    """Population shares from what a previous run actually saw.

    Every unit reaches the tap, audited or not, so the counts are the population
    rather than an estimate of it -- the labelled records plus the unlabelled
    moments account for every unit the run scored.
    """
    counts: Dict[str, float] = {}
    for rec in store.all():
        if rec.verifier_version == verifier_version:
            counts[rec.stratum] = counts.get(rec.stratum, 0.0) + 1.0
    for stratum, moments in store.unlabelled_moments(verifier_version).items():
        counts[stratum] = counts.get(stratum, 0.0) + float(moments["n"])
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def resid_sd_from(previous: Any) -> Dict[str, float]:
    """``stratum -> resid_sd`` out of a :class:`PPIResult`, or ``{}``.

    Reads ``resid_sd`` and not the sd of the outcome. They are different
    quantities that are both plausible here, and the wrong one produces a plan
    that is merely suboptimal rather than an error -- so it would survive review.
    """
    if previous is None:
        return {}
    per = getattr(previous, "per_stratum", None) or {}
    return {name: float(cell["resid_sd"])
            for name, cell in per.items() if "resid_sd" in cell}


def plan(policy: AuditPolicy, *, weights: Dict[str, float],
         expected_units: int,
         resid_sd: Optional[Dict[str, float]] = None) -> SamplePlan:
    """Neyman allocation, converted to per-stratum inclusion probabilities.

    Parameters
    ----------
    weights:
        Population share per stratum. Need not sum to exactly 1; it is
        normalised, because these usually come from counting a previous run and
        arriving at 0.9999 should not be an error.
    expected_units:
        How many units the next run is expected to score. Rates are
        ``n_h / (W_h * expected_units)``, so an estimate that is too low
        oversamples and one that is too high undersamples -- both bounded, and
        the realised inclusion probability is recorded per unit either way, so a
        wrong guess costs precision and never correctness.
    resid_sd:
        Per-stratum sd of ``f - Y`` from the last calibration. Missing entries
        fall back to an equal-residual assumption, which is what proportional
        allocation already assumes -- so the first run, with no history, plans
        proportionally and is right to.

    Notes
    -----
    The total is the Neyman-optimal sample size for the requested half-width:
    under optimal allocation ``se = sum(W_h sd_h) / sqrt(n)``, so

        n = (sum(W_h sd_h) * z / halfwidth) ** 2

    That is what makes ``target_halfwidth`` a specification rather than a wish.
    With no history the sds are equal and cancel, leaving a plan driven entirely
    by the floors -- which is the honest answer to "how many labels do I need"
    before anything has been measured.
    """
    warnings: List[str] = []
    if not policy.enabled:
        return SamplePlan(rates={}, target_n={}, weights={}, resid_sd={},
                          default_rate=0.0, expected_units=expected_units,
                          warnings=["audit disabled; nothing will be sampled"])
    if not weights:
        return SamplePlan(rates={}, target_n={}, weights={}, resid_sd={},
                          default_rate=0.0, expected_units=expected_units,
                          warnings=["no strata to plan for"])

    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("stratum weights sum to zero")
    w = {k: v / total_w for k, v in weights.items()}

    sd = dict(resid_sd or {})
    known = [sd[k] for k in w
             if k in sd and math.isfinite(sd[k]) and sd[k] > 0]
    missing = [k for k in w if k not in sd or not math.isfinite(sd[k]) or sd[k] <= 0]
    if missing and len(missing) == len(w):
        # Nothing measured anywhere. Equal residuals is exactly the assumption
        # proportional allocation already makes, and any positive constant gives
        # the same allocation -- so 1.0 is as good as a guess at the real scale
        # and does not pretend to be one.
        for k in missing:
            sd[k] = 1.0
        warnings.append(
            "no residual history: allocating proportionally, which assumes the "
            "verifier is equally unreliable everywhere")
    elif missing:
        # Some measured, some not -- and here the constant above is a *bug*. It
        # is on a different scale from the measured values, so a stratum nobody
        # has measured outranks one measured at 0.4 purely because 1.0 > 0.4.
        #
        # Fill with the largest measured sd instead. The asymmetry decides it:
        # under-sampling a stratum nobody has measured keeps it unmeasured, which
        # is self-perpetuating, while over-sampling it costs some budget once and
        # corrects itself next round when there is a real number to use.
        stand_in = max(known)
        for k in missing:
            sd[k] = stand_in
        warnings.append(
            f"no residual history for {sorted(missing)}; allocated at the "
            f"largest measured residual sd ({stand_in:.3f}) so an unmeasured "
            "stratum is over- rather than under-sampled until it is measured")

    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - policy.alpha / 2.0)
    weighted_sd = sum(w[k] * sd[k] for k in w)
    if weighted_sd <= 0 or policy.target_halfwidth <= 0:
        n_total = policy.max_labels
    else:
        n_total = math.ceil((weighted_sd * z / policy.target_halfwidth) ** 2)

    if n_total > policy.max_labels:
        warnings.append(
            f"a half-width of {policy.target_halfwidth} needs about {n_total} "
            f"labels; capped at max_labels={policy.max_labels}, so the interval "
            "will be wider than asked for")
        n_total = policy.max_labels

    share = {k: w[k] * sd[k] for k in w}
    share_total = sum(share.values())
    target = {k: int(round(n_total * share[k] / share_total)) for k in w}

    # -- floors, applied after allocation and never before -------------------
    #
    # Applying them first would make the allocation optimise around a constraint
    # rather than be corrected by one, and the difference shows whenever a floor
    # binds: the remaining budget should still be split by Neyman, not reduced
    # proportionally.
    dominant = max(w, key=lambda k: w[k])
    for k in target:
        floor = policy.min_dominant if k == dominant else policy.min_per_stratum
        if target[k] < floor:
            target[k] = floor
    raised = sum(target.values())
    if raised > policy.max_labels:
        warnings.append(
            f"the per-stratum floors ask for {raised} labels, above "
            f"max_labels={policy.max_labels}. The floors win: they are what "
            "makes each stratum's variance estimable at all, and a cap that "
            "overrides them buys a cheaper plan that cannot be read.")

    rates: Dict[str, float] = {}
    for k in w:
        expected_in_stratum = w[k] * expected_units
        if expected_in_stratum <= 0:
            rates[k] = 1.0
            continue
        rate = target[k] / expected_in_stratum
        if rate >= 1.0:
            warnings.append(
                f"stratum {k!r} needs {target[k]} labels from an expected "
                f"{expected_in_stratum:.0f} units; auditing all of them and "
                "still falling short")
            rate = 1.0
        rates[k] = rate

    return SamplePlan(rates=rates, target_n=target, weights=w, resid_sd=sd,
                      default_rate=0.0, total_n=sum(target.values()),
                      expected_units=expected_units, warnings=warnings)
