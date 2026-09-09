"""Prediction-powered inference for a stratified mean.

The problem this solves. An audit produces a few hundred units where both the
cheap verifier `f` and the expensive oracle `Y` scored the same output, and tens
of thousands where only `f` did. The labelled units alone give an unbiased
estimate of `E[Y]` with an interval set by *their* count. PPI keeps the lack of
bias and borrows the unlabelled `f` values to narrow the interval -- so the
budget that buys 200 labels can answer a question that would otherwise need 600.

The estimator, per stratum, is the power-tuned form (PPI++):

    theta_hat = lam * mean(f_unlab)  +  mean(y_lab - lam * f_lab)

At ``lam = 0`` it degenerates to the labelled-only mean, which is the guarantee
that matters: **a useless verifier costs nothing**. At ``lam = 1`` it is the
classical PPI rectifier. In between, ``lam`` is chosen to minimise the variance

    Var = lam**2 * Var(f_unlab) / N  +  Var(y - lam*f) / n

and the first of those two terms is the one that is easy to leave out. Dropping
it does not break any obvious invariant; it just quietly reports intervals about
4pp under-covered (mutation C in the test suite).

Written against five things that went wrong while building it -- the docstrings
below name each where it bites:

1. Estimating ``lam`` on the same labelled units it is then applied to
   under-reports the standard error. Fixed by K-fold cross-fitting.
2. A `z` quantile is too narrow at audit sample sizes. Fixed with `t`.
3. Coverage is about 0.92, not 0.95, when the dominant stratum has fewer than
   :data:`MIN_N_DOMINANT` labels. Not a bug -- the skew of a binary outcome --
   but it must be *said*, so it is a warning and a locked test.
4. Symmetric noise is unbiased on balanced binary outcomes, so a test built from
   symmetric flips validates nothing about bias. Every test here uses a
   directional perturbation.
5. Replacing ``Var(y - lam*f)`` with ``Var(y)`` barely moves coverage, because
   ``lam`` adapts around the error. Coverage tests do not catch it; the mutation
   tests are chosen to be errors that *must* change the conclusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MIN_N_DOMINANT",
    "PPIError",
    "PPIResult",
    "Stratum",
    "ppi_mean_stratified",
    "t_ppf",
]

#: Labels the heaviest stratum needs before the reported interval means what it
#: says. Below it, measured coverage is about 0.92 against a nominal 0.95 --
#: see pitfall 3 in the module docstring, and
#: ``test_known_limitation_small_dominant_stratum``.
MIN_N_DOMINANT = 80

#: Folds used to cross-fit ``lam``. Five is the usual compromise: enough that
#: each fold's coefficient is estimated on most of the data, few enough that the
#: per-fold coefficients do not become noise themselves.
DEFAULT_K_FOLDS = 5


class PPIError(ValueError):
    """The input cannot support an estimate at all.

    Distinct from a *warning*, which says the estimate is weaker than it looks.
    The caller -- :class:`~agentdescent.audit.calibrator.Calibrator` -- turns
    this into a stale rectifier rather than propagating it, because "not enough
    labels yet" is an ordinary early state of an audit and must degrade the
    acceptance gate to conservative rather than stop the run.
    """


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t, via the Cornish-Fisher expansion in ``1/df``.

    A `t` and not a `z`, which is pitfall 2. Strata in a real audit carry 40-100
    labels; the normal quantile is 2-3% too small there, and an interval that is
    2% too narrow is an interval that covers 93% of the time while claiming 95%.

    Accuracy, measured against the quantile obtained by bisecting the exact CDF
    (regularized incomplete beta): **worst absolute error 1.4e-5 for df >= 9**,
    falling to 3e-8 by df = 30. Below df = 9 the expansion degrades quickly --
    3.8e-3 at df = 3 -- which does not matter here because
    :data:`MIN_N_DOMINANT` keeps the dominant stratum far above that and
    :func:`ppi_mean_stratified` refuses fewer than two labels outright.

    The four correction terms are the standard expansion; truncating to two
    costs a factor of ~30 in accuracy at df = 10, which is cheap to avoid.
    """
    if df <= 0:
        raise PPIError(f"degrees of freedom must be positive, got {df!r}")
    z = NormalDist().inv_cdf(p)
    z2 = z * z
    z3, z5, z7, z9 = z2 * z, z2 * z2 * z, z2 * z2 * z2 * z, z2 * z2 * z2 * z2 * z
    return (z
            + (z3 + z) / (4.0 * df)
            + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df ** 2)
            + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * df ** 3)
            + (79.0 * z9 + 776.0 * z7 + 1482.0 * z5 - 1920.0 * z3 - 945.0 * z)
            / (92160.0 * df ** 4))


@dataclass(frozen=True)
class Stratum:
    """One layer of the sampling design, with its labelled and unlabelled halves.

    ``weight`` is the layer's share of the **population**, recorded when the
    sample was drawn -- not its share of the sample. They differ by exactly the
    amount stratification was introduced to create, and using the sample share
    is mutation A: it turns a stratified sample back into a simple one and
    coverage falls to 0.18.

    ``f_unlab`` is the verifier's score on units in this layer that were *not*
    sent to the oracle. Not the labelled ones again -- reusing them would make
    the two terms of the variance dependent, and the interval would be a
    fiction.

    **The estimator never keeps those scores.** Everything the mathematics does
    with the unlabelled half is ``size``, ``mean`` and ``var`` -- three
    sufficient statistics -- so ``__post_init__`` reduces the array to
    :attr:`n_unlab`, :attr:`mean_unlab` and :attr:`var_unlab` and the array is
    not read again.

    That is not an optimisation, it is what lets the audit store exist. A run
    scores tens of thousands of units and sends a few hundred to the oracle;
    keeping every unlabelled score to satisfy this constructor would mean
    persisting the whole run to compute a correction that needs nine numbers.
    :meth:`from_moments` is the constructor for a caller that kept the running
    summary instead, and it is the one
    :class:`~agentdescent.audit.calibrator.Calibrator` uses.
    """

    name: str
    weight: float
    f_lab: np.ndarray
    y_lab: np.ndarray
    #: The raw unlabelled scores, when the caller happens to have them. Reduced
    #: to the three fields below and then unused; ``None`` when they were never
    #: kept.
    f_unlab: Optional[np.ndarray] = None
    n_unlab: int = 0
    mean_unlab: float = 0.0
    #: Sample variance, ``ddof=1``. Zero is the right value for fewer than two
    #: unlabelled units, and the estimator drops the term rather than trusting it.
    var_unlab: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "f_lab", np.asarray(self.f_lab, dtype=float))
        object.__setattr__(self, "y_lab", np.asarray(self.y_lab, dtype=float))
        if self.f_lab.shape != self.y_lab.shape:
            raise PPIError(
                f"stratum {self.name!r}: {self.f_lab.size} verifier scores but "
                f"{self.y_lab.size} oracle scores -- they must be paired")
        if self.weight < 0.0:
            raise PPIError(f"stratum {self.name!r}: weight {self.weight!r} < 0")
        if self.f_unlab is not None:
            arr = np.asarray(self.f_unlab, dtype=float)
            object.__setattr__(self, "f_unlab", arr)
            object.__setattr__(self, "n_unlab", int(arr.size))
            object.__setattr__(self, "mean_unlab",
                               float(np.mean(arr)) if arr.size else 0.0)
            object.__setattr__(self, "var_unlab",
                               float(np.var(arr, ddof=1)) if arr.size >= 2 else 0.0)
        elif self.n_unlab < 0:
            raise PPIError(f"stratum {self.name!r}: n_unlab {self.n_unlab!r} < 0")
        if self.var_unlab < 0.0:
            raise PPIError(
                f"stratum {self.name!r}: var_unlab {self.var_unlab!r} < 0")

    @classmethod
    def from_moments(cls, name: str, weight: float, f_lab, y_lab, *,
                     n_unlab: int, mean_unlab: float,
                     var_unlab: float) -> "Stratum":
        """Build from a running summary of the unlabelled half rather than its scores.

        Keyword-only on purpose: three bare floats in a row are exactly the kind
        of argument list that gets transposed, and a swapped mean and variance
        produces a plausible number rather than an error.
        """
        return cls(name=name, weight=weight, f_lab=f_lab, y_lab=y_lab,
                   f_unlab=None, n_unlab=int(n_unlab),
                   mean_unlab=float(mean_unlab), var_unlab=float(var_unlab))

    @property
    def n(self) -> int:
        return int(self.f_lab.size)


@dataclass(frozen=True)
class PPIResult:
    """The estimate, its interval, and everything needed to distrust it.

    ``warnings`` is part of the result rather than a logging side effect on
    purpose: a caller that stores the number must be able to store why it was
    weak, because by the time anyone asks, the log is gone.
    """

    #: Estimate of the population mean of ``Y``.
    theta: float
    ci: Tuple[float, float]
    se: float
    #: Welch-Satterthwaite degrees of freedom over every variance component.
    df: float
    #: Labelled-count-weighted mean of the cross-fitted coefficients. Near 0
    #: means the verifier carried no usable signal and the estimate fell back to
    #: the labelled units alone.
    lambda_: float
    #: ``Var(labelled-only) / Var(this)``. The factor by which the labelled
    #: sample was effectively multiplied. **1.0 means the verifier bought
    #: nothing** -- and if it stays there, the answer is a better verifier, not
    #: a bigger audit.
    gain_factor: float
    n: int
    n_unlab: int
    alpha: float
    warnings: List[str] = field(default_factory=list)
    #: Per stratum: ``weight``, ``n``, ``n_unlab``, ``theta``, ``var``,
    #: ``lambda``, and ``resid_sd`` -- the sd of ``f - Y``, which is what
    #: Neyman allocation needs and what a sampler must **not** confuse with the
    #: sd of ``Y``.
    per_stratum: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def halfwidth(self) -> float:
        return 0.5 * (self.ci[1] - self.ci[0])


def _folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    """Indices split into ``k`` roughly equal folds, deterministically shuffled."""
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [f for f in np.array_split(idx, k) if f.size]


def _lambda_from(f: np.ndarray, y: np.ndarray, n: int, n_unlab: int) -> float:
    """The variance-minimising coefficient, from one set of labelled pairs.

    Minimising ``lam**2 Var(f_unlab)/N + Var(y - lam f)/n`` over ``lam`` gives

        lam* = Cov(f, y) / (Var(f) * (1 + n/N))

    Clipped to ``[0, 1]``. Outside that range the estimator is still unbiased,
    but a negative coefficient means "the verifier is anti-correlated with the
    truth", which on an audit is a sign the labels are mismatched rather than a
    quantity to exploit -- and above 1 the unlabelled term grows faster than the
    rectifier term shrinks.

    ``N == 0`` returns 0: with nothing unlabelled there is nothing to borrow,
    and the estimator must degenerate to the labelled-only mean rather than
    divide by zero.
    """
    if n_unlab == 0 or f.size < 2:
        return 0.0
    var_f = float(np.var(f, ddof=1))
    if var_f <= 0.0:
        return 0.0
    cov = float(np.cov(f, y, ddof=1)[0, 1])
    lam = cov / (var_f * (1.0 + n / n_unlab))
    return float(min(1.0, max(0.0, lam)))


def _lambda_crossfit(f: np.ndarray, y: np.ndarray, n_unlab: int, *,
                     k_folds: int, seed: int) -> np.ndarray:
    """A per-unit coefficient, each fitted **without** the unit it is applied to.

    This is pitfall 1, and it is the reason this function exists rather than one
    scalar. Fitting ``lam`` on all the labels and then applying it to those same
    labels makes the residuals ``y - lam*f`` look smaller than they are, because
    ``lam`` was chosen to make them small. Measured at n between 40 and 80, the
    reported standard error came in about 7% under the true sampling sd and
    coverage fell from 0.95 to 0.91.

    **Run the coverage test before changing anything here.** The failure is
    invisible in a point estimate -- it moves only the width of the interval.
    """
    n = int(f.size)
    if n < 2:
        return np.zeros(n)
    k = max(2, min(k_folds, n))
    out = np.zeros(n)
    for fold in _folds(n, k, seed):
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        if mask.sum() < 2:                      # too little left to fit on
            out[fold] = 0.0
            continue
        out[fold] = _lambda_from(f[mask], y[mask], n, n_unlab)
    return out


def ppi_mean_stratified(strata: Sequence[Stratum], *, alpha: float = 0.05,
                        k_folds: int = DEFAULT_K_FOLDS,
                        seed: int = 0) -> PPIResult:
    """Estimate ``E[Y]`` over a stratified population, using the unlabelled ``f``.

    Parameters
    ----------
    strata:
        One :class:`Stratum` per layer. ``weight`` must be the **population**
        share and the weights must sum to 1.
    alpha:
        ``1 - alpha`` is the nominal coverage. 0.05 gives a 95% interval.
    k_folds:
        Folds for cross-fitting ``lam``; see :func:`_lambda_crossfit`.
    seed:
        Fixes the fold split, so the same labels give the same interval twice.

    Raises
    ------
    PPIError
        When no estimate is possible: no strata, a layer with fewer than two
        labels, or weights that do not sum to 1. The last is strict on purpose
        -- weights that are off are a bug in the sampler that recorded them, and
        normalising them silently would produce a plausible answer to the wrong
        question.
    """
    if not strata:
        raise PPIError("no strata")
    total_w = sum(s.weight for s in strata)
    if abs(total_w - 1.0) > 1e-6:
        raise PPIError(
            f"stratum weights sum to {total_w!r}, not 1. They are population "
            "shares recorded at sampling time; a mismatch means the sampler and "
            "the estimator disagree about what population this describes.")
    for s in strata:
        if s.n < 2:
            raise PPIError(
                f"stratum {s.name!r} has {s.n} labelled unit(s); at least 2 are "
                "needed for a variance. Wait for more oracle results.")

    warnings: List[str] = []
    theta = 0.0
    var_total = 0.0
    var_classical = 0.0
    components: List[Tuple[float, float]] = []      # (variance, df)
    lam_num = 0.0
    n_lab = n_unlab = 0
    per: Dict[str, Dict[str, float]] = {}

    for s in strata:
        lam_i = _lambda_crossfit(s.f_lab, s.y_lab, s.n_unlab,
                                 k_folds=k_folds, seed=seed)
        lam_bar = float(np.mean(lam_i))

        # The rectifier, per unit, each with its own out-of-fold coefficient.
        resid = s.y_lab - lam_i * s.f_lab
        theta_h = lam_bar * s.mean_unlab + float(np.mean(resid))

        var_resid = float(np.var(resid, ddof=1))
        var_lab = var_resid / s.n
        if s.n_unlab >= 2:
            var_unlab = (lam_bar ** 2) * s.var_unlab / s.n_unlab
        else:
            # One unlabelled unit carries no usable variance estimate, and zero
            # is the honest value only because `lam` is then ~0 anyway.
            var_unlab = 0.0
        var_h = var_lab + var_unlab

        w2 = s.weight ** 2
        theta += s.weight * theta_h
        var_total += w2 * var_h
        var_classical += w2 * float(np.var(s.y_lab, ddof=1)) / s.n
        if var_lab > 0:
            components.append((w2 * var_lab, s.n - 1))
        if var_unlab > 0 and s.n_unlab >= 2:
            components.append((w2 * var_unlab, s.n_unlab - 1))

        lam_num += lam_bar * s.n
        n_lab += s.n
        n_unlab += s.n_unlab
        per[s.name] = {
            "weight": s.weight, "n": s.n, "n_unlab": s.n_unlab,
            "theta": theta_h, "var": var_h, "lambda": lam_bar,
            # sd of the RESIDUAL f - Y, not of Y. Neyman allocation wants this
            # one, and the difference is silent: using sd(Y) sends the budget to
            # whichever layer has the most variable outcome rather than the
            # layer where the verifier is least trustworthy.
            "resid_sd": float(np.std(s.f_lab - s.y_lab, ddof=1)),
        }

    # Welch-Satterthwaite over every component that contributed variance.
    if var_total <= 0.0 or not components:
        raise PPIError(
            "every stratum has zero variance; there is nothing to put an "
            "interval around")
    df = var_total ** 2 / sum(v ** 2 / d for v, d in components if d > 0)
    se = math.sqrt(var_total)
    crit = t_ppf(1.0 - alpha / 2.0, df)
    ci = (theta - crit * se, theta + crit * se)

    dominant = max(strata, key=lambda s: s.weight)
    if dominant.n < MIN_N_DOMINANT:
        warnings.append(
            f"dominant stratum {dominant.name!r} has {dominant.n} labels "
            f"(< MIN_N_DOMINANT={MIN_N_DOMINANT}); measured coverage there is "
            f"about 0.92 against a nominal {1 - alpha:.2f}, so this interval is "
            "optimistic by roughly 3 points")
    gain = (var_classical / var_total) if var_total > 0 else float("nan")
    if gain < 1.02:
        warnings.append(
            f"gain_factor {gain:.3f}: the verifier is buying almost nothing. "
            "More oracle labels will help; more in-loop evaluation will not.")

    return PPIResult(
        theta=theta, ci=ci, se=se, df=df,
        lambda_=(lam_num / n_lab if n_lab else 0.0),
        gain_factor=gain, n=n_lab, n_unlab=n_unlab, alpha=alpha,
        warnings=warnings, per_stratum=per)
