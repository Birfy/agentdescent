"""From a store of audited pairs to a correction the acceptance gate can apply.

This is the join between the two halves of the audit layer. The store holds a
few hundred units where both scorers ran and a running summary of the many where
only the verifier did; :mod:`~agentdescent.audit.ppi` turns those into an
estimate of the true mean quality. What the gate needs is the *difference*:

    Delta = E[f] - E[Y]

and the gate applies it as ``p_true = p_hat - delta_hat``. What this module
therefore has to produce is not one number but three: the correction, its
uncertainty, and -- the one the plan does not ask for -- :attr:`resid_sd`, the
*spread* of the verifier's error rather than its mean. ``delta_hat`` cancels out
of a comparison between two candidates scored by the same verifier;
``resid_sd`` does not, and on real data it is the larger term by 5.5x. See
:mod:`agentdescent.audit.gate` for the arithmetic and the measurement.

``E[f]`` is not estimated. Every unit the run scored was seen by the tap --
audited or not -- so the population mean of ``f`` is a **known quantity**,
computed from the labelled scores and the unlabelled moments together. Only
``E[Y]`` is estimated, which is why ``delta_se`` and ``se`` are the same number
here and would stop being so the day someone computes ``E[f]`` from a subsample.

Three things this module refuses to do, each because the alternative fails
quietly:

* **Mix verifier versions.** A correction estimated for one verifier says
  nothing about the next, and the arithmetic works fine either way.
* **Touch the improvement pool.** Labels used to *edit* the verifier were chosen
  to make those units agree with it, so a bias estimated on them reads as more
  honest than the truth. :meth:`AuditStore.for_calibration` asserts this; the
  calibrator asks for it by name rather than filtering itself.
* **Guess when it cannot estimate.** Too few labels, a stratum with one unit, a
  verifier that just changed -- all return a **stale** rectification rather than
  a number or an exception. Stale is a legitimate state with a defined meaning
  for the gate: widen the interval and commit less, rather than correcting by an
  amount nobody measured.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ppi import MIN_N_DOMINANT, PPIError, Stratum, ppi_mean_stratified
from .store import AuditStore

__all__ = ["Calibrator", "Rectification", "STALE_INFLATION",
           "population_resid_sd"]

#: Variance multiplier the acceptance gate applies while a rectification is
#: stale. Not a correction -- a stale rectifier has no number to correct *with*
#: -- but a widening, so a gate that cannot be told how biased its verifier is
#: commits less rather than the same amount with more confidence.
STALE_INFLATION = 2.0


@dataclass(frozen=True)
class Rectification:
    """The correction, its uncertainty, and whether it may be used at all.

    A stale rectification still carries whatever numbers were last computed, so
    a caller can log what it *would* have applied. It must not apply them --
    :attr:`is_stale` is the only field that decides that.
    """

    verifier_version: str
    #: ``E[f] - E[Y]``. Positive means the verifier scores higher than the truth,
    #: which is the direction that lets a loop accept changes that improved
    #: nothing. Subtract it from the measured rate.
    delta_hat: float
    delta_se: float
    #: The PPI estimate of ``E[Y]`` and its interval -- the true mean quality,
    #: as opposed to what the verifier believes it to be.
    theta: float
    theta_ci: Tuple[float, float]
    #: Standard error of :attr:`delta_hat`. Equal to :attr:`delta_se` by
    #: construction; both exist because the gate reads ``se`` and a reader of the
    #: record reads ``delta_se``, and losing the pairing to save a float would be
    #: a poor trade.
    se: float
    n: int
    n_unlab: int
    gain_factor: float
    is_stale: bool
    stale_reason: Optional[str]
    #: Population sd of the residual ``f - Y``, weighted across strata -- the
    #: *spread* of the verifier's error, as opposed to :attr:`delta_hat`, its
    #: mean. Two different facts, and the gate needs this one: a mean error
    #: cancels out of a comparison between two candidates and this does not.
    #: See :mod:`agentdescent.audit.gate`.
    resid_sd: float = float("nan")
    warnings: List[str] = field(default_factory=list)
    computed_at: float = 0.0
    #: Earliest and latest ``dispatched_at`` among the records used.
    #:
    #: The plan called this ``generations_covered``. The tap hooks the reward
    #: function, which has no notion of a generation -- it sees ``(task,
    #: output)`` and nothing about where in a run it came from -- so the honest
    #: coverage statement is a time range.
    covers: Tuple[float, float] = (0.0, 0.0)

    @classmethod
    def stale(cls, verifier_version: str, reason: str,
              previous: Optional["Rectification"] = None) -> "Rectification":
        """A rectification that must not be applied, and says why.

        Carries the previous numbers forward when there are any, so a log can
        show what was withheld and on what evidence.
        """
        if previous is not None:
            return cls(
                verifier_version=verifier_version,
                delta_hat=previous.delta_hat, delta_se=previous.delta_se,
                theta=previous.theta, theta_ci=previous.theta_ci,
                se=previous.se, n=previous.n, n_unlab=previous.n_unlab,
                gain_factor=previous.gain_factor,
                is_stale=True, stale_reason=reason, resid_sd=previous.resid_sd,
                warnings=list(previous.warnings), computed_at=previous.computed_at,
                covers=previous.covers)
        nan = float("nan")
        return cls(verifier_version=verifier_version, delta_hat=nan, delta_se=nan,
                   theta=nan, theta_ci=(nan, nan), se=nan, n=0, n_unlab=0,
                   gain_factor=nan, is_stale=True, stale_reason=reason,
                   computed_at=time.time())


def population_resid_sd(strata) -> float:
    """Sd of ``f - Y`` over the whole population, from the labelled pairs.

    Not the average of the strata's own residual sds. A verifier that is
    uniformly +0.4 generous in one stratum and uniformly right in another has
    zero spread *inside* each and plenty across them, and the within-stratum
    average would report zero -- which is the number that would tell the gate a
    proxy is a measurement.

    So both terms::

        Var = sum_h W_h * (sd_h ** 2 + (mean_h - Delta) ** 2)

    The weights are population shares, so this describes the units the run
    scored rather than the units it happened to audit. Strata are equally
    sampled internally by construction (the tap holds one rate per stratum), so
    a plain mean within a stratum needs no weighting of its own.
    """
    parts = [(s.weight, np.asarray(s.f_lab, dtype=float)
              - np.asarray(s.y_lab, dtype=float)) for s in strata]
    parts = [(w, r) for w, r in parts if r.size]
    if not parts:
        return float("nan")
    total_w = sum(w for w, _ in parts)
    if total_w <= 0.0:
        return float("nan")
    grand = sum(w * float(r.mean()) for w, r in parts) / total_w
    var = sum(w * (float(r.var(ddof=1)) if r.size >= 2 else 0.0)
              + w * (float(r.mean()) - grand) ** 2 for w, r in parts) / total_w
    return math.sqrt(max(0.0, var))


class Calibrator:
    """Keeps one rectification per verifier version, and knows when to distrust it.

    Parameters
    ----------
    store:
        Where the audited pairs and the unlabelled moments live.
    min_labels:
        Below this many resolved calibration labels the result is stale rather
        than wide. A very wide interval and "we do not know yet" are different
        claims, and only the second one stops a caller reading a number off it.
    min_per_stratum:
        A stratum with fewer than this many labels is **merged into the largest
        one** rather than dropped. Dropping it would silently change the
        population the estimate describes; merging keeps every unit represented
        and costs only resolution.
    """

    def __init__(self, store: AuditStore, *, alpha: float = 0.05,
                 seed: int = 0, min_labels: int = 30,
                 min_per_stratum: int = 5) -> None:
        self.store = store
        self.alpha = alpha
        self.seed = seed
        self.min_labels = min_labels
        self.min_per_stratum = min_per_stratum
        self._cache: Dict[str, Rectification] = {}
        self._stale_reason: Optional[str] = None
        self._lock = threading.RLock()

    # -- the gate's entry point ----------------------------------------------

    def current(self, verifier_version: str) -> Rectification:
        """The rectification to apply now, computing it if it is not cached.

        Never raises. Every failure mode -- no labels, one label in a stratum, a
        verifier that changed under us -- comes back as a stale rectification,
        because the caller is a merge decision and a merge decision has to be
        made.
        """
        with self._lock:
            if self._stale_reason is not None:
                return Rectification.stale(verifier_version, self._stale_reason,
                                           self._cache.get(verifier_version))
            cached = self._cache.get(verifier_version)
            if cached is not None:
                return cached
        return self.recompute(verifier_version)

    def recompute(self, verifier_version: str) -> Rectification:
        """Re-read the store and re-estimate. Clears any manual stale mark."""
        try:
            result = self._estimate(verifier_version)
        except PPIError as exc:
            result = Rectification.stale(verifier_version, str(exc))
        with self._lock:
            self._stale_reason = None
            self._cache[verifier_version] = result
            return result

    def mark_stale(self, reason: str) -> None:
        """Withhold every rectification until the next :meth:`recompute`.

        Called when the verifier changes. Deliberately global rather than
        per-version: the point of the call is that the caller has just learned
        the instrument moved, and at that moment it does not yet know which
        version identifier the new one will have.
        """
        with self._lock:
            self._stale_reason = reason

    @property
    def stale_reason(self) -> Optional[str]:
        with self._lock:
            return self._stale_reason

    # -- the estimate --------------------------------------------------------

    def _estimate(self, verifier_version: str) -> Rectification:
        records = self.store.for_calibration(verifier_version)
        if len(records) < self.min_labels:
            return Rectification.stale(
                verifier_version,
                f"{len(records)} calibration labels for {verifier_version!r}; "
                f"{self.min_labels} needed before an estimate means anything")

        moments = self.store.unlabelled_moments(verifier_version)
        grouped: Dict[str, List] = defaultdict(list)
        for rec in records:
            grouped[rec.stratum].append(rec)
        by_stratum, moments = self._merge_thin(grouped, moments)

        # The population share of each stratum is a *count*, not an estimate:
        # the tap saw every unit the run scored and put each in exactly one
        # stratum, so the sample frame here is the population.
        sizes = {name: len(rows) + int(moments.get(name, {}).get("n", 0))
                 for name, rows in by_stratum.items()}
        total = sum(sizes.values())
        if total <= 0:
            return Rectification.stale(verifier_version, "no units observed")

        strata, f_pop = [], 0.0
        for name, rows in by_stratum.items():
            m = moments.get(name, {"n": 0, "mean": 0.0, "var": 0.0})
            weight = sizes[name] / total
            f_lab = np.array([r.verifier_score for r in rows], dtype=float)
            y_lab = np.array([r.oracle_score for r in rows], dtype=float)
            strata.append(Stratum.from_moments(
                name, weight, f_lab, y_lab,
                n_unlab=int(m["n"]), mean_unlab=float(m["mean"]),
                var_unlab=float(m["var"])))
            # E[f] over the whole layer: labelled and unlabelled together.
            f_pop += weight * ((float(f_lab.sum()) + m["mean"] * m["n"])
                               / max(1, len(rows) + int(m["n"])))

        ppi = ppi_mean_stratified(strata, alpha=self.alpha, seed=self.seed)
        delta = f_pop - ppi.theta
        stamps = [r.dispatched_at for r in records]
        warnings = list(ppi.warnings)

        return Rectification(
            verifier_version=verifier_version,
            delta_hat=delta, delta_se=ppi.se,
            theta=ppi.theta, theta_ci=ppi.ci, se=ppi.se,
            n=ppi.n, n_unlab=ppi.n_unlab, gain_factor=ppi.gain_factor,
            is_stale=False, stale_reason=None,
            resid_sd=population_resid_sd(strata), warnings=warnings,
            computed_at=time.time(), covers=(min(stamps), max(stamps)))

    @staticmethod
    def _pool(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
        """Combine two strata's unlabelled moments without their scores.

        Chan's parallel form. Adding the variances would be wrong by exactly the
        term below -- two layers that are each internally uniform but sit at
        different means have zero variance apiece and plenty combined -- and
        wrong in the direction that makes the interval too narrow.
        """
        na, nb = int(a["n"]), int(b["n"])
        if not na:
            return dict(b)
        if not nb:
            return dict(a)
        n = na + nb
        delta = b["mean"] - a["mean"]
        mean = a["mean"] + delta * nb / n
        m2 = (a["var"] * (na - 1) if na >= 2 else 0.0) \
            + (b["var"] * (nb - 1) if nb >= 2 else 0.0) \
            + delta * delta * na * nb / n
        return {"n": n, "mean": mean, "var": m2 / (n - 1) if n >= 2 else 0.0}

    def _merge_thin(self, grouped: Dict[str, List],
                    moments: Dict[str, Dict[str, float]]):
        """Fold strata below ``min_per_stratum`` into the largest one.

        Merging rather than dropping, and the difference matters. A dropped
        stratum removes its units from the population the estimate describes, so
        the answer silently becomes "the bias among the units we happened to
        sample enough of" -- a different question, and a flattering one when the
        thin stratum is where the verifier is worst.

        Both halves move together. An earlier version merged the labelled
        records and left the unlabelled moments behind, which dropped those units
        from the population count and from ``E[f]`` -- the very silent
        substitution this method exists to prevent, reintroduced inside it.
        """
        thin = [k for k, v in grouped.items() if len(v) < self.min_per_stratum]
        if not thin or len(thin) == len(grouped):
            return dict(grouped), dict(moments)
        host = max((k for k in grouped if k not in thin),
                   key=lambda k: len(grouped[k]))
        merged = {k: list(v) for k, v in grouped.items() if k not in thin}
        pooled = {k: dict(v) for k, v in moments.items() if k not in thin}
        pooled.setdefault(host, {"n": 0, "mean": 0.0, "var": 0.0})
        for k in thin:
            merged[host].extend(grouped[k])
            if k in moments:
                pooled[host] = self._pool(pooled[host], moments[k])
        return merged, pooled
