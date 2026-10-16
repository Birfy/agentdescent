"""The verifier bootstrap flywheel: the audit's labels improve the verifier.

The audit package already turns labelled records into a *correction of the
loop*: `Calibrator` produces a rectification (``delta_hat``, ``se``,
``resid_sd``) that `RectifiedAcceptance` uses to discount the gate's evidence.
What it never does is turn those labels into a *better verifier* -- the
correction is subtracted at the decision, and the scorer itself is left
exactly as it was. This module closes that gap under a bootstrap gate.

A :class:`VerifierBoost` wraps any :class:`~agentdescent.policies.
VerifierProtocol` (the shipped :class:`~agentdescent.verifier.ThreeLayerVerifier`)
and, from the audit's **IMPROVEMENT** pool, fits a monotone calibration curve

    g: verifier_score -> oracle-consistent score

(isotonic regression -- pool-adjacent-violators, numpy-only, so score order is
preserved). The curve is **activated only if it provably reduces residual
error on a held-out slice of the same pool**: ``E[(g(f) - Y)^2]`` must beat
``E[(f - Y)^2]`` by ``min_gain``. That is the bootstrap gate, and it is what
keeps a self-improving verifier honest -- a curve that learned to be
confident-but-wrong fails validation and is recorded as a rejection, not
installed.

**What it may and may not touch** (the governance invariants):

* It trains on the **IMPROVEMENT pool only**. The CALIBRATION pool is never
  used for training: that pool exists to *measure* the verifier, and using its
  labels to *change* the verifier would let a verifier fit its own judge
  (``store.for_calibration`` already enforces the two-pool separation).
* It corrects the **cheap and learned layers** -- the scores the aggregator
  uses to rank candidates (conflict resolution, the fusion tournament) and the
  audit's own trust/priority signal. A calibrated cheap layer sends better
  candidates to the commit gate without ever touching the gate.
* It delegates ``eval_counts`` and ``full_eval`` **raw**. The commit decision
  stays on the raw verifier's full-set judgment, rectified by the audit. The
  correction never substitutes for the oracle's authority over what commits.
* It declares itself as a **new verifier version** when it activates
  (:meth:`version_tag`), so the audit's ``VerifierWatch`` withdraws the old
  calibration and fresh CALIBRATION labels measure the boosted verifier. That
  is the recursive part done honestly: the flywheel's output is measured by
  the same audit that produced its training labels, and a boost that is not
  actually better is caught twice -- by the bootstrap gate at install time and
  by the calibration of the new version afterwards.

The uncertainty the learned layer reports is raised to at least the boost's
own validation residual (``max(resid_sd, uncertainty)``) when active, so a
poorly-fitting boost never under-claims uncertainty to the audit's priority
formula (``blast_radius * uncertainty / trust``). A tight residual leaves a
larger inner uncertainty unchanged; a loose residual raises the floor. This is
a safety floor, not a claim that a good boost spends less oracle budget.

**Scale assumption.** The curve is fit on per-unit ``AuditRecord.verifier_score``
(the audited reward) and applied to artifact-level ``cheap_eval`` /
``learned_eval``. Callers must ensure those layer scores live on the same
scale as the audited per-unit reward; for a non-linear ``g``,
``g(mean f) != mean g(f)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..evolvable import Evolvable
from .store import AuditStore

__all__ = ["BoostState", "VerifierBoost", "isotonic_fit"]


# -- the correction learner -----------------------------------------------------


def isotonic_fit(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Monotone non-decreasing regression of ``y`` on ``x`` (PAV, numpy-only).

    Pool-adjacent-violators: sort by ``x``, then repeatedly merge adjacent
    blocks whose fitted value has fallen below its predecessor's. Returns the
    block boundary x-coordinates and the fitted step values -- a piecewise
    constant, monotone curve that never inverts score order (flat PAV blocks may
    still turn distinct raw scores into ties), which is exactly what a
    *ranking* correction must do (an arbitrary regressor could invert two
    candidates' order and break the tournament).

    Deliberately numpy rather than scipy: the audit package's PPI estimator is
    the only numpy consumer today, and scipy is not a dependency.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    x_sum: List[float] = []
    w: List[float] = []
    y_sum: List[float] = []
    for xi, yi in zip(x, y):
        x_sum.append(float(xi))
        w.append(1.0)
        y_sum.append(float(yi))
        while len(w) >= 2 and y_sum[-1] / w[-1] < y_sum[-2] / w[-2]:
            x_sum[-2] += x_sum[-1]
            w[-2] += w[-1]
            y_sum[-2] += y_sum[-1]
            del x_sum[-1], w[-1], y_sum[-1]
    xs = np.array([x_sum[k] / w[k] for k in range(len(w))])
    vals = np.array([y_sum[k] / w[k] for k in range(len(w))])
    return xs, vals


def _mse(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return math.inf
    return float(np.mean((np.asarray(a, dtype=float)
                          - np.asarray(b, dtype=float)) ** 2))


def _finite(v: Optional[float]) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class BoostState:
    """One fit's verdict: what the bootstrap gate decided, and why."""

    active: bool
    fitted: bool
    n_train: int
    n_val: int
    error_before: float
    error_after: float
    reason: str
    fit_at: float
    #: Validation MSE of the best *constant* predictor (the train slice's mean
    #: oracle score). ``inf`` -- also the default, so older constructors keep
    #: working -- means "not measured".
    error_const: float = math.inf


class VerifierBoost:
    """A verifier that improves itself from audited labels, under a bootstrap gate.

    Wraps a :class:`~agentdescent.policies.VerifierProtocol`. On
    :meth:`retrain` it reads the store's IMPROVEMENT pool, fits the monotone
    calibration curve, validates it on a held-out slice, and either activates
    it (the cheap/learned layers become ``g(score)``) or records the rejection.
    ``eval_counts`` / ``full_eval`` are delegated raw -- the commit gate never
    sees the correction.

    ``store`` is the same :class:`~agentdescent.audit.store.AuditStore` the
    audit writes to, so the boost's training pool and the audit's measurement
    pool are one store with two asserted halves.
    """

    def __init__(
        self,
        verifier: Any,
        store: AuditStore,
        *,
        min_labels: int = 50,
        min_val_labels: int = 10,
        validation_frac: float = 0.2,
        min_gain: float = 0.05,
        seed: int = 0,
        max_records: int = 2000,
        auto: bool = True,
        version_for_training: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        if not (0.0 <= min_gain <= 1.0):
            raise ValueError(f"min_gain must be in [0, 1], got {min_gain}")
        if not (0.0 < validation_frac < 1.0):
            raise ValueError(f"validation_frac must be in (0, 1), got {validation_frac}")
        if int(min_labels) <= int(min_val_labels):
            raise ValueError(
                f"min_labels ({min_labels}) must be greater than "
                f"min_val_labels ({min_val_labels}) so the training split "
                f"cannot be emptied by the validation holdout")
        self._inner = verifier
        self.store = store
        self.min_labels = int(min_labels)
        self.min_val_labels = int(min_val_labels)
        self.validation_frac = float(validation_frac)
        self.min_gain = float(min_gain)
        self.seed = int(seed)
        self.max_records = int(max_records)
        self._version_for_training = version_for_training
        self._xs: Optional[np.ndarray] = None
        self._vals: Optional[np.ndarray] = None
        self._fit_hash: str = ""
        self._resid_sd = math.inf
        self._state = BoostState(False, False, 0, 0, math.inf, math.inf,
                                 "not fitted yet", 0.0)
        # Preserve the wrapped verifier's full-set equivalence declaration
        # (missing / legacy False stay independent measurements -- #273 P1).
        from ..verifier import shares_eval_counts
        self.full_eval_matches_counts = shares_eval_counts(verifier)
        if auto:
            self.retrain()

    # -- the flywheel ---------------------------------------------------------

    def retrain(self, verifier_version: Optional[str] = None) -> BoostState:
        """Refit the calibration curve from the current IMPROVEMENT pool.

        The bootstrap gate: fit on the train slice, validate on a held-out
        slice, activate only if the corrected validation error beats the raw
        by ``min_gain``. A failed gate deactivates the boost and keeps the
        wrapped verifier raw -- a curve that does not help is recorded as a
        rejection, not installed.

        ``verifier_version`` (or ``version_for_training``) filters the pool to
        labels scored by the judge currently under measurement, so an older
        verifier's labels cannot train the curve for a newer one.
        """
        version = verifier_version
        if version is None and self._version_for_training is not None:
            version = self._version_for_training()
        recs = [r for r in self.store.for_improvement(version)
                if r.resolved and _finite(r.verifier_score)
                and _finite(r.oracle_score)]
        recs = recs[-self.max_records:]
        if len(recs) < self.min_labels:
            return self._deactivate(
                f"insufficient improvement labels ({len(recs)} < "
                f"{self.min_labels})")
        rng = random.Random(self.seed)
        shuffled = list(recs)
        rng.shuffle(shuffled)
        n_val = max(self.min_val_labels, int(len(shuffled) * self.validation_frac))
        # Keep at least one training point: n_val must not consume the pool.
        n_val = min(n_val, len(shuffled) - 1)
        train, val = shuffled[n_val:], shuffled[:n_val]
        if not train:
            return self._deactivate("empty training split after validation holdout")
        try:
            xs = np.array([r.verifier_score for r in train], dtype=float)
            ys = np.array([r.oracle_score for r in train], dtype=float)
            fit_xs, fit_vals = isotonic_fit(xs, ys)
        except Exception as e:  # noqa: BLE001 - a broken fit is a rejection, not a crash
            return self._deactivate(f"isotonic fit failed: {type(e).__name__}: {e}")
        if fit_xs.size == 0 or fit_vals.size == 0:
            return self._deactivate("isotonic fit returned an empty curve")

        def _apply_candidate(f: float) -> float:
            return float(np.interp(float(f), fit_xs, fit_vals,
                                   left=float(fit_vals[0]),
                                   right=float(fit_vals[-1])))

        error_before = _mse([r.verifier_score for r in val],
                            [r.oracle_score for r in val])
        error_after = _mse([_apply_candidate(r.verifier_score) for r in val],
                           [r.oracle_score for r in val])
        # A curve that only shifts the verifier's mean passes the raw-verifier
        # test above while carrying no ranking information at all: it must also
        # beat the best constant predictor. (A constant offset is the audit
        # ``Calibrator``'s job -- ``delta_hat`` -- not the ranking layer's.)
        const = float(np.mean(ys))
        error_const = _mse([const] * len(val), [r.oracle_score for r in val])
        beats_raw = error_after <= error_before * (1.0 - self.min_gain)
        beats_const = error_after <= error_const * (1.0 - self.min_gain)
        if beats_raw and not beats_const:
            return self._deactivate(
                f"no discrimination beyond a constant: corrected error "
                f"{error_after:.4f} does not beat the constant-mean baseline "
                f"{error_const:.4f} by min_gain {self.min_gain} (raw "
                f"{error_before:.4f}); a pure bias shift is the Calibrator's "
                f"job, not the ranking layer's",
                n_train=len(train), n_val=len(val),
                error_before=error_before, error_after=error_after,
                error_const=error_const)
        if beats_raw and np.unique(fit_vals).size < 2:
            return self._deactivate(
                "degenerate calibration curve: fewer than 2 distinct fitted levels",
                n_train=len(train), n_val=len(val),
                error_before=error_before, error_after=error_after,
                error_const=error_const)
        if beats_raw:
            digest = hashlib.sha256(json.dumps({
                "xs": fit_xs.tolist(), "vals": fit_vals.tolist(),
                "n_train": len(train)}, sort_keys=True).encode()).hexdigest()
            # Assign curve + state together only after the gate passes, so a
            # concurrent cheap_eval never applies an unvalidated curve under an
            # older active version_tag.
            self._xs, self._vals = fit_xs, fit_vals
            self._fit_hash = digest[:12]
            self._resid_sd = math.sqrt(error_after)
            self._state = BoostState(True, True, len(train), len(val),
                                     error_before, error_after,
                                     "validated", time.time(), error_const)
            return self._state
        return self._deactivate(
            f"bootstrap gate: corrected error {error_after:.4f} is not below "
            f"raw {error_before:.4f} by min_gain {self.min_gain}",
            n_train=len(train), n_val=len(val),
            error_before=error_before, error_after=error_after,
            error_const=error_const)

    def _deactivate(
        self,
        reason: str,
        *,
        n_train: int = 0,
        n_val: int = 0,
        error_before: float = math.inf,
        error_after: float = math.inf,
        error_const: float = math.inf,
    ) -> BoostState:
        """Record a rejected fit. Preserve measured diagnostics when known.

        Call sites that already computed validation MSE (and the constant
        baseline) must pass them through so ``status()`` / ``retrain()`` report
        the numbers that justified the rejection rather than the default
        ``inf`` ("not measured").
        """
        self._xs = None
        self._vals = None
        self._fit_hash = ""
        self._resid_sd = math.inf
        self._state = BoostState(
            False, True, int(n_train), int(n_val),
            float(error_before), float(error_after),
            reason, time.time(), float(error_const),
        )
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state.active

    def _apply(self, f: float) -> float:
        """Map a verifier score through the fitted calibration curve."""
        if self._xs is None or self._vals is None:
            return float(f)
        return float(np.interp(float(f), self._xs, self._vals,
                               left=float(self._vals[0]),
                               right=float(self._vals[-1])))

    def version_tag(self) -> str:
        """The audit's ``version_extra``: changes whenever the boost changes.

        Inactive and active fits produce different tags, and an active tag is
        stable across calls until the next ``retrain`` -- so the audit's
        fingerprint (and with it ``VerifierWatch``'s calibration withdrawal)
        tracks exactly the state the loop is using."""
        if not self._state.active:
            return "boost:inactive"
        return f"boost:active:{self._fit_hash}"

    def status(self) -> dict:
        s = self._state
        return {
            "active": s.active,
            "fitted": s.fitted,
            "n_train": s.n_train,
            "n_val": s.n_val,
            "error_before": s.error_before,
            "error_after": s.error_after,
            "error_const": s.error_const,
            "resid_sd": self._resid_sd,
            "reason": s.reason,
            "fit_at": s.fit_at,
            "version_tag": self.version_tag(),
        }

    # -- the verifier surface --------------------------------------------------
    #
    # The correction touches the layers the aggregator uses to *rank* and the
    # audit uses for its trust signal; the commit gate (`eval_counts` /
    # `full_eval`) is delegated raw. See the module docstring for why.

    def rule_eval(self, artifact: Evolvable) -> float:
        return self._inner.rule_eval(artifact)

    def cheap_eval(self, artifact: Evolvable) -> float:
        f = self._inner.cheap_eval(artifact)
        return self._apply(f) if self._state.active else f

    def learned_eval(self, artifact: Evolvable) -> Tuple[float, float]:
        score, uncertainty = self._inner.learned_eval(artifact)
        if not self._state.active:
            return score, uncertainty
        # Safety floor: never report less uncertainty than the validation
        # residual. Audit priority rises with uncertainty, so this cannot
        # claim a poorly-fitting boost needs less oracle spend.
        return self._apply(score), max(self._resid_sd, uncertainty)

    def eval_counts(self, artifact: Evolvable,
                    floor: Optional[float] = None) -> Tuple[float, float]:
        return self._inner.eval_counts(artifact, floor=floor)

    def full_eval(self, artifact: Evolvable) -> float:
        return self._inner.full_eval(artifact)

    def oracle_eval(self, artifact: Evolvable) -> float:  # legacy alias
        return self._inner.full_eval(artifact)
