"""Spending the measured bias: the audit's number reaching the gate that commits.

Everything before this module measures. This one is the only place the
measurement changes an outcome, and it is deliberately the smallest piece of the
package.

The plan's Phase 4 says: correct the measured rate and carry the correction's
uncertainty::

    p_true = p_hat - delta_hat
    var_true = var_p + se ** 2

Both halves of that turn out to be aimed slightly wrong, and the real Phase 0
audit says by how much. On 177 HotpotQA pairs, with the shipped gate reading 32
held-out tasks at a measured rate of 0.688:

===========================  ===========  =====================================
term                         value        share of the binomial term (0.00671)
===========================  ===========  =====================================
``sigma_eps ** 2 / n``       0.00454      68%
``se(delta) ** 2``           0.00082      12%
===========================  ===========  =====================================

The plan carries the 12% term and omits the 68% one, which is **5.5x larger**.

Why the correction itself mostly does not matter
------------------------------------------------

``delta_hat`` is one number applied to *both* sides of a comparison, so it
cancels out of ``cand - base`` exactly. So does its standard error. A gate that
asks "is this candidate better than that one" is almost immune to a verifier
that is uniformly generous -- which is the good news, and is why this module
does not read as a correction at all.

What does **not** cancel is the verifier's *disagreement* with the truth on each
side's own held-out set. ``mean(f) - Delta`` estimates ``mean(Y)`` with variance
``sigma_eps ** 2 / n``, independently on each side, and that term is pure
uncertainty the gate has been spending as if it were evidence. Correcting a bias
the gate never suffered from, while ignoring the noise it did, is the shape of
the mistake worth naming here.

``delta_hat`` is still applied, for two reasons that are real but secondary:

* **The variance scale.** A Beta posterior's spread is ``p(1-p)/(A+1)``. At a
  measured 0.90 that is 0.09; at the true 0.73 it is 0.20. Reporting the wrong
  ``p`` makes the gate 2.2x overconfident about a difference in either direction.
* **The rates in the refusal.** ``held-out regression 0.812 -> 0.781`` is read by
  a person, and two numbers that are both 0.17 too high are two wrong numbers.

And ``se(delta)`` is carried, once rather than twice, under a name that says what
it is standing in for: :attr:`Adjustment.drift`, the allowance for ``Delta`` not
actually being the same on both sides. It would not be, if a candidate shifted
its outputs into a stratum where the verifier is more generous -- which is the
failure this whole package exists to catch. ``se`` is not an estimate of that
drift; it is the only number to hand of roughly the right size, and the caller
can pass a better one.

How it is applied
-----------------

By **discounting the counts**, not by editing the posterior. A rate measured by
a noisy proxy over ``n`` tasks is worth some smaller number of tasks scored by
an oracle, and :func:`discount_for` solves for exactly that number. Scaling
``(successes, failures)`` by the resulting ``kappa <= 1`` leaves the rate
untouched, which means:

* the regression guard, which reads rates, is untouched;
* ``observed_delta``, which the aggregator folds back into the artifact's prior,
  is untouched;
* the artifact's prior is untouched -- it is separate evidence, and not
  something the verifier measured;
* every acceptance policy benefits, not just the shipped one, because the change
  is in the context rather than in the rule.

On the Phase 0 numbers, ``kappa`` is about **0.60**: thirty-two tasks judged by
that LLM judge carry the information of nineteen judged by exact match. A
candidate that scores 0.625 -> 0.750 commits on the first reading and does not
commit on the second.

The clipping is one-directional on purpose. Shifting a rate out of ``[0, 1]``
and clipping it back shrinks the gap between the two sides, never widens it, so
the failure mode of the correction is a candidate that does not commit.

Constraint 1 -- the audit must not be able to change which diffs commit --
survives because ``enabled=False`` returns ``inner.accept(ctx)`` on the
untouched context. Not "equivalent to": the same call.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from ..advantage import _ForwardsInstall
from .calibrator import STALE_INFLATION, Calibrator, Rectification

__all__ = ["Adjustment", "RectifiedAcceptance", "VerifierWatch",
           "discount_for", "rectified_counts"]

#: Never scale the measured counts below this fraction of themselves. Not a
#: guard against over-correction -- a gate that cannot tell two candidates apart
#: *should* refuse both, and at this floor it does -- but against counts of
#: exactly zero, where the rate is no longer defined and the refusal would
#: report ``0.000 -> 0.000`` for a candidate that swept the held-out set.
MIN_KAPPA = 1e-3


@dataclass(frozen=True)
class Adjustment:
    """What the audit did to one merge decision, and why.

    Returned by :meth:`RectifiedAcceptance.explain` so a caller can log the
    arithmetic without re-running the gate. ``applied=False`` means the context
    reached the inner policy untouched.
    """

    applied: bool
    reason: str
    #: ``E[f] - E[Y]`` as applied. Zero when only the variance was adjusted.
    delta_hat: float = 0.0
    #: Population sd of ``f - Y``. The term the plan omits.
    sigma_eps: float = 0.0
    #: Allowance for ``Delta`` differing between the two sides, as a standard
    #: deviation. Split evenly across them, so the comparison carries it once.
    drift: float = 0.0
    #: Count multiplier actually applied, per side.
    kappa_base: float = 1.0
    kappa_cand: float = 1.0
    #: Variance of the candidate's measured *rate* -- as an estimate of the
    #: verifier's own mean (``var_before``) and of the truth (``var_after``).
    #: Not posterior variances: the gate's prior is left out of this arithmetic
    #: entirely (see :func:`discount_for`).
    var_before: float = 0.0
    var_after: float = 0.0
    #: True when the correction's own uncertainty exceeds the measurement's --
    #: the gate is now audit-limited, and more held-out tasks will not help.
    audit_limited: bool = False
    stale: bool = False

    @property
    def evidence_kept(self) -> float:
        """Fraction of the candidate's held-out evidence that survived."""
        return self.kappa_cand

    def to_detail(self) -> str:
        """One clause, for the tail of a refusal a person will read."""
        if not self.applied:
            return ""
        if self.stale:
            return (f"audit stale ({self.reason}), evidence discounted to "
                    f"{self.kappa_cand:.2f} of itself")
        bits = [f"audit: delta_hat {self.delta_hat:+.3f}",
                f"sigma_eps {self.sigma_eps:.3f}",
                f"evidence x{self.kappa_cand:.2f}"]
        if self.audit_limited:
            bits.append("AUDIT-LIMITED: the correction is less certain than the "
                        "measurement, so more held-out tasks will not help")
        return "; ".join(bits)


def rectified_counts(counts: Tuple[float, float], delta: float
                     ) -> Tuple[Tuple[float, float], float]:
    """Shift ``(successes, failures)`` so the rate reads ``p - delta``.

    Returns the new counts and the shift that was actually achieved, which is
    smaller in magnitude than ``delta`` when the shift would leave ``[0, 1]``.
    Both sides of a comparison are shifted by the same ``delta``, so a clipped
    shift can only bring them closer together -- the conservative direction.
    """
    successes, failures = counts
    n = successes + failures
    if n <= 0.0 or not delta:
        return counts, 0.0
    p = successes / n
    moved = min(1.0, max(0.0, p - delta))
    return (n * moved, n * (1.0 - moved)), p - moved


def discount_for(counts: Tuple[float, float], extra_var: float,
                 *, min_kappa: float = MIN_KAPPA) -> Tuple[float, float, float]:
    """How many of these observations are worth believing, given ``extra_var``.

    A held-out rate over ``n`` tasks has binomial variance ``p (1 - p) / n``
    *about the verifier's own mean*. About the **truth** it also carries
    ``extra_var``. The two are the same statement as: it is worth a smaller
    number of oracle-scored tasks. Setting the variances equal,

    ::

        p (1 - p) / n_eff = p (1 - p) / n + extra_var
        kappa = n_eff / n = p (1 - p) / (p (1 - p) + n * extra_var)

    ``p`` is the Laplace-smoothed rate ``(s + 1) / (n + 2)`` -- the mean of the
    posterior the gate is about to build -- so a sweep of 32/32 has a variance
    scale rather than a zero, and ``kappa`` is defined everywhere.

    The gate's **prior does not appear**, and that is the correction to an
    earlier version of this function which targeted a posterior variance
    instead. The prior is separate information, not something the verifier
    measured; and targeting the posterior made the achievable widening depend on
    how much history an artifact had, so an artifact with forty commits behind it
    could not be made to doubt its verifier at all. Discounting the measurement
    and leaving the prior to speak for itself is both simpler and the thing that
    was meant.

    Returns ``(kappa, var_before, var_after)``, the last two being variances of
    the *mean* rather than of a single observation.
    """
    successes, failures = counts
    n = successes + failures
    if n <= 0.0:
        return 1.0, 0.0, 0.0
    p = (successes + 1.0) / (n + 2.0)
    spread = p * (1.0 - p)
    var = spread / n
    if extra_var <= 0.0:
        return 1.0, var, var
    kappa = spread / (spread + n * extra_var)
    return min(1.0, max(min_kappa, kappa)), var, var + extra_var


class RectifiedAcceptance(_ForwardsInstall):
    """An acceptance gate that knows its measurement came from a proxy.

    Wraps another :class:`~agentdescent.policies.AcceptancePolicy`. The Beta
    test, the regression guard and the annealing schedule are untouched; what
    changes is the evidence handed to them, which is discounted by however much
    the verifier disagrees with ground truth.

    ::

        gate = RectifiedAcceptance(calibrator=cal, verifier_version=version)
        evolve(tasks, policies=Policies(acceptance=gate), ...)

    Parameters
    ----------
    inner:
        The rule that actually decides. Defaults to the shipped gate, with the
        run's thresholds filled in by the aggregator at install time.
    calibrator:
        Where the correction comes from. Re-read on every decision, so a
        rectification that goes stale mid-run takes effect at the next merge.
    verifier_version:
        The version to ask the calibrator about -- a string, or a callable
        returning one for a verifier that can change under the run.
    rectification:
        A fixed correction instead of a calibrator. For a run that measured its
        bias once, offline, and does not intend to keep measuring.
    enabled:
        ``False`` delegates to ``inner`` on the untouched context. This is the
        constraint that lets the audit be turned on mid-run: off, it is not
        approximately the old behaviour, it *is* the old call.
    drift_allowance:
        Standard deviation to carry for ``Delta`` differing between the two
        sides being compared. ``None`` uses the rectification's own ``se``,
        which is the right order of magnitude and not an estimate of the thing
        (see the module docstring).
    inflate_when_stale:
        Variance multiplier while no usable correction exists. ``1.0`` passes
        through instead, which is the choice to treat "we have not measured the
        verifier" and "the verifier is unbiased" as the same claim.
    """

    def __init__(self, inner: Any = None, *,
                 calibrator: Optional[Calibrator] = None,
                 verifier_version: Union[str, Callable[[], str]] = "",
                 rectification: Optional[Rectification] = None,
                 enabled: bool = True,
                 drift_allowance: Optional[float] = None,
                 inflate_when_stale: float = STALE_INFLATION,
                 explain_refusals: Optional[bool] = None,
                 min_kappa: float = MIN_KAPPA) -> None:
        from ..defaults import DefaultAcceptance

        if inflate_when_stale < 1.0:
            raise ValueError("inflate_when_stale below 1.0 would make an "
                             "unmeasured verifier count for more than a "
                             "measured one")
        if not 0.0 < min_kappa <= 1.0:
            raise ValueError("min_kappa must be in (0, 1]")
        self.inner = inner if inner is not None else DefaultAcceptance()
        # Saying *which* refusals the audit caused is the number that tells a
        # reader whether the audit is earning its budget, and it costs a second
        # `inner.accept` on the untouched context -- the whole inner gate re-run,
        # not "one extra Monte-Carlo draw" as this used to claim.
        #
        # `None` means: only for a policy we know is safe to run twice.
        # `DefaultAcceptance.accept` computes and returns; it mutates nothing.
        # An arbitrary policy may record a decision, decrement a budget, call a
        # model or advance a counter, and would do it twice for one merge and
        # only on the refusals -- a divergence that shows up in some runs and
        # not others. Its owner knows whether that is safe; this does not.
        #
        # Keyed on the policy rather than on who supplied it: passing a
        # *configured* `DefaultAcceptance` is the ordinary way to use this, and
        # a rule keyed on "did the caller pass something" would have silently
        # dropped the attribution for it.
        self.explain_refusals = (isinstance(self.inner, DefaultAcceptance)
                                 if explain_refusals is None
                                 else explain_refusals)
        self.calibrator = calibrator
        self.verifier_version = verifier_version
        self.rectification = rectification
        self.enabled = enabled
        self.drift_allowance = drift_allowance
        self.inflate_when_stale = inflate_when_stale
        self.min_kappa = min_kappa

    # -- what the correction is ----------------------------------------------

    def version(self) -> str:
        v = self.verifier_version
        return v() if callable(v) else v

    def current(self) -> Optional[Rectification]:
        """The rectification in force, or ``None`` if there is no source."""
        if self.calibrator is not None:
            return self.calibrator.current(self.version())
        return self.rectification

    def explain(self, ctx) -> Adjustment:
        """What :meth:`accept` would do to ``ctx``, without deciding anything."""
        if not self.enabled:
            return Adjustment(False, "disabled")
        rect = self.current()
        if rect is None:
            return self._widen(ctx, "no calibrator and no rectification")
        if rect.is_stale:
            return self._widen(ctx, rect.stale_reason or "stale")
        sigma = float(rect.resid_sd)
        if not sigma == sigma or sigma < 0.0:      # NaN, or a nonsense sd
            return self._widen(
                ctx, "the rectification carries no residual sd, and that is the "
                     "term the variance is mostly made of")
        drift = (rect.se if self.drift_allowance is None
                 else self.drift_allowance)
        drift = 0.0 if drift != drift else abs(float(drift))
        delta = float(rect.delta_hat)
        delta = 0.0 if delta != delta else delta

        base_n = max(0.0, sum(ctx.base_counts))
        cand_n = max(0.0, sum(ctx.cand_counts))
        # sigma_eps**2 / n on each side; the drift split evenly so that the
        # *difference* the gate reads carries it exactly once.
        extra_base = (sigma ** 2 / base_n if base_n else 0.0) + drift ** 2 / 2.0
        extra_cand = (sigma ** 2 / cand_n if cand_n else 0.0) + drift ** 2 / 2.0

        shifted_base, _ = rectified_counts(ctx.base_counts, delta)
        shifted_cand, _ = rectified_counts(ctx.cand_counts, delta)
        k_base, _, _ = discount_for(shifted_base, extra_base,
                                    min_kappa=self.min_kappa)
        k_cand, v_before, v_after = discount_for(
            shifted_cand, extra_cand, min_kappa=self.min_kappa)
        return Adjustment(
            True, "rectified", delta_hat=delta, sigma_eps=sigma, drift=drift,
            kappa_base=k_base, kappa_cand=k_cand,
            var_before=v_before, var_after=v_after,
            audit_limited=drift ** 2 > v_before, stale=False)

    # -- the decision ---------------------------------------------------------

    def accept(self, ctx):
        adj = self.explain(ctx)
        if not adj.applied:
            # Constraint 1: the same call the run would have made without the
            # audit, on the same object.
            return self.inner.accept(ctx)

        decision = self.inner.accept(self._adjusted(ctx, adj))
        if decision.accept:
            return decision
        note = adj.to_detail()
        if self.explain_refusals:
            # A refusal is the only place the audit can have changed an outcome.
            # See `__init__` for why this is not done on a caller's own policy
            # unless they ask.
            if self.inner.accept(ctx).accept:
                note = f"refused by the audit -- {note}"
        return replace(decision, detail=f"{decision.detail}; {note}"
                       if decision.detail else note)

    # -- internals ------------------------------------------------------------

    def _adjusted(self, ctx, adj: Adjustment):
        base, _ = rectified_counts(ctx.base_counts, adj.delta_hat)
        cand, _ = rectified_counts(ctx.cand_counts, adj.delta_hat)
        return replace(
            ctx,
            base_counts=(base[0] * adj.kappa_base, base[1] * adj.kappa_base),
            cand_counts=(cand[0] * adj.kappa_cand, cand[1] * adj.kappa_cand))

    def _widen(self, ctx, reason: str) -> Adjustment:
        """No usable correction: keep the rates, spend less of the evidence."""
        if self.inflate_when_stale <= 1.0:
            return Adjustment(False, f"{reason} (pass-through)")
        k_base, _, _ = self._stale_kappa(ctx.base_counts)
        k_cand, v_before, v_after = self._stale_kappa(ctx.cand_counts)
        return Adjustment(True, reason, kappa_base=k_base, kappa_cand=k_cand,
                          var_before=v_before, var_after=v_after, stale=True)

    def _stale_kappa(self, counts):
        """No number to correct with, so spend ``1 / inflate`` of the evidence.

        Which is exactly what the multiplier means: ``kappa = var / (var *
        inflate)``, independent of the rate and of ``n``.
        """
        _, var, _ = discount_for(counts, 0.0, min_kappa=self.min_kappa)
        extra = var * (self.inflate_when_stale - 1.0)
        return discount_for(counts, extra, min_kappa=self.min_kappa)


class VerifierWatch:
    """Marks a calibrator stale when the instrument it calibrated may have moved.

    A correction estimated for one verifier says nothing about the next, and
    nothing in the arithmetic notices: feed it a changed verifier and it returns
    a confident number for a question nobody asked. There is no way to detect
    this after the fact, which is why the hook is here and not in a report.

    Two signals, and they cover different things:

    * :meth:`check` recomputes the verifier's fingerprint and compares. Exact,
      and blind to the case that matters most -- a verifier whose *prompt* the
      loop is evolving has the same module, qualname and source. Pass a
      fingerprint function that folds the prompt in (``verifier_fingerprint(fn,
      extra=prompt)``) and it stops being blind.
    * :meth:`on_merge` fires on a committed diff that could be the verifier,
      named by artifact id, by governance layer, or by glob over the diff's op
      keys.

    Nothing is watched by default. A caller that names nothing gets the
    fingerprint check alone, which is the honest default for a run whose
    verifier is a fixed function -- and exactly the wrong one for a run that
    evolves its own judge. A false positive here costs one recompute; a false
    negative is the failure the package exists to prevent, so name too much
    rather than too little.
    """

    def __init__(self, calibrator: Calibrator, *,
                 fingerprint: Optional[Callable[[], str]] = None,
                 artifact_ids: Iterable[str] = (),
                 key_globs: Sequence[str] = (),
                 layers: Iterable[int] = ()) -> None:
        self.calibrator = calibrator
        self.fingerprint = fingerprint
        self.artifact_ids = frozenset(artifact_ids)
        self.key_globs = tuple(key_globs)
        self.layers = frozenset(int(l) for l in layers)
        self._seen: Optional[str] = None

    def check(self) -> bool:
        """Re-read the fingerprint; mark stale and return True if it changed."""
        if self.fingerprint is None:
            return False
        now = self.fingerprint()
        was, self._seen = self._seen, now
        if was is None or was == now:
            return False
        self.calibrator.mark_stale(
            f"verifier fingerprint changed {was!r} -> {now!r}")
        return True

    def on_merge(self, artifact, diff) -> bool:
        """Call after a diff commits. True means the calibration was withdrawn."""
        why = self._why(artifact, diff)
        if why is None:
            return False
        self.calibrator.mark_stale(why)
        return True

    def _why(self, artifact, diff) -> Optional[str]:
        artifact_id = getattr(artifact, "id", None) or getattr(diff, "target", "")
        if artifact_id in self.artifact_ids:
            return f"{artifact_id!r} committed, and it is the verifier"
        if self.layers:
            from ..governance import classify

            try:
                layer = classify(artifact)
            except Exception:                      # not an Evolvable; skip
                layer = None
            if layer is not None and int(layer) in self.layers:
                return (f"{artifact_id!r} committed at governance layer "
                        f"{getattr(layer, 'name', layer)}, which is watched")
        if self.key_globs and diff is not None:
            keys = list(getattr(diff, "ops", {}) or {})
            hit = [k for k in keys
                   if any(fnmatch.fnmatch(str(k), g) for g in self.key_globs)]
            if hit:
                return (f"{artifact_id!r} committed edits to "
                        f"{', '.join(sorted(map(str, hit))[:3])}, which the "
                        f"verifier reads")
        return None
