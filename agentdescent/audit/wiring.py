"""One call that wires the six pieces, because the wiring is where it goes wrong.

Everything in this package is opt-in and nothing in the shipped runtimes builds
any of it, so switching the audit on means assembling six objects and four
cross-references by hand::

    store = AuditStore("audit.jsonl")
    reward = AuditedReward(judge, oracle=GoldAnswer(em), store=store)
    calibrator = Calibrator(store)
    gate = RectifiedAcceptance(calibrator=calibrator,
                               verifier_version=reward.verifier_version)
    watch = VerifierWatch(calibrator, fingerprint=lambda: reward.verifier_version)
    evolve(tasks, reward=reward, run=RenderTap(my_run),
           policies=Policies(acceptance=gate), ...)

Every one of those cross-references is silent when wrong. A ``verifier_version``
copied by hand and then changed in one place leaves the gate asking the
calibrator about a verifier that never ran, and the answer is a **stale**
rectification -- indistinguishable from "we have not collected enough labels
yet", which is what a person will assume. A calibrator pointed at a second store
reads an empty one and is stale forever. A ``RenderTap`` around a different
``run`` than the loop uses records an empty signature on every unit, and the
ordering report then has one artifact.

:func:`attach` builds them together::

    audit = attach(judge, oracle=GoldAnswer(em), store="audit.jsonl", run=my_run)
    evolve(tasks, reward=audit.reward, run=audit.run,
           policies=Policies(acceptance=audit.acceptance), ...)

It changes no default. Not calling it is the current behaviour, and an
:class:`Audit` whose ``enabled`` is ``False`` returns the inner gate's own
decision on the untouched context -- the same call, not an equivalent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

from .calibrator import Calibrator, Rectification
from .gate import RectifiedAcceptance, VerifierWatch
from .sampler import AuditPolicy, SamplePlan
from .sources import GoldAnswer, NullOracle
from .store import AuditStore
from .tap import AuditedReward, RenderTap

__all__ = ["Audit", "attach"]


@dataclass
class Audit:
    """The assembled layer. Hand the three fields to ``evolve()``.

    ``reward``, ``run`` and ``acceptance`` are what the loop takes; ``store``,
    ``calibrator`` and ``watch`` are what a person or another process reads.
    """

    #: Goes in as ``evolve(reward=...)``. Returns the verifier's score, always.
    reward: AuditedReward
    #: Goes in as ``evolve(run=...)``. ``None`` when no ``run`` was given --
    #: the audit still works, it just cannot say *which* artifact produced a
    #: unit, so the ordering report sees one artifact instead of five.
    run: Optional[RenderTap]
    #: Goes in as ``Policies(acceptance=...)``.
    acceptance: RectifiedAcceptance
    store: AuditStore
    calibrator: Calibrator
    #: Withdraws the calibration when the verifier may have moved. Call
    #: :meth:`~agentdescent.audit.gate.VerifierWatch.check` between rounds, or
    #: :meth:`~agentdescent.audit.gate.VerifierWatch.on_merge` from a hook.
    watch: VerifierWatch

    @property
    def verifier_version(self) -> str:
        return self.reward.verifier_version

    @property
    def enabled(self) -> bool:
        return self.acceptance.enabled

    def rectification(self) -> Rectification:
        """The correction in force. Never raises; stale is an answer."""
        return self.calibrator.current(self.verifier_version)

    def recompute(self) -> Rectification:
        """Re-read the store and re-estimate. Call after a batch resolves."""
        return self.calibrator.recompute(self.verifier_version)

    def status(self) -> Dict[str, Any]:
        """What the audit knows, for a round hook or a log line."""
        rect = self.rectification()
        return {
            "verifier_version": self.verifier_version,
            "seen": self.reward.seen,
            "audited": len(self.store),
            "pending": len(self.store.pending()),
            "delta_hat": rect.delta_hat,
            "resid_sd": rect.resid_sd,
            "se": rect.se,
            "is_stale": rect.is_stale,
            "stale_reason": rect.stale_reason,
        }


def attach(verifier: Callable[[Any, str], float], *,
           oracle: Optional[Any] = None,
           store: Union[AuditStore, str, None] = None,
           run: Optional[Callable[[str, Any], str]] = None,
           inner: Any = None,
           enabled: bool = True,
           plan: Optional[SamplePlan] = None,
           policy: Optional[AuditPolicy] = None,
           sample_rate: float = 0.1,
           stratify: Optional[Callable[[Any, str, float], str]] = None,
           calibration_fraction: float = 0.7,
           draw_by: str = "task",
           seed: int = 0,
           verifier_version: Optional[str] = None,
           version_extra: Any = None,
           watch_ids: Iterable[str] = (),
           watch_globs: Sequence[str] = ()) -> Audit:
    """Assemble the audit around ``verifier`` and return what ``evolve()`` needs.

    Parameters
    ----------
    verifier:
        The cheap scorer the loop optimises against, ``(task, output) -> float``.
    oracle:
        Ground truth. A bare callable ``(task, output) -> float`` is wrapped in
        :class:`~agentdescent.audit.sources.GoldAnswer`, because that is what
        every caller with a gold answer already has and asking them to wrap it
        adds an import and a chance to forget that an oracle must never raise
        into the rollout. ``None`` records the questions without answering them.
    store:
        An :class:`~agentdescent.audit.store.AuditStore`, or a path to open one
        at. A path is the usual case: the process that resolves a deferred
        oracle is not this one.
    run:
        The loop's ``run``. Wrapped in a
        :class:`~agentdescent.audit.tap.RenderTap` so each unit records which
        artifact produced it. Leave it out and the audit still estimates the
        bias; it just cannot attribute a unit to an artifact.
    plan:
        A :class:`~agentdescent.audit.sampler.SamplePlan` from a previous round,
        which carries per-stratum rates and overrides ``sample_rate``.
    enabled:
        ``False`` collects records and **does not correct anything** -- the gate
        delegates to ``inner`` on the untouched context. The honest way to run a
        first round: measure before you spend.
    watch_ids, watch_globs:
        Artifact ids and diff-key globs that mean the verifier changed. Nothing
        is watched by default, which is right for a fixed function and exactly
        wrong for a run that evolves its own judge -- see
        :class:`~agentdescent.audit.gate.VerifierWatch`.
    """
    if isinstance(store, str):
        store = AuditStore(store)
    elif store is None:
        store = AuditStore()

    if oracle is None:
        oracle = NullOracle()
    elif not hasattr(oracle, "submit"):
        # A bare `(task, output) -> float`. Wrapping it here is not a
        # convenience: `GoldAnswer` swallows exceptions into `.errors`, and an
        # unwrapped scorer that raises would fail the rollout it was auditing.
        oracle = GoldAnswer(oracle)

    rates = dict(plan.rates) if plan is not None else None
    default_rate = plan.default_rate if plan is not None else sample_rate
    if policy is not None:
        calibration_fraction = policy.calibration_fraction
        if not policy.enabled:
            enabled = False

    reward = AuditedReward(
        verifier, oracle=oracle, store=store, draw_by=draw_by,
        sample_rate=default_rate, stratify=stratify, rates=rates,
        calibration_fraction=calibration_fraction, seed=seed,
        verifier_version=verifier_version, version_extra=version_extra)

    calibrator = Calibrator(store, alpha=policy.alpha if policy else 0.05,
                            seed=seed)
    acceptance = RectifiedAcceptance(
        inner, calibrator=calibrator, enabled=enabled,
        # A callable, not the string: the version is derived from the verifier,
        # and a copy taken now would go stale silently the moment anything mixed
        # into the fingerprint changed.
        verifier_version=lambda: reward.verifier_version)
    watch = VerifierWatch(calibrator,
                          fingerprint=lambda: reward.verifier_version,
                          artifact_ids=watch_ids, key_globs=watch_globs)
    watch.check()                       # establish the baseline, not a change
    return Audit(reward=reward, run=RenderTap(run) if run is not None else None,
                 acceptance=acceptance, store=store, calibrator=calibrator,
                 watch=watch)
