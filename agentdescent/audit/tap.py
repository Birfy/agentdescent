"""The tap: a reward function that also, sometimes, asks the truth.

:class:`AuditedReward` wraps the cheap verifier and satisfies
:data:`~agentdescent.evolution.Reward` exactly, so it goes in where the reward
already goes::

    evolve(tasks, reward=AuditedReward(judge, oracle=GoldAnswer(exact_match)), ...)

**It returns the verifier's score, always, unchanged.** The oracle's answer never
reaches the loop: not the acceptance gate, not the tournament, not the regression
guard. Turning the audit on cannot change which diffs commit, and that is a
property of the shape of this class rather than of anyone remembering it -- the
one call that could leak truth into the loop is the `return`, and it returns `f`.

## Why here and not at the verifier

:class:`~agentdescent.verifier.ThreeLayerVerifier` looks like the natural home,
and it is the wrong one. Its layers all score ``(artifact, tasks) -> float``: an
aggregate over a task list. Estimating a bias needs *paired* observations --
this unit scored `f` by the verifier and `Y` by the truth -- and an aggregate has
already averaged the pairing away. Worse, a verifier-level oracle would be asked
for a fresh measurement, so `f` and `Y` would come from two different rollouts
and their difference would carry rollout variance that no amount of sampling
removes.

At the reward level both score the **same output**, which the artifact was run
to produce anyway. The expensive half is already paid; the oracle adds only its
own cost. And because this sits below the verifier, the merge path is untouched:
:meth:`~agentdescent.aggregator.Aggregator._audit`, which calls
``verifier.oracle_eval`` synchronously and would happily block a merger for the
duration of a wet-lab experiment, is never involved.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Callable, Dict, Optional

from .records import (AuditRecord, Purpose, new_record_id, output_digest,
                      verifier_fingerprint)
from .sources import NullOracle
from .store import AuditStore

#: Rendered artifact for the rollout currently being scored on this thread.
#: `eval_one` calls `run` and then `reward` on one thread, so a thread-local set
#: by :class:`RenderTap` is readable by :class:`AuditedReward` -- and a run
#: without a `RenderTap` simply records an empty signature rather than failing.
_current_render = threading.local()


class RenderTap:
    """Optional wrapper for ``run`` that records *which* artifact produced an output.

    The reward signature is ``(task, output)``: it never sees the artifact, so
    without this the audit knows what was answered but not what answered it. That
    is enough to estimate a bias -- the pairing key is ``(task_id, output)`` --
    and not enough to go back and ask which state was responsible for a bad one.

    Wrap ``run`` with it when you want the second::

        evolve(tasks, reward=audited, run=RenderTap(my_run), ...)
    """

    def __init__(self, run: Callable[[str, Any], str]) -> None:
        self.run = run

    def __call__(self, rendered: str, task: Any) -> str:
        _current_render.signature = hashlib.sha256(
            rendered.encode("utf-8")).hexdigest()[:16]
        return self.run(rendered, task)


def _take_signature() -> str:
    """Read the pending signature **and clear it**. One render, one reward.

    Leaving it set was silently wrong rather than merely untidy. `eval_one` is
    `reward(task, run(render, task))` -- one render per reward -- but the reward
    is also reachable without a render in front of it (a caller scoring an
    output by hand, a code path that does not go through `run`, a `RenderTap`
    installed for only part of a run). A sticky thread-local attributes those to
    whichever artifact this thread rendered last, which is a wrong provenance
    that looks exactly like a right one. Consuming it makes the unattributable
    case empty, which is honest and visible.
    """
    signature = getattr(_current_render, "signature", "")
    _current_render.signature = ""
    return signature


class AuditedReward:
    """A cheap verifier that hands a sampled minority of its work to the truth.

    Parameters
    ----------
    verifier:
        The cheap scorer the loop optimises against -- an agent judging the
        output, a learned scorer, a heuristic. ``(task, output) -> float``.
    oracle:
        Ground truth. :class:`~agentdescent.audit.sources.GoldAnswer` when it
        returns now, :class:`~agentdescent.audit.sources.DeferredOracle` when it
        arrives later. Defaults to
        :class:`~agentdescent.audit.sources.NullOracle`, which still records the
        questions -- useful when the answerer has not been asked yet.
    store:
        Where records go. Defaults to an in-memory store; pass
        ``AuditStore("audit.jsonl")`` to keep them.
    sample_rate:
        Probability a unit is audited, when no stratum-specific rate applies.
    stratify:
        Optional ``(task, output, verifier_score) -> str``. Names the layer a
        unit belongs to, so ``rates`` can spend more of the budget where the
        residual varies most. The stratum is recorded either way.
    rates:
        Per-stratum inclusion probabilities, falling back to ``sample_rate``.
    calibration_fraction:
        Share of audited units assigned :attr:`Purpose.CALIBRATION`; the rest
        become :attr:`Purpose.IMPROVEMENT`. The split is drawn at random rather
        than taken in order, because units arrive grouped by task and by
        artifact and any ordered split would correlate the two pools with
        whatever the ordering happens to encode.
    seed:
        Base seed for the inclusion draw.
    verifier_version:
        Overrides the fingerprint derived from ``verifier``. Pass one when the
        verifier is an agent whose behaviour lives in a prompt or a model id
        rather than in the source of the function.
    version_extra:
        Mixed into the derived fingerprint. The cheaper way to say the same
        thing: ``version_extra={"model": "...", "prompt_sha": "..."}``.

    Notes
    -----
    The inclusion draw is seeded **per unit**, from
    ``(seed, verifier_version, task_id, output)``, not taken from a shared
    stream. Evaluation is concurrent (``eval_concurrency`` defaults to 8), and a
    shared stream would make whether a unit is audited depend on how many other
    units happened to be scored first -- so the same run, replayed, would audit a
    different sample and the seed would document nothing.
    :meth:`~agentdescent.verifier.ThreeLayerVerifier.learned_eval` seeds
    per-artifact for the same reason.
    """

    def __init__(self, verifier: Callable[[Any, str], float], *,
                 oracle: Optional[Any] = None,
                 store: Optional[AuditStore] = None,
                 sample_rate: float = 0.1,
                 stratify: Optional[Callable[[Any, str, float], str]] = None,
                 rates: Optional[Dict[str, float]] = None,
                 calibration_fraction: float = 0.7,
                 seed: int = 0,
                 verifier_version: Optional[str] = None,
                 version_extra: Any = None) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0, 1], got {sample_rate!r}")
        if not 0.0 <= calibration_fraction <= 1.0:
            raise ValueError(
                f"calibration_fraction must be in [0, 1], got {calibration_fraction!r}")
        self.verifier = verifier
        self.oracle = oracle if oracle is not None else NullOracle()
        self.store = store if store is not None else AuditStore()
        self.sample_rate = sample_rate
        self.stratify = stratify
        self.rates = dict(rates or {})
        self.calibration_fraction = calibration_fraction
        self.seed = seed
        self.verifier_version = verifier_version or verifier_fingerprint(
            verifier, extra=version_extra)
        #: Units seen, and units audited. A sample rate that never fires is the
        #: quiet failure this makes loud.
        self.seen = 0
        self.audited = 0
        self._lock = threading.Lock()

    # -- the Reward contract -------------------------------------------------

    def __call__(self, task: Any, output: str) -> float:
        score = float(self.verifier(task, output))
        try:
            self._maybe_audit(task, output, score)
        except Exception:  # noqa: BLE001
            # An audit is a side channel. It must not be able to fail a rollout
            # the loop would otherwise have scored, because that would make
            # enabling the audit change the run -- the one thing this layer
            # promises it cannot do. Losing a record is the cheaper failure.
            pass
        return score

    # -- sampling ------------------------------------------------------------

    def rate_for(self, stratum: str) -> float:
        return self.rates.get(stratum, self.sample_rate)

    def _draw(self, task: Any, output: str) -> float:
        """A uniform ``[0, 1)`` deterministic in the unit, not in call order."""
        key = "\0".join([str(self.seed), self.verifier_version,
                         str(getattr(task, "id", task)), output_digest(output)])
        h = hashlib.sha256(key.encode("utf-8")).digest()
        # Two independent draws from one hash: separate byte ranges, so adding
        # the purpose split did not shift which units get included.
        return int.from_bytes(h[:8], "big") / float(1 << 64)

    def _purpose_draw(self, task: Any, output: str) -> float:
        key = "\0".join(["purpose", str(self.seed), self.verifier_version,
                         str(getattr(task, "id", task)), output_digest(output)])
        h = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(h[8:16], "big") / float(1 << 64)

    def _maybe_audit(self, task: Any, output: str, score: float) -> Optional[AuditRecord]:
        with self._lock:
            self.seen += 1
        stratum = "all" if self.stratify is None else str(
            self.stratify(task, output, score))
        prob = self.rate_for(stratum)
        if prob <= 0.0 or self._draw(task, output) >= prob:
            return None

        purpose = (Purpose.CALIBRATION
                   if self._purpose_draw(task, output) < self.calibration_fraction
                   else Purpose.IMPROVEMENT)
        record = AuditRecord(
            record_id=new_record_id(),
            task_id=str(getattr(task, "id", task)),
            artifact_signature=_take_signature(),
            output=output,
            verifier_version=self.verifier_version,
            verifier_score=score,
            inclusion_prob=prob,
            purpose=purpose,
            stratum=stratum,
            sampler_seed=self.seed,
        )
        self.store.append(record)
        with self._lock:
            self.audited += 1

        truth = self.oracle.submit(record, task, output)
        if truth is not None:
            self.store.resolve(record.record_id, float(truth))
        return record

    # -- reporting -----------------------------------------------------------

    def pending(self) -> list:
        """Audited units still waiting on truth."""
        return self.store.pending(verifier_version=self.verifier_version)

    def calibration_set(self) -> list:
        """Resolved CALIBRATION records for *this* verifier version."""
        return self.store.for_calibration(self.verifier_version)

    def __repr__(self) -> str:
        return (f"AuditedReward(version={self.verifier_version!r}, "
                f"seen={self.seen}, audited={self.audited}, "
                f"oracle={type(self.oracle).__name__})")
