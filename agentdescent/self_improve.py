"""Self-improvement control loop: decide what to improve next, and do it.

The framework's rungs each know one thing -- :mod:`agentdescent.portfolio` knows
which decision slot's evolution is paying for itself, :mod:`agentdescent.evidence`
reconciles the archive's per-slot and per-port evidence, :func:`agentdescent.meta.meta_run`
evolves one slot and validates it on unseen problems, and
:mod:`agentdescent.admit` turns an adoption's trust rings into one verdict. What
none of them does alone is **decide the next move**: which slots to improve, on
which domains, and how to move the budget afterwards. This module is that loop.

One **tick** is:

1. **read evidence** -- :func:`~agentdescent.evidence.snapshot` composes the
   portfolio's per-slot return-per-cost, the harness leaderboard, and the exam
   stamps; :func:`~agentdescent.evidence.rank_slots` picks the slots whose
   evolution is paying (falling back to round-robin on a cold archive);
2. **evolve** each target slot on each domain with
   :func:`~agentdescent.meta.meta_run` (adopt -> :func:`~agentdescent.meta.archive_rule`);
3. **trust-gate** each adoption with :func:`~agentdescent.admit.admit_record`
   (a demote/flag downgrades the tick's verdict);
4. **move budget** -- :func:`~agentdescent.portfolio.portfolio_report` returns
   the next tick's per-slot budget from the updated archive.

Ticks repeat a bounded ``ticks`` times; the report is the per-tick decision
trail. The evolution step is intentionally a plain per-slot ``meta_run`` so the
loop is deterministic and dependency-free; a joint or population variant can be
substituted later without changing the loop's shape.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .admit import AdmitVerdict, admit_record
from .consolidation import ConsolidationStore
from .evidence import rank_slots, snapshot
from .meta import (
    SLOTS, MetaReward, SlotSpec, _archived_rules, _require_disjoint,
    archive_rule, meta_run, problem_descriptor,
)
from .portfolio import layers_from_archive, portfolio_report
from .portfolio_evolve import ReallocationPolicy

__all__ = ["SelfImproveConfig", "SelfImproveTick", "SelfImproveReport",
           "self_improve"]

Domain = Tuple[Any, Any]          # (train problems, val problems)
AdmitFn = Callable[[Any], Any]    # record -> AdmitVerdict or a status string


@dataclass(frozen=True)
class SelfImproveConfig:
    """What the loop improves, on what, and how hard.

    ``slots`` is the candidate set and ``specs`` maps each to its gate/compile
    (``meta_run``'s ``spec``). ``domains`` maps a domain name to
    ``(train_problems, val_problems)``; each pair is evolved for every target
    slot. ``top_k`` slots are evolved per tick, chosen by evidence.

    ``seeds``/``val_seeds`` are the outer seeds (must be disjoint, as
    ``meta_run`` requires); ``min_gain`` and ``transfer_floor`` are
    ``meta_run``'s adoption guards (zero gain and a fit that does not transfer
    are both refused). ``min_exam`` is the seeding bar applied before a rule is
    used as a prior (``None`` keeps every non-demoted rule; ``"passed"`` seeds
    only certified ones; ``"flagged"`` also keeps flagged rules).
    ``reallocation`` is the rule that moves the next tick's budget; ``None``
    uses the shipped fixed ``portfolio_report`` policy. ``config`` is the
    domain config whose fingerprint keys the archive rows and the harness
    leaderboard. ``run_id_prefix`` names this session's archived records --
    ``None`` stamps ``self-improve-<epoch>`` so separate sessions append; pass a
    fixed prefix for a reproducible/rerun-is-idempotent session.
    """

    slots: Sequence[str]
    specs: Mapping[str, SlotSpec]
    domains: Mapping[str, Domain]
    ticks: int = 3
    top_k: int = 2
    floor: float = 0.05
    min_samples: int = 1
    min_gain: float = 0.0
    transfer_floor: float = 0.0
    min_exam: Optional[str] = None
    reallocation: Optional[ReallocationPolicy] = None
    artifact_id: str = "meta-rule"
    config: Optional[Dict[str, Any]] = None
    run_id_prefix: Optional[str] = None
    seeds: Sequence[int] = (0, 1, 2, 3)
    val_seeds: Sequence[int] = (4, 5, 6, 7)


@dataclass(frozen=True)
class SelfImproveTick:
    """One pass of the loop: targets, per-slot verdicts, and the budget move."""

    index: int
    targets: Tuple[str, ...]
    #: ``"<slot>:<domain>" -> "adopt" / "reject" / "flag" / "demote"``.
    verdicts: Dict[str, str]
    #: ``"<slot>:<domain>" -> mean held-out gain``.
    gains: Dict[str, float]
    #: Per-layer budget before the tick, keyed as the portfolio names layers
    #: (``meta:<slot>``).
    budget_before: Dict[str, float]
    #: The portfolio's reallocated budgets for the next tick (same keys).
    budget_after: Dict[str, float]
    #: Keys whose adoption passed the trust gate with **no ring consulted**
    #: (the framework's provisional default), so ``pass`` is not confused with
    #: "examined and passed".
    unexamined: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class SelfImproveReport:
    """The whole session: every tick, whether anything improved, total gain.

    ``total_gain`` sums the mean held-out gains of the adopted ``slot:domain``
    rows, so a domain list weights a config more heavily -- read it as a rough
    effort signal, not a normalised score. Per-row gains are on each tick.
    """

    ticks: Tuple[SelfImproveTick, ...]
    improved: bool
    total_gain: float


def _check(config: SelfImproveConfig) -> None:
    if not config.slots:
        raise ValueError("self_improve() got no slots")
    missing = [s for s in config.slots if s not in config.specs]
    if missing:
        raise ValueError(f"no spec for slot(s) {missing}")
    unknown = [s for s in config.slots if s not in SLOTS]
    if unknown:
        raise ValueError(f"not evolvable slot(s) {unknown}; choose from {SLOTS}")
    if not config.domains:
        raise ValueError("self_improve() got no domains")
    if config.ticks < 0:
        raise ValueError("ticks must be >= 0")
    if config.top_k < 1:
        raise ValueError("top_k must be >= 1")
    if set(config.seeds) & set(config.val_seeds):
        raise ValueError("seeds and val_seeds must be disjoint")
    for name, pair in config.domains.items():
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError(f"domain {name!r} must map to (train, val)")
        train, val = pair
        _require_disjoint(train, val, config.seeds, config.val_seeds)


def _gate(gates: AdmitFn, record: Any) -> Tuple[str, str, bool]:
    """Run the trust gate and return ``(verdict, reason, examined)``.

    ``examined`` is False when the gate consulted no ring (the default
    ``admit_record`` on a freshly archived record sees ``exam=None`` and no
    other status), so the report can tell a provisional pass from a reviewed
    one."""
    outcome = gates(record)
    if isinstance(outcome, AdmitVerdict):
        return outcome.verdict, outcome.reason, bool(outcome.rings)
    return str(outcome), "", True


def self_improve(
    store: ConsolidationStore,
    config: SelfImproveConfig,
    *,
    propose: Optional[Callable[[str, Any, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    meta_reward: Optional[MetaReward] = None,
    admit_fn: Optional[AdmitFn] = None,
    **evolve_kwargs: Any,
) -> SelfImproveReport:
    """Run the closed loop for ``config.ticks`` ticks.

    Each tick re-reads the archive, so it evolves the slots the evidence says
    are paying **now** -- not a schedule fixed up front -- and **compounds**:
    each target starts from the archive's best rule for that slot (via
    ``initial_state``, like ``continuous_meta_run``), so tick 2 refines tick 1's
    adoption instead of re-evolving the fresh seed.

    Adopted rules land in the same archive everything else reads
    (``_archived_rules``, ``exam_archive``, ``union``). A trust verdict of
    ``demote`` / ``flag`` **quarantines** the record with
    ``store.mark_exam(...)``, so the archive's own demotion/exclusion paths see
    it -- not just this report.

    ``propose`` or ``model`` is required (``meta_run``'s reflector).
    ``admit_fn`` overrides the trust gate (default
    :func:`~agentdescent.admit.admit_record`); it receives the archived record
    and returns an :class:`~agentdescent.admit.AdmitVerdict` or a status string.
    The default gate consults the record's own ``exam`` stamp and no other ring,
    so on a freshly archived record it passes **provisionally**; such ticks list
    the key under ``unexamined`` rather than counting as reviewed. Supply rings
    (or run ``exam_archive``) to make the gate bite.
    """
    if propose is None and model is None:
        raise ValueError("self_improve() needs propose= or model=")
    _check(config)
    gates = admit_fn or (lambda record: admit_record(record))
    prefix = config.run_id_prefix or f"self-improve-{int(time.time())}"

    out: list[SelfImproveTick] = []
    improved = False
    total_gain = 0.0
    for tick in range(config.ticks):
        snap = snapshot(store, config=config.config,
                        artifact_id=config.artifact_id)
        # rotate the fallback/exploration order by tick, so a tied or cold
        # archive moves through the slots instead of always picking a prefix
        targets = rank_slots(snap, k=config.top_k, floor=config.floor,
                             min_samples=config.min_samples,
                             allow=list(config.slots), rotate=tick)
        if not targets:
            targets = list(config.slots)[:config.top_k]
        budget_before = {
            layer.name: layer.budget
            for layer in layers_from_archive(store, artifact_id=config.artifact_id)
        }
        verdicts: Dict[str, str] = {}
        gains: Dict[str, float] = {}
        unexamined: list[str] = []
        for slot in targets:
            spec = config.specs[slot]
            # compound ticks: seed and validate against the archive's best rule
            # for this slot (like continuous_meta_run). The baseline is the
            # PRIOR rule, not the fresh seed, so a tick that finds nothing new
            # does not "re-adopt" last tick's rule and inflate the archive.
            # If the caller supplies ``initial_state`` in evolve_kwargs,
            # evolution starts there while validation still compares against
            # the prior (the baseline); pass ``baseline=`` to override that.
            from .verified_seed import verified_rules
            prior_rules = list(verified_rules(
                store, slot=slot, spec=spec, config=config.config,
                artifact_id=config.artifact_id,
                min_exam=config.min_exam).rules)
            prior = prior_rules[0] if prior_rules else None
            prior_value = ((prior.rendered or spec.render(spec.initial()))
                           if prior is not None else spec.render(spec.initial()))
            prior_state = dict(prior.state) if prior is not None else None
            for domain, pair in config.domains.items():
                train, val = pair
                seeded: Dict[str, Any] = dict(evolve_kwargs)
                if prior_state is not None and "initial_state" not in seeded:
                    seeded["initial_state"] = prior_state
                # meta_run keeps its two adoption guards (``mean_gain <=
                # min_gain`` and ``transfer_floor``); hand-rolling evolve +
                # validate here made the loop adopt zero-gain and overfit
                # rules that meta_run refuses.
                meta_report, result = meta_run(
                    train, val, slot=slot, spec=spec, propose=propose,
                    model=model, meta_reward=meta_reward, seeds=config.seeds,
                    val_seeds=config.val_seeds, min_gain=config.min_gain,
                    transfer_floor=config.transfer_floor,
                    baseline=prior_value, **seeded)
                key = f"{slot}:{domain}"
                verdicts[key] = "reject"
                gains[key] = float(meta_report.mean_gain)
                if meta_report.verdict != "adopt":
                    continue
                # the domain rides at the descriptor's top level, so records
                # from different domains are distinguishable and
                # problem_from_archive (which reads "slot"/"inner") ignores it
                descriptor = problem_descriptor(slot)
                descriptor["domain"] = domain
                record = archive_rule(
                    store, meta_report, result,
                    run_id=f"{prefix}-t{tick}-{slot}-{domain}", slot=slot,
                    spec=spec, config=config.config, problem=descriptor,
                    artifact_id=config.artifact_id)
                verdict, reason, examined = _gate(gates, record)
                if verdict != "pass":
                    # quarantine: stamp the record so the archive's own
                    # demotion/exclusion paths see it, not just this report
                    exam = "demoted" if verdict == "demote" else "flagged"
                    store.mark_exam(record.run_id, exam=exam,
                                    note=(reason or verdict)[:500])
                    verdicts[key] = verdict
                else:
                    verdicts[key] = "adopt"
                    improved = True
                    total_gain += float(meta_report.mean_gain)
                    if not examined:
                        unexamined.append(key)
        layers = layers_from_archive(store, artifact_id=config.artifact_id)
        if config.reallocation is not None:
            reallocated = config.reallocation.apply(layers)
        else:
            reallocated = portfolio_report(
                layers, floor=config.floor,
                min_samples=config.min_samples).reallocated
        out.append(SelfImproveTick(
            index=tick, targets=tuple(targets), verdicts=verdicts,
            gains=gains, budget_before=budget_before,
            budget_after=dict(reallocated),
            unexamined=tuple(unexamined),
            reason=f"evidence routed the tick to {', '.join(targets)}"))
    return SelfImproveReport(ticks=tuple(out), improved=improved,
                             total_gain=total_gain)
