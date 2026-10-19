"""Meta-population: the archive seeds a population, the population promotes.

``continuous_meta_run`` seeds one session from the archive's best rule; a
population is what makes "continuing improvement" *parallel* -- the archive's
top rules each seed an island, every island searches from a different starting
rule, and the island whose evidence is best is promoted back to the archive.
Lineages that accumulate are what metaproductivity means; this is the
population version of that claim, in the same shape as the rest of the meta
surface:

* **The islands are sessions.** Each island is ``meta_evolve`` on the shared
  train problems, seeded with ``initial_state`` from a distinct archived rule
  -- so the population explores different directions instead of one.
* **The validation is shared and isolated.** Every island validates the same
  ``val_problems`` at ``val_seeds``, disjoint from training by
  ``_require_disjoint``; the per-island gain is against that island's own
  prior, so "a better rule" and "a better starting point" are the same number.
* **Promotion is evidence-based and fail-closed.** Only an island whose verdict
  is ``adopt`` (mean gain on unseen problems positive and the transfer not
  negative) can be promoted; the best such island's rule is written back via
  ``archive_rule``. An island that errored is reported, never promoted, and does
  not stop the others.
* **Nothing new touches the archive's shape.** ``ConsolidationStore`` records
  stay exactly as ``archive_rule`` writes them; the population is a caller-side
  loop over the same durable, atomic, idempotent store.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .consolidation import ConsolidationStore
from .evolution import EvolutionResult, Task
from .meta import (
    MetaReward, MetaRunReport, Problem, SlotSpec, _archived_rules,
    _mean_gain, _require_disjoint, _session_problems, _transfer_from,
    archive_rule, auc, meta_evolve, meta_validate, problem_descriptor,
)

__all__ = ["PopulationIsland", "MetaPopulationReport", "meta_population_run"]


@dataclass(frozen=True)
class PopulationIsland:
    """One island of a :func:`meta_population_run`: a rule's search, and how it went.

    ``seeded_from`` names the archived rule this island started from. ``error``
    carries a failed session's message; an errored island is never promoted.
    """

    index: int
    seeded_from: Optional[str]
    value: str
    mean_gain: float
    transfer: Optional[float]
    verdict: str
    reason: str
    error: Optional[str] = None


@dataclass(frozen=True)
class MetaPopulationReport:
    """What a :func:`meta_population_run` did, and why.

    ``islands`` is one row per starting rule, best seed first; ``promoted_from``
    is the index of the island whose rule was written back (``None`` when no
    island cleared the adopt bar -- the archive is left unchanged, which is a
    result, not an error).
    """

    slot: str
    islands: Tuple[PopulationIsland, ...]
    promoted_from: Optional[int]
    promoted_run_id: Optional[str]
    archive_count: int
    reason: str


def _island_session(
    slot: str,
    spec: SlotSpec,
    propose: Callable[[str, Task, str, float], Optional[str]],
    train: Mapping[str, Problem],
    val: Mapping[str, Problem],
    *,
    prior_state: Dict[str, str],
    prior_value: str,
    seeded_from: Optional[str],
    index: int,
    meta_reward: MetaReward,
    seeds: Sequence[int],
    val_seeds: Sequence[int],
    model: Optional[Any] = None,
    **evolve_kwargs: Any,
) -> Tuple[PopulationIsland, EvolutionResult]:
    """One island: evolve from ``prior_state``, validate against the shared sets.

    Returns ``(row, result)``; ``result`` is what :func:`archive_rule` needs if
    the island is promoted. An exception inside the session becomes an
    ``error`` row with a ``None`` result -- the run goes on, but that island can
    never be promoted.
    """
    try:
        result = meta_evolve(
            train, slot=slot, spec=spec, propose=propose, model=model,
            meta_reward=meta_reward, seeds=seeds,
            initial_state=dict(prior_state), **evolve_kwargs)
        value = result.rendered or prior_value
        train_validation = meta_validate(
            spec, prior_value, value, train, seeds=seeds, meta_reward=meta_reward)
        validation = meta_validate(
            spec, prior_value, value, val, seeds=val_seeds, meta_reward=meta_reward)
        mean_gain = _mean_gain(validation)
        train_gain = _mean_gain(train_validation)
        transfer = _transfer_from(train_gain, mean_gain)
    except Exception as error:  # noqa: BLE001 - one island failing must not fail the run
        return PopulationIsland(
            index=index, seeded_from=seeded_from, value=prior_value,
            mean_gain=0.0, transfer=None, verdict="error",
            reason=f"{type(error).__name__}: {error}", error=str(error)[:300]), \
            None  # type: ignore[return-value]
    if mean_gain <= 0.0:
        verdict, reason = "reject", (
            f"gained {mean_gain:+.3f} on unseen problems, not a positive gain")
    elif transfer is not None and transfer < 0.0:
        verdict, reason = "reject", (
            f"transfer ratio {transfer:.3f} is negative -- a fit to the "
            "problems it was evolved on")
    else:
        verdict, reason = "adopt", (
            f"mean gain {mean_gain:+.3f} on unseen problems, "
            f"transfer {transfer if transfer is None else round(transfer, 3)}")
    return PopulationIsland(
        index=index, seeded_from=seeded_from, value=value,
        mean_gain=mean_gain, transfer=transfer, verdict=verdict, reason=reason), \
        result


def meta_population_run(
    store: ConsolidationStore,
    slot: str,
    spec: SlotSpec,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    *,
    tasks_for: Callable[[Any], Sequence[Task]],
    run: Callable[[str, Task], str],
    reward: Callable[[Task, str], float],
    inner_propose: Callable[[str, Task, str, float], Optional[str]],
    inner_strategy: Any,
    config: Optional[Dict[str, Any]] = None,
    islands: int = 3,
    problems: Optional[Mapping[str, Problem]] = None,
    val_problems: Optional[Mapping[str, Problem]] = None,
    artifact_id: str = "meta-rule",
    train_runs: int = 2,
    val_runs: int = 1,
    inner: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0, 1),
    val_seeds: Sequence[int] = (10, 11),
    repo_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    **evolve_kwargs: Any,
) -> Tuple[MetaPopulationReport, EvolutionResult]:
    """Run a population of meta sessions from the archive's best rules, promote the winner.

    The archive's rule records for this slot, spec and configuration are ranked
    as ``continuous_meta_run`` ranks them (same configuration fingerprint first,
    then measured reward); the top ``islands`` distinct rules each seed a
    session. Every session trains on the same ``problems`` (archive-derived or
    the caller's fallback) and validates on the same ``val_problems`` at
    ``val_seeds`` disjoint from the training seeds, so the islands differ only
    in *where they started*. The best island whose verdict is ``adopt`` is
    written back via :func:`archive_rule`, and the next call sees its rule at
    the top of the archive.

    Parameters
    ----------
    store:
        The :class:`~agentdescent.consolidation.ConsolidationStore` that makes
        the loop continuous. It must already hold at least one rule for this
        slot/spec/config -- a population needs something to diversify from;
        call :func:`~agentdescent.meta.continuous_meta_run` once first.
    slot, spec, propose, model:
        As :func:`~agentdescent.meta.meta_run`.
    tasks_for, run, reward, inner_propose, inner_strategy:
        The inner domain functions, as :func:`~agentdescent.meta.continuous_meta_run`.
    config:
        This session's configuration, used to rank archived rules and written
        back with the promoted rule.
    islands:
        How many starting rules to seed. At most ``islands`` islands run; fewer
        when the archive holds fewer rules.
    problems, val_problems:
        The train/validation problems (archive-derived first, caller fallback
        when the archive cannot fill a set), as ``continuous_meta_run``.
    artifact_id, train_runs, val_runs, inner:
        As :func:`~agentdescent.meta.continuous_meta_run`.
    meta_reward, seeds, val_seeds:
        As :func:`~agentdescent.meta.meta_run`.
    run_id:
        The archived ``run_id`` of the promoted rule; a timestamp id when
        omitted.
    repo_dir:
        Base directory for each island's ledger. Every island runs in its own
        **fresh** repo -- a shared ``repo_path`` would make the engine resume
        the ledger and silently ignore the islands' distinct ``initial_state``,
        turning a population into a run of clones. Pass ``repo_dir`` to keep the
        per-island repos in one place; the default is the system tempdir.
    **evolve_kwargs:
        Everything else :func:`meta_evolve` takes, except ``repo_path`` (each
        island gets a fresh one under ``repo_dir``) and ``strategy``/``run``/
        ``reward``/``agent``.

    Returns ``(report, result)``; ``report.islands`` holds the per-island rows,
    ``report.promoted_from`` the index whose rule was archived, and ``result``
    the promoted island's ``EvolutionResult`` (or the first island's when
    nothing was promoted). ``result.meta`` carries the report.
    """
    if not (isinstance(islands, int) and islands >= 1):
        raise ValueError(f"islands must be an int >= 1, got {islands!r}")
    for taken in ("strategy", "run", "reward", "agent", "initial_state",
                  "repo_path"):
        if taken in evolve_kwargs:
            raise TypeError(
                f"meta_population_run() sets {taken}= itself -- pass repo_dir= "
                "for the per-island repos' base directory")
    if propose is None and model is None:
        raise ValueError("meta_population_run() needs propose= or model=")
    if repo_dir is not None:
        os.makedirs(repo_dir, exist_ok=True)
    inner = dict(inner or {})
    rules = _archived_rules(store, slot=slot, spec=spec, config=config,
                            artifact_id=artifact_id)
    if not rules:
        raise ValueError(
            "meta_population_run() found no archived rule for this slot/spec/"
            "config -- a population needs distinct starting rules to diversify "
            "from; run continuous_meta_run() once first")
    train, val = _session_problems(
        store, rules, tasks_for=tasks_for, spec=spec, run=run, reward=reward,
        propose=inner_propose, strategy=inner_strategy,
        problems=problems, val_problems=val_problems,
        train_runs=train_runs, val_runs=val_runs, inner=inner)
    _require_disjoint(train, val, seeds, val_seeds)
    score = meta_reward or auc

    rows: List[PopulationIsland] = []
    results: List[EvolutionResult] = []
    winners: List[Tuple[float, Optional[float], int, PopulationIsland, EvolutionResult]] = []
    for index, rule in enumerate(rules[:islands]):
        prior_state = dict(rule.state or {})
        prior_value = (spec.render(prior_state) if prior_state
                       else spec.render(spec.initial()))
        island, result = _island_session(
            slot, spec, propose, train, val,
            prior_state=prior_state, prior_value=prior_value,
            seeded_from=rule.run_id, index=index, meta_reward=score,
            seeds=seeds, val_seeds=val_seeds, model=model,
            repo_path=tempfile.mkdtemp(prefix="agentdescent-meta-pop-",
                                       dir=repo_dir),
            **evolve_kwargs)
        rows.append(island)
        if result is not None:
            results.append(result)
        if island.verdict == "adopt":
            winners.append((island.mean_gain, island.transfer, index,
                            island, result))

    promoted_from, promoted_run_id, chosen = None, None, None
    if winners:
        # best mean gain first, then best transfer; lowest index breaks ties
        winners.sort(key=lambda w: (w[0], w[1] if w[1] is not None else -1.0,
                                    -w[2]), reverse=True)
        _, _, idx, island, result = winners[0]
        chosen = island
        rid = run_id or f"meta-pop-{slot}-{int(time.time())}"
        seed_value = _seed_value_of(spec, rules, idx)
        report = MetaRunReport(
            slot=slot, seed_value=seed_value, value=island.value,
            validation={}, mean_gain=island.mean_gain, transfer=island.transfer,
            verdict="adopt", reason=island.reason)
        archive_rule(store, report, result, run_id=rid, slot=slot, spec=spec,
                     config=config, problem=problem_descriptor(slot, **inner),
                     artifact_id=artifact_id)
        promoted_from, promoted_run_id = idx, rid

    reason = (f"promoted island {promoted_from} (seeded from {chosen.seeded_from})"
              if chosen is not None else
              "no island cleared the adopt bar; archive unchanged")
    report = MetaPopulationReport(
        slot=slot, islands=tuple(rows), promoted_from=promoted_from,
        promoted_run_id=promoted_run_id, archive_count=store.count(),
        reason=reason)
    returned = (winners[0][4] if winners else
                (results[0] if results else None))
    if returned is not None:
        try:
            returned.meta = _report_dict(report)
        except AttributeError:
            pass
    return report, returned  # type: ignore[return-value]


def _seed_value_of(spec: SlotSpec, rules: Sequence[Any], index: int) -> str:
    rule = rules[index]
    state = dict(rule.state or {})
    return spec.render(state) if state else spec.render(spec.initial())


def _report_dict(report: MetaPopulationReport) -> Dict[str, Any]:
    return {
        "slot": report.slot,
        "islands": [{
            "index": i.index, "seeded_from": i.seeded_from, "value": i.value,
            "mean_gain": i.mean_gain, "transfer": i.transfer,
            "verdict": i.verdict, "reason": i.reason, "error": i.error,
        } for i in report.islands],
        "promoted_from": report.promoted_from,
        "promoted_run_id": report.promoted_run_id,
        "archive_count": report.archive_count,
        "reason": report.reason,
    }
