"""Self-exam: the archive's adoptions are provisional until the oracle says so.

``archive_rule`` writes an adopted rule into the archive, and ``_archived_rules``
keeps ranking it and seeding the next search from it. But an adoption is a
verdict the meta loop reached on its own numbers -- even the audited numbers of
:mod:`agentdescent.meta_audit` are, until the oracle resolves the sample. The
self-exam is the rung that makes the verdict **provisional**: it walks the
archive's adopted rules, re-validates each one against the problems its own
descriptor describes using an independent oracle, and stamps the outcome on the
record:

* ``passed`` -- the oracle agrees the rule transfers (its gain on the rebuilt
  problems clears ``min_gain``);
* ``flagged`` -- the oracle disagrees; the rule stays seedable, and the flag
  count climbs;
* ``demoted`` -- a rule flagged ``max_flags`` times in a row is excluded from
  future seeding by ``_archived_rules``, so a rule that looked good to its own
  reward and then failed truth twice stops steering the search.

Adoptions are thus provisional until examined, and the archive is
self-correcting: what the loop believed is checked, and what fails the check is
retired. ``ConsolidationRecord.exam`` / ``exam_flags`` / ``exam_note`` carry the
stamp (optional, backwards compatible -- an archive written before the exam
existed reads back as ``None`` and stays seedable until examined).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .consolidation import ConsolidationStore
from .evolution import Task
from .meta import (
    Problem, SlotSpec, _archived_rules, _mean_gain, meta_validate,
    problem_from_archive,
)

__all__ = ["ExaminedRule", "MetaExamReport", "exam_archive"]


@dataclass(frozen=True)
class ExaminedRule:
    """One archived rule's self-exam: what the oracle said about it."""

    run_id: str
    oracle_gain: float
    verdict: str            # "passed" / "flagged" / "demoted"
    reason: str


@dataclass(frozen=True)
class MetaExamReport:
    """What an :func:`exam_archive` pass did, and why.

    ``examined`` is one row per non-demoted rule examined; ``passed`` /
    ``flagged`` / ``demoted`` are the counts. ``demoted`` rules are no longer
    returned by ``_archived_rules``, so seeding and population selection skip
    them from here on.
    """

    slot: str
    examined: Tuple[ExaminedRule, ...]
    passed: int
    flagged: int
    demoted: int
    reason: str


def exam_archive(
    store: ConsolidationStore,
    *,
    slot: str,
    spec: SlotSpec,
    oracle: Callable[[Any], float],
    tasks_for: Callable[[Any], Sequence[Task]],
    run: Callable[[str, Task], str],
    reward: Callable[[Task, str], float],
    propose: Callable[[str, Task, str, float], Optional[str]],
    strategy: Any,
    config: Optional[Dict[str, Any]] = None,
    artifact_id: str = "meta-rule",
    seeds: Sequence[int] = (0, 1),
    min_gain: float = 0.0,
    max_flags: int = 2,
    **inner_defaults: Any,
) -> MetaExamReport:
    """Re-validate the archive's adopted rules against an independent oracle.

    For every non-demoted rule record for ``slot``/``spec``/``config``, the rule
    is re-validated on the problems its own ``problem`` descriptor describes
    (rebuilt by :func:`~agentdescent.meta.problem_from_archive`) using
    ``oracle`` as the meta reward, at ``seeds``. The oracle gain is compared to
    ``min_gain`` and the record is stamped via
    :meth:`~agentdescent.consolidation.ConsolidationStore.mark_exam`:

    * ``oracle_gain >= min_gain`` -> ``passed`` (flags reset);
    * ``oracle_gain < min_gain`` -> ``flagged`` (flags + 1);
    * ``flags >= max_flags`` -> ``demoted``, excluded from future seeding.

    Parameters
    ----------
    store:
        The archive whose rule records are examined.
    slot, spec:
        As :func:`~agentdescent.meta.meta_run` -- which decision field and how
        values compile.
    oracle:
        A :class:`~agentdescent.meta.MetaReward` -- ``MetaOutcome -> float`` that
        scores an inner run with truth. The same oracle a ``meta_run_audited``
        resolves its sample against.
    tasks_for, run, reward, propose, strategy:
        The inner domain functions, passed to :func:`problem_from_archive` to
        rebuild each rule's problems.
    config, artifact_id:
        As :func:`~agentdescent.meta.continuous_meta_run`.
    seeds:
        The seeds the re-validation runs the rebuilt problems at.
    min_gain:
        The oracle gain a rule must clear to pass. ``0.0`` means "must not
        regress under truth".
    max_flags:
        Consecutive failed exams before a rule is demoted.
    **inner_defaults:
        Defaults for the rebuilt problems' inner ``evolve()`` settings, filled
        in where the archived descriptor omits them (as
        :func:`problem_from_archive` accepts).

    Returns a :class:`MetaExamReport`; the archive now carries the stamps.
    """
    if not (isinstance(max_flags, int) and max_flags >= 1):
        raise ValueError(f"max_flags must be an int >= 1, got {max_flags!r}")
    rules = _archived_rules(store, slot=slot, spec=spec, config=config,
                            artifact_id=artifact_id)
    seed_value = spec.render(spec.initial())
    examined: list = []
    for rec in rules:
        rule_value = (spec.render(dict(rec.state or {})) if rec.state
                      else rec.rendered or seed_value)
        problem = problem_from_archive(
            store, rec.run_id, tasks_for=tasks_for, spec=spec, run=run,
            reward=reward, propose=propose, strategy=strategy,
            **inner_defaults)
        validation = meta_validate(
            spec, seed_value, rule_value, {rec.run_id: problem},
            seeds=seeds, meta_reward=oracle)
        oracle_gain = _mean_gain(validation)
        if oracle_gain < min_gain:
            flags = int(rec.exam_flags or 0) + 1
            if flags >= max_flags:
                verdict, note = "demoted", (
                    f"oracle gain {oracle_gain:+.3f} below min_gain "
                    f"{min_gain:+.3f} for {flags} consecutive exams; retired")
            else:
                verdict, note = "flagged", (
                    f"oracle gain {oracle_gain:+.3f} below min_gain "
                    f"{min_gain:+.3f} (flag {flags}/{max_flags})")
        else:
            verdict, note = "passed", (
                f"oracle gain {oracle_gain:+.3f} at or above min_gain "
                f"{min_gain:+.3f}")
            flags = 0
        store.mark_exam(rec.run_id, exam=verdict, flags=flags, note=note)
        examined.append(ExaminedRule(run_id=rec.run_id, oracle_gain=oracle_gain,
                                     verdict=verdict, reason=note))

    passed = sum(1 for e in examined if e.verdict == "passed")
    flagged = sum(1 for e in examined if e.verdict == "flagged")
    demoted = sum(1 for e in examined if e.verdict == "demoted")
    reason = (f"{len(examined)} rule(s) examined: {passed} passed, "
              f"{flagged} flagged, {demoted} demoted")
    return MetaExamReport(slot=slot, examined=tuple(examined), passed=passed,
                          flagged=flagged, demoted=demoted, reason=reason)
