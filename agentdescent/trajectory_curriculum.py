"""Trajectory-driven curriculum: the problems come from the run's own failures.

``curriculum_meta_run`` (#429) gives the meta loop a challenger that generates
frontier problems; the challenger's hardest and least-examined question is
*where the problems come from*. The trajectory ledger (#248) already records
every rollout -- including the ones that failed -- and every archived rule links
its evidence through ``ConsolidationRecord.trajectory_store``. This module is
that memory turned into curriculum: a :class:`MetaProblemChallenger` whose
``generate`` reads a rule's recorded failures and whose ``validate`` builds a
frontier problem out of the tasks the rule actually got wrong.

The guarantee of #429 is inherited unchanged: the challenger never touches the
final evaluation -- ``curriculum_meta_run`` keeps ``val_problems`` fixed and the
held-out split is computed from the seed tasks alone, so a problem mined from a
failure can only ever appear in the training frontier.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .consolidation import ConsolidationStore
from .evolution import Task
from .meta import Problem, evolve_problem
from .meta_curriculum import MetaChallengerConfig, MetaProblemChallenger
from .trajectory import PURPOSE_ROLLOUT, Trajectory, TrajectoryStore

__all__ = [
    "failed_rollouts",
    "problem_from_tasks",
    "trajectory_challenger",
]


def failed_rollouts(
    trajectory_root: str,
    *,
    artifact_id: Optional[str] = None,
    purpose: str = PURPOSE_ROLLOUT,
    failure_threshold: float = 0.5,
    max_failures: int = 8,
) -> List[Trajectory]:
    """The recorded rollouts that failed, up to ``max_failures``.

    Reads the :class:`~agentdescent.trajectory.TrajectoryStore` at
    ``trajectory_root`` and returns the ``purpose`` records whose ``reward`` is
    below ``failure_threshold`` (the threshold itself does not count), in store
    iteration order, up to ``max_failures``. A corrupt record is skipped by the
    store (diagnostics data must not hide the rest); an empty or missing store
    yields an empty list, never an exception."""
    store = TrajectoryStore(trajectory_root)
    if max_failures < 1:
        return []
    out: List[Trajectory] = []
    for record in store.iter_records(artifact_id=artifact_id, purpose=purpose):
        if float(record.reward) < failure_threshold:
            out.append(record)
            if len(out) >= max_failures:
                break
    return out


def problem_from_tasks(
    tasks: Sequence[Task],
    *,
    slot: str,
    run: Callable[[str, Task], str],
    reward: Callable[[Task, str], float],
    propose: Callable[[str, Task, str, float], Optional[str]],
    strategy: Any,
    **inner: Any,
) -> Problem:
    """A runnable inner :class:`~agentdescent.meta.Problem` over exactly these tasks."""
    return evolve_problem(
        list(tasks), reward, slot=slot, run=run, propose=propose,
        strategy=strategy, **inner)


def trajectory_challenger(
    archive_store: ConsolidationStore,
    *,
    tasks_for: Callable[[Sequence[str]], Sequence[Task]],
    build_problem: Callable[[Sequence[Task]], Problem],
    slot: str,
    artifact_id: str = "meta-rule",
    failure_threshold: float = 0.5,
    max_failures: int = 8,
    exclude_task_ids: Optional[Sequence[str]] = None,
    config: Optional[MetaChallengerConfig] = None,
) -> MetaProblemChallenger:
    """A :class:`MetaProblemChallenger` that mines the archive's recorded failures.

    The challenger's generation spec is a **run_id**: ``generate`` reads that
    rule's ``trajectory_store``, takes the failed rollouts, and returns a config
    carrying the failed tasks; ``validate`` builds a frontier problem over them
    via ``build_problem`` (raising -- and so counting invalid -- when there is
    nothing to mine); ``propose`` advances to the next archived rule that
    recorded trajectories, so the frontier turns across the archive's failures
    instead of one rule's.

    ``exclude_task_ids`` are never mined into the training frontier: the caller
    passes the task ids its held-out problems evaluate, so a task the inner run
    touched and that also sits in ``val_problems`` cannot contaminate the
    validation measurement. A batch that filters down to nothing raises (and so
    counts invalid) rather than training on the validation set.

    Parameters
    ----------
    archive_store:
        The archive whose rule records link trajectory stores.
    tasks_for:
        Maps the failed ``task_id``s to the ``Task`` objects they mean -- the
        caller's data, read through the trajectory record.
    build_problem:
        ``tasks -> Problem``; :func:`problem_from_tasks` is the natural choice
        for an ``evolve_problem``-shaped domain.
    slot:
        The decision slot the inner problems evolve (for the problem descriptor
        and for ``problem_from_tasks``).
    artifact_id:
        The archive ``artifact_id`` of rule records (as ``continuous_meta_run``).
    failure_threshold, max_failures:
        What counts as a failed rollout and how many each frontier batch mines.
    exclude_task_ids:
        Task ids that must never reach ``build_problem`` (typically the held-out
        tasks' ids). ``None`` excludes nothing.
    config:
        The challenger's frontier dials (``generated_per_round``,
        ``frontier_share`` ...).
    """
    runs = _rule_runs_with_trajectories(archive_store, artifact_id=artifact_id)
    excluded = set(exclude_task_ids or ())

    def generate(spec: str, seed: int) -> Dict[str, Any]:
        root = _trajectory_root_for(archive_store, spec)
        if root is None:
            raise ValueError(f"rule {spec!r} has no trajectory store to mine")
        records = failed_rollouts(root, failure_threshold=failure_threshold,
                                  max_failures=max_failures)
        if not records:
            raise ValueError(f"no failed rollouts under {spec!r} below "
                             f"{failure_threshold}")
        task_ids = list(dict.fromkeys(r.task_id for r in records))
        task_ids = [t for t in task_ids if t not in excluded]
        if not task_ids:
            raise ValueError(
                f"all failed rollouts under {spec!r} carry excluded task ids; "
                f"nothing left to mine outside the validation set")
        return {"run_id": spec, "tasks": list(tasks_for(task_ids))}

    def validate(config: Dict[str, Any]) -> Problem:
        tasks = config.get("tasks") or []
        if not tasks:
            raise ValueError("mined config carries no tasks")
        return build_problem(tasks)

    def propose(spec: str, report: Dict[str, float]) -> str:
        if spec in runs:
            idx = runs.index(spec)
            return runs[(idx + 1) % len(runs)]
        return runs[0] if runs else spec

    return MetaProblemChallenger(
        generate=generate, validate=validate, propose=propose,
        spec=runs[0] if runs else "",
        config=config or MetaChallengerConfig())


def _rule_runs_with_trajectories(
    store: ConsolidationStore, *, artifact_id: str,
) -> List[str]:
    """Archived rule run_ids that recorded a trajectory store, in archive order."""
    out = []
    for rec in store.iter_archive():
        if rec.artifact_id != artifact_id:
            continue
        if rec.trajectory_store:
            out.append(rec.run_id)
    return out


def _trajectory_root_for(store: ConsolidationStore, run_id: str) -> Optional[str]:
    rec = store.find(run_id)
    if rec is None:
        return None
    return rec.trajectory_store
