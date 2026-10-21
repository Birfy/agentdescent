"""Trajectory-driven curriculum has to be right where memory becomes the exam.

The meta loop's challenger generates frontier problems; this challenger mines
them from the run's own recorded failures (#248 trajectory ledger, linked by
#334's `trajectory_store`). The fixtures pin the surfaces that make this
honest: failures are read as evidence (successes and corrupt records excluded),
a mined failure becomes a runnable Problem, a rule with no trajectories is
counted invalid rather than crashed over, and the whole thing plugs into
`curriculum_meta_run` without ever touching its fixed validation set.
"""

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    MetaChallengerConfig,
    MetaOutcome,
    ParamSlot,
    SingleSlot,
    Task,
    curriculum_meta_run,
    failed_rollouts,
    problem_descriptor,
    problem_from_tasks,
    trajectory_challenger,
)
from agentdescent.sampling import DifficultyWeighted
from agentdescent.trajectory import PURPOSE_ROLLOUT, Trajectory, TrajectoryStore


# -- the deterministic capability domain ----------------------------------------


def _cap(rendered):
    import re
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, task):
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return "cap:0"
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return f"cap:{a}"
    return None


def _task_by_id(task_id):
    a = int(task_id[1:]) if len(task_id) > 1 else 0
    return Task(id=task_id, prompt=f"item {task_id}",
                meta={"gold": str(2 * a), "a": a})


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


CONFIG = {"dataset": "cap-arithmetic", "scorer": "exact"}
INNER = {"rounds": 5, "n_workers": 1, "target_reward": 1.0}


def _helpful(value, seed):
    if getattr(value, "c", 0.2) > 0.2:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    return MetaOutcome(curve=[0.2, 0.3], final=0.3)


def _helpful_val(value, seed):
    return _helpful(value, seed)


# -- fixtures -------------------------------------------------------------------


def _record_failures(root, task_ids, *, reward=0.0):
    store = TrajectoryStore(root)
    for tid in task_ids:
        store.record(Trajectory(
            artifact_id="policy-slot", task_id=tid, env_fingerprint="",
            rendered="cap:2", purpose=PURPOSE_ROLLOUT, output="?",
            reward=reward))
    return store


def _archive_rule(store, run_id, trajectory_root, *, c=0.3):
    store.archive(
        EvolutionResult(state={"c": f"{c:.1f}"},
                        rendered=f"# Policy parameters\nc: {c:.1f}",
                        final_reward=0.8, history=[], ledger_log=[]),
        run_id=run_id, strategy="ParamSlot", config=CONFIG,
        problem=problem_descriptor("task_sampler", **INNER),
        artifact_id="meta-rule", trajectory_store=trajectory_root)


def _challenger(store, *, c=0.3, root=None, **kw):
    return trajectory_challenger(
        store,
        tasks_for=lambda ids: [_task_by_id(i) for i in ids],
        build_problem=lambda tasks: problem_from_tasks(
            tasks, slot="task_sampler", run=_run, reward=_reward,
            propose=_propose, strategy=SingleSlot(initial_value="cap:0"),
            **INNER),
        slot="task_sampler", **kw)


# -- failure mining -------------------------------------------------------------


def test_failed_rollouts_read_failures_not_successes(tmp_path):
    root = str(tmp_path / "traj")
    store = TrajectoryStore(root)
    store.record(Trajectory(artifact_id="policy-slot", task_id="t8",
                            env_fingerprint="", rendered="cap:2",
                            purpose=PURPOSE_ROLLOUT, output="?", reward=0.0))
    store.record(Trajectory(artifact_id="policy-slot", task_id="t2",
                            env_fingerprint="", rendered="cap:8",
                            purpose=PURPOSE_ROLLOUT, output="4", reward=1.0))
    found = failed_rollouts(root, failure_threshold=0.5, max_failures=8)
    assert [r.task_id for r in found] == ["t8"]
    capped = failed_rollouts(root, failure_threshold=0.5, max_failures=0)
    assert capped == []


def test_a_mined_failure_becomes_a_runnable_problem(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    root = str(tmp_path / "traj")
    _record_failures(root, ["t5", "t6", "t7", "t8"])
    _archive_rule(archive, "run-1", root)
    challenger = _challenger(archive)
    config = challenger.generate(challenger.spec, seed=0)
    assert sorted(t.id for t in config["tasks"]) == ["t5", "t6", "t7", "t8"]
    problem = challenger.validate(config)
    outcome = problem(_sampler_slot().compile("# Policy parameters\nc: 0.4"),
                      seed=0)
    assert isinstance(outcome, MetaOutcome)
    assert len(outcome.curve) > 0


def test_a_rule_with_no_trajectories_is_counted_invalid_not_crashed(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(archive, "run-1", None)     # no trajectory_store
    challenger = _challenger(archive)
    with pytest.raises(ValueError):
        challenger.generate(challenger.spec, seed=0)


def test_a_validation_task_id_never_reaches_build_problem(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    root = str(tmp_path / "traj")
    # t6 sits in the held-out set but was also recorded as a failed rollout
    _record_failures(root, ["t5", "t6", "t7"])
    _archive_rule(archive, "run-1", root)
    reached: list = []

    def build_problem(tasks):
        reached.extend(t.id for t in tasks)
        return problem_from_tasks(tasks, slot="task_sampler", run=_run,
                                  reward=_reward, propose=_propose,
                                  strategy=SingleSlot(initial_value="cap:0"),
                                  **INNER)

    challenger = trajectory_challenger(
        archive, tasks_for=lambda ids: [_task_by_id(i) for i in ids],
        build_problem=build_problem, slot="task_sampler",
        exclude_task_ids=["t6"])
    config = challenger.generate(challenger.spec, seed=0)
    assert sorted(t.id for t in config["tasks"]) == ["t5", "t7"]
    challenger.validate(config)
    assert reached == ["t5", "t7"]          # t6 never reached build_problem
    # an all-excluded batch counts invalid rather than training on the val set
    challenger = trajectory_challenger(
        archive, tasks_for=lambda ids: [_task_by_id(i) for i in ids],
        build_problem=build_problem, slot="task_sampler",
        exclude_task_ids=["t5", "t6", "t7"])
    with pytest.raises(ValueError):
        challenger.generate(challenger.spec, seed=0)


# -- integration with curriculum_meta_run ---------------------------------------


def test_curriculum_run_mines_failures_and_keeps_validation_fixed(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    root = str(tmp_path / "traj")
    _record_failures(root, ["t5", "t6", "t7"])
    _archive_rule(archive, "run-1", root)
    challenger = _challenger(archive,
                             config=MetaChallengerConfig(
                                 generated_per_round=2, frontier_share=0.4))
    report, result = curriculum_meta_run(
        store=archive, slot="task_sampler", spec=_sampler_slot(),
        challenger=challenger, propose=_nudge_up(),
        problems={"ta": _helpful, "tb": _helpful},
        val_problems={"va": _helpful_val}, config=CONFIG,
        seeds=(0, 1, 2, 3), val_seeds=(10, 11), rounds=3, n_workers=1,
        repo_dir=str(tmp_path / "repos"))
    # the frontier was generated from the mined failures
    assert report["frontier"]["generated_total"] >= 1
    # the validation set never contains a frontier problem
    assert set(report["validation"].keys()) == {"va"}
    assert result.meta["verdict"] == report["verdict"]


def _nudge_up():
    import re
    def propose(rendered, task, output, reward):
        m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
        c = float(m.group(1)) if m else 0.2
        if reward < 0.7:
            return f"c: {min(1.0, c + 0.1):.2f}"
        return f"c: {c:.2f}"
    return propose
