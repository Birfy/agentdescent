"""The meta-evolution product surface has to be right in the places a two-level
loop is dangerous.

meta_run() evolves a decision slot of the inner search, then judges the result
on problems the outer loop never saw. The fixtures pin: the report carries the
compiled value and a verdict, the validation problems are disjoint from the
training set, a value that helps unseen problems is adopted, one that
regresses is rejected, and the report rides the run result so a CLI or plugin
can show the decision.
"""

import re

import pytest

from agentdescent import (
    MetaRunReport,
    ParamSlot,
    SingleSlot,
    Task,
    auc,
    evolve,
    evolve_problem,
    meta_run,
)
from agentdescent.meta import meta_validate, transfer_ratio
from agentdescent.sampling import DifficultyWeighted


# -- a deterministic capability domain -------------------------------------------


def _cap(rendered):
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, task):
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return "cap:0"
    if task.meta.get("a", 0) > _cap(rendered):
        return f"cap:{_cap(rendered) + 1}"
    return None


def _tasks(prefix="s", n=8):
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def _sampler_slot():
    # the evolved slot: the UCB exploration constant of DifficultyWeighted
    return ParamSlot(
        factory=lambda c: DifficultyWeighted(c=c),
        params={"c": 0.2},
        bounds={"c": (0.05, 1.0)})


def _inner_problem(tasks, *, rounds=5):
    return evolve_problem(
        tasks, _reward, slot="task_sampler", run=_run, propose=_propose,
        strategy=SingleSlot(initial_value="cap:0"),
        rounds=rounds, n_workers=1, target_reward=1.0)


def _deterministic_propose(rendered, task, output, reward):
    """An offline hill-climb on the exploration constant: nudge up on a low
    meta-reward (hard tasks carry the signal), keep it otherwise."""
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    c = float(m.group(1)) if m else 0.2
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


# -- the product surface ---------------------------------------------------------


def test_meta_run_runs_and_reports(tmp_path):
    train = {"a": _inner_problem(_tasks(prefix="a")),
             "b": _inner_problem(_tasks(prefix="b"))}
    val = {"v": _inner_problem(_tasks(prefix="z"))}
    report, result = meta_run(
        train, val, slot="task_sampler", spec=_sampler_slot(),
        propose=_deterministic_propose, seeds=(0, 1), val_seeds=(10, 11),
        rounds=1, n_workers=1, repo_path=str(tmp_path / "repo"))
    assert isinstance(report, MetaRunReport)
    assert report.slot == "task_sampler"
    assert report.value
    assert report.seed_value
    assert set(report.validation.keys()) == {"v"}        # validation only
    assert isinstance(report.mean_gain, float)
    assert report.verdict in ("adopt", "reject")
    assert report.reason
    # the report rides the result so a CLI can show the decision
    assert result.meta == {
        "slot": report.slot, "seed_value": report.seed_value,
        "value": report.value, "validation": report.validation,
        "mean_gain": report.mean_gain, "transfer": report.transfer,
        "verdict": report.verdict, "reason": report.reason}
    # the value compiles to a usable sampler policy
    policy = _sampler_slot().compile(report.value)
    assert hasattr(policy, "pick")


def test_meta_run_requires_validation_problems():
    with pytest.raises(ValueError):
        meta_run({"t1": _inner_problem(_tasks())}, {}, slot="task_sampler",
                 spec=_sampler_slot(), propose=_deterministic_propose)


def test_the_validation_is_isolated_from_training():
    train = {"a": _inner_problem(_tasks(prefix="a")),
             "b": _inner_problem(_tasks(prefix="b")),
             "c": _inner_problem(_tasks(prefix="c")),
             "d": _inner_problem(_tasks(prefix="d"))}
    val = {"e": _inner_problem(_tasks(prefix="e")),
           "f": _inner_problem(_tasks(prefix="f"))}
    report, _ = meta_run(
        train, val, slot="task_sampler", spec=_sampler_slot(),
        propose=_deterministic_propose, seeds=(0, 1), val_seeds=(10, 11),
        rounds=1, n_workers=1)
    # the validation problems are disjoint from the training ones by name
    assert set(report.validation.keys()) == {"e", "f"}
    assert not (set(report.validation.keys()) & set(train.keys()))


def test_adopt_and_reject_verdicts():
    # a value that clearly helps unseen problems is adopted; one that regresses
    # is rejected. Drive the verdict by constructing the report decision path
    # directly through meta_run's helpers on a controlled outcome set.
    from agentdescent.meta import MetaOutcome

    slot = _sampler_slot()
    def helping(value, seed):
        # the evolved value (c > 0.2, nudged up by the offline proposer) beats
        # the seed (c == 0.2)
        if getattr(value, "c", 0.2) > 0.2:
            return MetaOutcome(curve=[0.5, 1.0], final=1.0)
        return MetaOutcome(curve=[0.2, 0.3], final=0.3)

    def harming(value, seed):
        # the evolved value is worse than the seed on this unseen problem
        if getattr(value, "c", 0.2) > 0.2:
            return MetaOutcome(curve=[0.0], final=0.0)
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)

    # helping on unseen -> adopt
    report, _ = meta_run(
        {"a": helping, "b": helping}, {"v": helping}, slot="task_sampler",
        spec=slot, propose=_deterministic_propose, seeds=(0, 1),
        val_seeds=(1, 2), rounds=1, n_workers=1)
    assert report.verdict == "adopt"
    assert report.mean_gain >= 0.0

    # harming on unseen (regression) -> reject
    report2, _ = meta_run(
        {"a": helping, "b": helping}, {"v": harming}, slot="task_sampler",
        spec=slot, propose=_deterministic_propose, seeds=(0, 1),
        val_seeds=(1, 2), rounds=1, n_workers=1)
    assert report2.verdict == "reject"
    assert "not above min_gain" in report2.reason or "below" in report2.reason


def test_transfer_ratio_contract():
    report = {
        "a": {"gain": 0.2, "n": 4, "before": 0.4, "after": 0.6},
        "b": {"gain": 0.1, "n": 4, "before": 0.4, "after": 0.5},
    }
    assert transfer_ratio(report, "a", "b") == pytest.approx(0.5)
    assert transfer_ratio({"a": {"gain": 0.0}}, "a", "b") is None


def test_meta_run_offline_and_deterministic(tmp_path):
    train = {"a": _inner_problem(_tasks(prefix="a")),
             "b": _inner_problem(_tasks(prefix="b"))}
    val = {"v": _inner_problem(_tasks(prefix="z"))}
    r1, _ = meta_run(train, val, slot="task_sampler", spec=_sampler_slot(),
                     propose=_deterministic_propose, seeds=(0, 1),
                     val_seeds=(10, 11), rounds=1, n_workers=1,
                     repo_path=str(tmp_path / "r1"))
    r2, _ = meta_run(train, val, slot="task_sampler", spec=_sampler_slot(),
                     propose=_deterministic_propose, seeds=(0, 1),
                     val_seeds=(10, 11), rounds=1, n_workers=1,
                     repo_path=str(tmp_path / "r2"))
    assert r1.value == r2.value
    assert r1.validation == r2.validation
