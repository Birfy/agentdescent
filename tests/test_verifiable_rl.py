"""Verifiable-reward policy improvement has to be right where truth is checkable.

The reward is exact match (1.0 / 0.0, carried in the outcome's detail); the
policy improves only on verifiable progress; the report carries the per-round
curve and the held-out before/after. The fixtures pin the surfaces that make
this honest: a rule that cannot be checked is not rewarded, the verifiable
problem carries output and gold, and policy_improve climbs a monotone verifiable
landscape.
"""

import re

import pytest

from agentdescent import (
    SourceSlot,
    Task,
    VerifiableReport,
    policy_improve,
    verifiable_problem,
    verifiable_reward,
)


def _cap_of(value):
    m = re.search(r"cap:(\d+)", value.t)
    return int(m.group(1)) if m else 0


def _cap_policy(text):
    class P:
        def __init__(self, t):
            self.t = t.strip()
        def render(self):
            return self.t
    return P(text)


def _cap_spec():
    return SourceSlot(initial_value="cap:0",
                      validate=lambda t: t.strip() + "\n", build=_cap_policy)


def _task(difficulty, seed=0):
    return Task(id=f"t{difficulty}", prompt=f"item {difficulty}",
                meta={"difficulty": difficulty, "gold": str(2 * difficulty)})


def _gold(task):
    return task.meta["gold"]


def _problem_fn(value, task):
    return task.meta["gold"] if task.meta["difficulty"] <= _cap_of(value) else "?"


def _task_for(seed):
    difficulties = (1, 2, 3, 4, 5, 6, 7, 8)
    return _task(difficulties[seed % len(difficulties)])


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return rendered
    return "cap:8"      # jump straight to the ceiling: a verifiable, big gain


def test_verifiable_reward_checks_exact_match():
    from agentdescent import MetaOutcome
    reward = verifiable_reward()
    assert reward(MetaOutcome(curve=[0], detail={"output": "6", "gold": "6"})) == 1.0
    assert reward(MetaOutcome(curve=[0], detail={"output": "?", "gold": "6"})) == 0.0
    assert reward(MetaOutcome(curve=[0], detail={})) == 0.0      # cannot check -> 0


def test_verifiable_problem_carries_output_and_gold():
    problem = verifiable_problem(_problem_fn, _gold, task_for=_task_for)
    outcome = problem(_cap_spec().compile("cap:4"), 3)          # difficulty 4
    assert outcome.detail["output"] == "8"
    assert outcome.detail["gold"] == "8"
    assert outcome.final == 1.0


def test_policy_improve_climbs_on_verifiable_reward():
    problems = {f"p{i}": verifiable_problem(_problem_fn, _gold, task_for=_task_for)
                for i in range(4)}
    val = {f"v{i}": verifiable_problem(_problem_fn, _gold, task_for=_task_for)
           for i in range(4)}
    report, result = policy_improve(
        problems, val, slot="task_sampler", spec=_cap_spec(),
        propose=_propose, seeds=(0, 1, 2, 3), val_seeds=(4, 5),
        rounds=3, n_workers=1, blast_radius=0.2, repo_path=None)
    assert isinstance(report, VerifiableReport)
    assert report.improved_reward > report.seed_reward
    assert "cap:8" in report.value
    assert result.meta["improved_reward"] == pytest.approx(report.improved_reward)
    # the curve is the engine's per-round held-out verifiable reward, not the
    # final value repeated for every round
    assert report.reward_curve == [round(float(h.held_out_reward), 4)
                                   for h in result.history]
    assert len(report.reward_curve) == len(result.history)


def test_the_report_scores_held_out_facts():
    calls = {"n": 0}

    def tracked_problem(value, seed):
        calls["n"] += 1
        return verifiable_problem(_problem_fn, _gold, task_for=_task_for)(value, seed)

    problems = {f"p{i}": verifiable_problem(_problem_fn, _gold, task_for=_task_for)
                for i in range(4)}
    val = {f"v{i}": tracked_problem for i in range(4)}
    report, result = policy_improve(
        problems, val, slot="task_sampler", spec=_cap_spec(),
        propose=_propose, seeds=(0, 1, 2, 3), val_seeds=(4, 5),
        rounds=1, n_workers=1, blast_radius=0.2, repo_path=None)
    assert calls["n"] > 0                       # held-out facts were scored
    assert report.seed_reward <= report.improved_reward


def test_policy_improve_needs_a_proposer():
    with pytest.raises(ValueError):
        policy_improve({"p": verifiable_problem(_problem_fn, _gold, task_for=_task_for)},
                       {"v": verifiable_problem(_problem_fn, _gold, task_for=_task_for)},
                       slot="task_sampler", spec=_cap_spec())
