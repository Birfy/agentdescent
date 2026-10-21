"""Self red-teaming has to be right where the framework attacks its own value.

The attacker mutates the training tasks outward and scores the value on the
mutations, collecting the OOD failures; retraining on them closes the gap. The
fixtures pin the surfaces that make this honest: the attack finds the failures
a low cap leaves open, the retraining closes them (coverage rises), and the
run needs enough problems to split.
"""

import re

import pytest

from agentdescent import (
    RedTeamReport,
    SingleSlot,
    Task,
    red_team,
    red_team_evolve,
)


def _cap(rendered):
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, task):
    return task.meta["gold"] if task.meta["difficulty"] <= _cap(rendered) else "?"


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return rendered
    return f"cap:{task.meta['difficulty']}"     # jump to the failing task


def _task(difficulty, seed=0):
    return Task(id=f"t{difficulty}", prompt=f"item {difficulty}",
                meta={"difficulty": difficulty, "gold": str(2 * difficulty)})


def _mutate(task, i):
    # push the difficulty outward: the attack's OOD probe (unique ids, so the
    # retraining can add the failures without id collisions)
    d = task.meta["difficulty"] + 1 + i % 4
    return Task(id=f"attack{i}:t{d}", prompt=f"item {d}",
                meta={"difficulty": d, "gold": str(2 * d)})


def _strategy():
    return SingleSlot(initial_value="cap:0")


def test_red_team_finds_the_failures_a_low_cap_leaves():
    value = "cap:0"
    failures = red_team(value, lambda v, t: _reward(t, _run(v, t)),
                        _mutate, [_task(1), _task(2), _task(3), _task(4)],
                        budget=8)
    assert len(failures) >= 1
    assert all(d[1] < 0.999 for d in failures)      # every failure really fails


def test_red_team_is_deterministic():
    kwargs = dict(value="cap:0", score=lambda v, t: _reward(t, _run(v, t)),
                  mutate=_mutate,
                  seed_tasks=[_task(1), _task(2), _task(3), _task(4)], budget=6)
    assert red_team(**kwargs) == red_team(**kwargs)


def test_red_team_evolve_closes_the_ood_gap():
    problems = {f"p{d}": _task(d) for d in (1, 2, 3, 4)}
    report, result = red_team_evolve(
        problems, run=_run, reward=_reward, propose=_propose,
        mutate=_mutate, strategy=_strategy(),
        attack_budget=8, retrain_rounds=2, rounds=1, n_workers=1,
        repo_path=None)
    assert isinstance(report, RedTeamReport)
    assert report.failures_found >= 1               # the attack found a gap
    assert report.coverage_after > report.coverage_before   # retraining closed it
    assert result.meta["failures_found"] == report.failures_found


def test_red_team_evolve_needs_enough_problems():
    with pytest.raises(ValueError):
        red_team_evolve({"p1": _task(1), "p2": _task(2), "p3": _task(3)},
                        run=_run, reward=_reward, propose=_propose,
                        mutate=_mutate, strategy=_strategy())


def test_retraining_starts_from_phase_one_s_state(monkeypatch):
    # without initial_state, phase 2 would restart from the seed and can
    # regress below phase 1's value; verify the second evolve() inherits
    # phase 1's state
    import importlib
    rt = importlib.import_module("agentdescent.red_team")
    seen = {"first": False}
    real = rt.evolve

    def spy(*args, **kwargs):
        if seen["first"]:
            seen["second_initial_state"] = kwargs.get("initial_state")
        else:
            seen["first"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(rt, "evolve", spy)
    problems = {f"p{d}": _task(d) for d in (1, 2, 3, 4)}
    red_team_evolve(
        problems, run=_run, reward=_reward, propose=_propose,
        mutate=_mutate, strategy=_strategy(),
        attack_budget=4, retrain_rounds=1, rounds=1, n_workers=1,
        repo_path=None)
    assert "second_initial_state" in seen
    assert seen["second_initial_state"] is not None   # phase 1's evolved state


def test_memorising_the_failures_does_not_close_the_gap():
    # a value that solves exactly the retrained ids must not score coverage
    # on a fresh attack -- otherwise "closing the OOD gap" is measured
    # in-sample
    def ids_run(rendered, task):
        # rendered = "ids:1,2,3" solves exactly those task ids
        solved = {int(x) for x in rendered.split(":")[1].split(",")}
        return task.meta["gold"] if task.meta["difficulty"] in solved else "?"

    def append_failing(rendered, task, output, reward):
        if reward >= 0.999:
            return rendered
        solved = rendered.split(":")[1].split(",")
        return f"ids:{','.join(solved + [str(task.meta['difficulty'])])}"

    def ids_mutate(task, i):
        d = task.meta["difficulty"] + 1 + i % 4
        return Task(id=f"attack{i}:t{d}", prompt=f"item {d}",
                    meta={"difficulty": d, "gold": str(2 * d)})

    problems = {f"p{d}": _task(d) for d in (1, 2, 3, 4)}
    report, result = red_team_evolve(
        problems, run=ids_run, reward=_reward, propose=append_failing,
        mutate=ids_mutate, strategy=SingleSlot(initial_value="ids:1,2,3,4"),
        attack_budget=8, retrain_rounds=6, rounds=1, n_workers=1,
        repo_path=None)
    # the memorising value scores 1.0 on the retrained failures but a fresh
    # attack still finds gaps, so coverage_after stays below 1.0
    assert report.failures_found >= 1
    assert report.coverage_after < 1.0
