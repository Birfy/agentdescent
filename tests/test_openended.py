"""Open-ended co-evolution must keep novelty on the frontier, not just score."""

import random

import pytest

from agentdescent.openended import (
    OpenEndedConfig,
    TaskGenome,
    open_ended_coevolve,
)


def _generator(parent, seed):
    rng = random.Random(seed)
    base = parent.difficulty if parent else 0.2
    difficulty = max(0.0, min(1.0, base + rng.choice([-0.08, 0.08])))
    return TaskGenome(f"t{seed}", difficulty=difficulty,
                      descriptor=(difficulty, seed % 7 / 7))


def _solver(task):
    return 1.0 - abs(task.difficulty - 0.6)


def test_coevolution_keeps_a_difficulty_frontier_and_best():
    report = open_ended_coevolve(_generator, _solver, OpenEndedConfig(
        generations=8, population=8, target_difficulty=0.6,
        frontier_width=0.2, novelty_threshold=0.01, seed=3))
    assert report.entries
    assert report.frontier
    assert all(abs(entry.task.difficulty - 0.6) <= 0.2
               for entry in report.frontier)
    assert report.best is not None
    assert report.best.score > 0.8
    assert 0 < report.coverage <= 1


def test_zero_novelty_children_are_not_added():
    def same(parent, seed):
        return TaskGenome(f"t{seed}", difficulty=0.5, descriptor=(0.5,))

    report = open_ended_coevolve(same, lambda _t: 0.5,
                                 OpenEndedConfig(generations=3, population=3,
                                                 novelty_threshold=0.1))
    assert len(report.entries) == 1


def test_same_seed_is_reproducible():
    config = OpenEndedConfig(generations=4, population=5, seed=11)
    one = open_ended_coevolve(_generator, _solver, config)
    two = open_ended_coevolve(_generator, _solver, config)
    assert [e.task.task_id for e in one.entries] == [e.task.task_id for e in two.entries]


def test_invalid_config_and_solvers_are_refused():
    with pytest.raises(ValueError):
        TaskGenome("", difficulty=0.5)
    with pytest.raises(ValueError):
        TaskGenome("t", difficulty=1.5)
    with pytest.raises(ValueError):
        TaskGenome("t", difficulty=0.5, descriptor=())
    with pytest.raises(ValueError):
        OpenEndedConfig(generations=0)
    with pytest.raises(ValueError):
        OpenEndedConfig(target_difficulty=1.5)
    with pytest.raises(ValueError):
        OpenEndedConfig(min_difficulty=0.8, max_difficulty=0.2)
    with pytest.raises(TypeError):
        open_ended_coevolve(None, _solver)
    with pytest.raises(TypeError):
        open_ended_coevolve(lambda *_: "bad", _solver)
    with pytest.raises(ValueError):
        open_ended_coevolve(_generator, lambda _t: float("nan"))
