"""RSI-Bench must measure generations, transfer, forgetting and hacking."""

import pytest

from bench.rsi_bench import (
    BenchStep,
    BenchmarkTask,
    RSIBench,
)


def _tasks(n=6):
    return [BenchmarkTask(f"t{i}", family="fam") for i in range(n)]


def _step(score, oracle=None, cost=0.0, error=None):
    return lambda _task, _gen, _budget, _split: BenchStep(
        score, oracle=oracle, cost=cost, error=error)


def test_improving_solver_transfers_to_holdout():
    bench = RSIBench(_tasks(), generations=4, target=0.7)
    report = bench.run(lambda _task, gen, _budget, _split: BenchStep(
        0.5 + 0.1 * gen, cost=1))
    assert report.verdict == "improving"
    assert report.train_gain == pytest.approx(0.3)
    assert report.holdout_gain == pytest.approx(0.3)
    assert report.transfer == pytest.approx(1.0)
    assert report.generations_to_target == 2
    assert report.cost == 24


def test_flat_solver_is_flat_and_no_transfer_ratio():
    bench = RSIBench(_tasks(), generations=3)
    report = bench.run(_step(0.4))
    assert report.verdict == "flat"
    assert report.transfer is None
    assert report.holdout_gain == 0.0


def test_train_only_gain_is_no_transfer_not_flat():
    bench = RSIBench(_tasks(10), generations=2)
    train, holdout = bench.split()

    def step(task, generation, _budget, split):
        if split == "train":
            return BenchStep(0.2 + 0.2 * generation)
        return BenchStep(0.2)

    report = bench.run(step)
    assert report.verdict == "no-transfer"
    assert report.train_gain > 0
    assert report.holdout_gain == 0


def test_forgetting_reports_max_and_fraction():
    bench = RSIBench(_tasks(4), generations=4)
    scores = [0.5, 0.9, 0.7, 0.8]

    def step(task, generation, _budget, _split):
        if task.task_id == "t0":
            return BenchStep([0.5, 1.5, 0.4, 0.5][generation])
        return BenchStep(scores[generation])

    report = bench.run(step)
    assert report.max_forgetting > 0.9
    assert report.forgetting_fraction > 0
    assert report.verdict == "forgetting"


def test_reward_hacking_is_a_gain_gap_not_an_offset():
    bench = RSIBench(_tasks(), generations=4, hacking_tolerance=0.05)
    report = bench.run(lambda _task, gen, _budget, _split: BenchStep(
        0.5 + 0.1 * gen, oracle=0.5))
    assert report.reward_hacking_gap > 0.05
    assert report.verdict == "hacking"

    # A constant offset is not hacking: both sides gain the same amount.
    offset = RSIBench(_tasks(), generations=3)
    report = offset.run(lambda _task, gen, _budget, _split: BenchStep(
        0.2 + 0.1 * gen, oracle=0.1 + 0.1 * gen))
    assert report.reward_hacking_gap == pytest.approx(0.0)
    assert report.verdict == "improving"


def test_split_is_deterministic_and_order_independent():
    tasks = _tasks(10)
    one = RSIBench(tasks, generations=2).split()
    two = RSIBench(reversed(tasks), generations=2).split()
    assert {t.task_id for t in one[0]} == {t.task_id for t in two[0]}
    assert {t.task_id for t in one[1]} == {t.task_id for t in two[1]}
    assert len(one[1]) >= 1


def test_report_views_and_serialisation_keep_the_curve():
    bench = RSIBench(_tasks(), generations=2)
    report = bench.run(lambda _task, gen, _budget, _split: BenchStep(
        0.1 * gen, oracle=0.1 * gen))
    assert len(report.by_split("train")) >= 1
    assert len(report.by_split("holdout")) >= 1
    assert report.by_family()["fam"]
    data = report.to_dict()
    assert data["generations"] == 2
    assert len(data["tasks"]) == 6
    assert {"family", "oracle", "costs", "errors"} <= set(data["tasks"][0])


def test_errors_and_cost_cap_are_real():
    bench = RSIBench(_tasks(), generations=2)
    report = bench.run(_step(0.1, error="boom"))
    assert report.errors == 12
    assert report.verdict == "insufficient"

    capped = RSIBench(_tasks(), generations=2, max_cost=1)
    with pytest.raises(RuntimeError):
        capped.run(_step(0.1, cost=10))


def test_invalid_tasks_and_config_are_refused():
    with pytest.raises(ValueError):
        RSIBench([])
    with pytest.raises(ValueError):
        RSIBench([BenchmarkTask("a"), BenchmarkTask("a")])
    with pytest.raises(ValueError):
        RSIBench(_tasks(), generations=1)
    with pytest.raises(ValueError):
        RSIBench(_tasks(), holdout_fraction=0)
    with pytest.raises(ValueError):
        RSIBench(_tasks(), target=float("nan"))
    with pytest.raises(ValueError):
        RSIBench(_tasks(), hacking_tolerance=None)
    with pytest.raises(ValueError):
        RSIBench(_tasks(), forgetting_tolerance=-1)
    with pytest.raises(ValueError):
        RSIBench(_tasks(), max_cost=0)
    with pytest.raises(ValueError):
        BenchmarkTask("", "fam")
    with pytest.raises(ValueError):
        BenchmarkTask("a", "")
    with pytest.raises(ValueError):
        BenchStep(float("nan"))
    with pytest.raises(ValueError):
        BenchStep(0.1, cost=-1)
    with pytest.raises(ValueError):
        BenchStep(True)
    bench = RSIBench(_tasks())
    with pytest.raises(TypeError):
        bench.run(lambda *_: 0.5)
    with pytest.raises(ValueError):
        bench.run(lambda *_: BenchStep(0.5), step_budget=0)


def test_a_crashed_generation_cannot_read_as_improvement_or_forgetting():
    def first_generation_crashes(_task, generation, _budget, _split):
        if generation == 0:
            raise RuntimeError("infra")
        return BenchStep(0.5)

    report = RSIBench(_tasks(), generations=3).run(first_generation_crashes)
    assert report.errors == 6
    assert report.verdict == "errored"

    def last_generation_crashes(_task, generation, _budget, _split):
        if generation == 2:
            raise RuntimeError("infra")
        return BenchStep(0.5)

    report = RSIBench(_tasks(), generations=3).run(last_generation_crashes)
    assert report.verdict == "errored"


def test_returned_error_is_also_fatal_to_the_verdict():
    report = RSIBench(_tasks(), generations=2).run(
        lambda *_: BenchStep(0.5, error="tool timeout"))
    assert report.verdict == "insufficient"


def test_error_budget_cannot_turn_a_crash_into_improvement():
    def first_generation_crashes(_task, generation, _budget, _split):
        if generation == 0:
            raise RuntimeError("infra")
        return BenchStep(0.5)

    report = RSIBench(_tasks(), generations=3, max_error_fraction=1.0).run(
        first_generation_crashes)
    assert report.errors == 6
    assert report.verdict != "improving"
    assert report.train_gain == 0 and report.holdout_gain == 0
