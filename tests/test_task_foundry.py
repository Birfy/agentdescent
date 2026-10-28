"""A task foundry has to reject junk before it becomes a curriculum."""

import pytest

from agentdescent.evolvable import stable_hash
from agentdescent.task_foundry import (
    FoundryConfig,
    TaskCandidate,
    TaskFoundry,
    failure_seeds,
)


def _candidate(i, prompt=None, difficulty=0.5):
    return TaskCandidate(f"t{i}", prompt or f"task number {i}",
                         difficulty=difficulty)


def _generator(seed, attempt):
    return TaskCandidate(f"{seed.task_id}-{attempt}",
                         f"{seed.prompt} variant {attempt}")


def _foundry(**config):
    cfg = FoundryConfig(**config)
    return TaskFoundry(cfg, generator=_generator,
                       validator=lambda candidate: 1.0)


def test_failure_seeds_extract_low_reward_rows_with_stable_ids():
    rows = [
        {"task_id": "a", "prompt": "p1", "reward": 0.2},
        {"task_id": "b", "prompt": "p2", "reward": 0.9},
        {"prompt": "p3", "reward": 0.1},
        {"prompt": "", "reward": 0.0},
    ]
    seeds = failure_seeds(rows, threshold=0.5)
    assert len(seeds) == 2
    assert seeds[0].task_id == "a"
    assert seeds[1].task_id.startswith("failure-")
    assert all(seed.difficulty >= 0.5 for seed in seeds)
    with pytest.raises(ValueError):
        failure_seeds(rows, threshold=1.5)
    with pytest.raises(ValueError):
        failure_seeds([{"prompt": "p", "reward": float("nan")}])
    assert failure_seeds([{"prompt": "p"}]) == ()


def test_holdout_membership_is_per_task_and_stable_as_the_pool_grows():
    all_ids = [f"t{i}" for i in range(100)]
    held_ids = [i for i in all_ids
                if stable_hash(("", i)) % 10_000 < 5_000][:5]
    train_ids = [i for i in all_ids
                 if stable_hash(("", i)) % 10_000 >= 5_000][:5]
    chosen = [TaskCandidate(i, i) for i in held_ids + train_ids]
    foundry = TaskFoundry(FoundryConfig(holdout_fraction=0.5, per_seed=1),
                          generator=lambda seed, _i: seed,
                          validator=lambda _candidate: 1.0)
    one = foundry.run(chosen)
    two = foundry.run(reversed(chosen))
    assert one.verdict == "ready"
    assert {t.task_id for t in one.holdout} == set(held_ids)
    assert {t.task_id for t in one.train} == set(train_ids)
    assert {t.task_id for t in one.train} == {t.task_id for t in two.train}
    assert {t.task_id for t in one.holdout} == {t.task_id for t in two.holdout}
    added = foundry.run(chosen + [TaskCandidate(f"x{i}", f"x{i}") for i in range(10)])
    assert {t.task_id for t in one.holdout} <= {t.task_id for t in added.holdout}


def test_invalid_and_duplicate_candidates_are_rejected():
    foundry = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0, per_seed=1, min_tasks=2),
        generator=lambda seed, _i: seed,
        validator=lambda candidate: 0.0 if candidate.task_id == "tbad" else 1.0,
    )
    report = foundry.run([_candidate("ok"), _candidate("bad")])
    assert report.verdict == "insufficient"
    assert report.rejection_counts()["invalid"] == 1

    duplicate = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0, per_seed=1),
        generator=lambda seed, _i: seed,
        validator=lambda _candidate: 1.0,
    )
    report = duplicate.run([
        TaskCandidate("a", "Same   Prompt"),
        TaskCandidate("b", "same prompt"),
    ])
    assert report.accepted == (TaskCandidate("a", "Same   Prompt"),)
    assert report.rejection_counts()["duplicate"] == 1


def test_near_duplicate_hook_and_contamination_and_learnability():
    near = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0, near_duplicate_threshold=0.8),
        generator=lambda seed, _i: seed,
        validator=lambda _candidate: 1.0,
        similarity=lambda left, right: 0.9 if "copy" in left.prompt else 0.1,
    )
    report = near.run([_candidate("a", "original"), _candidate("b", "copy")])
    assert report.rejection_counts()["near-duplicate"] == 1

    contaminated = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0, min_tasks=0),
        generator=lambda seed, _i: seed,
        validator=lambda _candidate: 1.0,
        contamination=lambda candidate: 1.0 if candidate.task_id == "tdirty" else 0.0,
    )
    assert contaminated.run([_candidate("dirty")]).verdict == "filtered"

    unlearnable = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0, min_learnability=0.5),
        generator=lambda seed, _i: seed,
        validator=lambda _candidate: 1.0,
        learnability=lambda candidate: 0.0 if candidate.task_id == "thopeless" else 1.0,
    )
    assert unlearnable.run([_candidate("hopeless")]).rejection_counts()["unlearnable"] == 1


def test_duplicate_ids_none_generator_and_candidate_cap():
    calls = []

    def generator(seed, attempt):
        calls.append((seed.task_id, attempt))
        return TaskCandidate("same", f"prompt {len(calls)}")

    foundry = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0, per_seed=2, max_candidates=3),
        generator=generator, validator=lambda _candidate: 1.0)
    report = foundry.run([_candidate("seed"), _candidate("seed2")])
    assert report.rejection_counts()["duplicate-id"] == 2
    assert len(calls) == 3

    none = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0),
        generator=lambda _seed, _i: None,
        validator=lambda _candidate: 1.0,
    )
    assert none.run([_candidate("seed")]).verdict == "insufficient"


def test_fraction_zero_is_all_train():
    foundry = _foundry(holdout_fraction=0.0, per_seed=1)
    report = foundry.run([_candidate("only")])
    assert report.verdict == "ready"
    assert report.holdout == ()
    assert report.train == report.accepted


def test_report_serialisation_and_counts():
    foundry = _foundry(holdout_fraction=0.0, per_seed=1)
    report = foundry.run([_candidate("a"), _candidate("b")])
    data = report.to_dict()
    assert set(data) == {"train", "holdout", "rejected", "verdict", "detail"}
    assert data["verdict"] == "ready"
    assert len(data["train"]) + len(data["holdout"]) == 2


def test_invalid_config_candidates_and_callables_are_refused():
    for kwargs in ({"per_seed": 0}, {"max_candidates": 0},
                   {"min_validation": 1.5}, {"holdout_fraction": -0.1},
                   {"holdout_fraction": 1.0}, {"near_duplicate_threshold": 2}):
        with pytest.raises(ValueError):
            FoundryConfig(**kwargs)
    with pytest.raises(ValueError):
        TaskCandidate("", "prompt")
    with pytest.raises(ValueError):
        TaskCandidate("a", "")
    with pytest.raises(ValueError):
        TaskCandidate("a", "p", difficulty=1.5)
    with pytest.raises(TypeError):
        TaskFoundry(FoundryConfig(), generator=None, validator=lambda _c: 1.0)


def test_too_easy_tasks_are_filtered():
    foundry = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0, min_learnability=0.2,
                      max_learnability=0.8, min_tasks=0),
        generator=lambda seed, _i: seed,
        validator=lambda _candidate: 1.0,
        learnability=lambda candidate: 1.0 if candidate.task_id == "teasy" else 0.5,
    )
    report = foundry.run([_candidate("easy")])
    assert report.rejection_counts()["too-easy"] == 1
    assert report.verdict == "filtered"


def test_callable_errors_are_recorded_not_fatal():
    def generator(seed, _attempt):
        if seed.task_id == "tboom":
            raise RuntimeError("generator")
        return seed

    foundry = TaskFoundry(
        FoundryConfig(holdout_fraction=0.0, per_seed=1),
        generator=generator, validator=lambda _candidate: 1.0,
        contamination=lambda candidate: 1 / 0 if candidate.task_id == "tdirty" else 0.0,
    )
    report = foundry.run([_candidate("ok"), _candidate("boom"), _candidate("dirty")])
    assert report.rejection_counts()["generator-error:RuntimeError"] == 1
    assert report.rejection_counts()["contamination-error:ZeroDivisionError"] == 1
    assert report.verdict == "filtered"
