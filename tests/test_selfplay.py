"""The solver-challenger product has to be right in the places self-play is
dangerous and a product is load-bearing.

A challenger that writes its own task distribution could drift into questions
that are unsolvable (degenerate hardness), trivially answered (no learning),
or leaking the evaluation the run reports on. Every fixture pins one of those,
plus the product's core claim: that the solver's score on *generated* tasks
rises as it learns, measured online from first-vs-last attempt.
"""

import random

from agentdescent import (
    Challenger,
    ChallengerConfig,
    SingleSlot,
    Task,
    curriculum_evolve,
)
from agentdescent.selfplay import FrontierSampler


# -- a deterministic domain with a growing capability ---------------------------
#
# The solver skill carries a capability cap ("cap:N" in its text). It answers
# exactly only when both operands are within the cap; a generated task with
# larger operands fails until the solver's propose raises the cap. So learning
# is gradual and driven by the challenger: the frontier's r0 measures how fast
# the solver catches up -- exactly the product's claim.

import re


def _cap(rendered):
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, task):
    cap = _cap(rendered)
    a, b = task.meta.get("a", 0), task.meta.get("b", 0)
    if a <= cap and b <= cap:
        return str(task.meta["gold"])
    return "maybe " + str(task.meta["gold"])


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return None
    a, b = task.meta.get("a", 0), task.meta.get("b", 0)
    need = max(a, b)
    if need > _cap(rendered):
        return f"cap:{need + 5}"
    return "cap:1"


def _seed_tasks(n=8):
    return [Task(id=f"s{i}", prompt=f"base item {i}",
                 meta={"gold": str(2 * i), "a": i, "b": i}) for i in range(1, n + 1)]


def _make_challenger(*, spec="20", invalid=False, generated=3):
    def generate(spec_text, seed):
        rng = random.Random(seed)
        limit = int(spec_text) if str(spec_text).isdigit() else 20
        if invalid and rng.random() < 0.4:
            return {"a": "not-an-int", "b": 3}
        return {"a": rng.randint(1, limit), "b": rng.randint(1, limit)}

    def validate(payload):
        a, b = payload["a"], payload["b"]
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ValueError("generated task must have integer operands")
        if not (1 <= a <= 1000 and 1 <= b <= 1000):
            raise ValueError("generated operands out of range")
        return Task(id=f"g{a}-{b}", prompt=f"What is {a} + {b}?",
                    meta={"gold": str(a + b), "a": a, "b": b})

    def propose(spec_text, report):
        if report.get("r0", 0.0) > 0.05 and str(spec_text).isdigit() \
                and int(spec_text) < 100:
            return str(int(spec_text) + 10)
        return spec_text

    return Challenger(generate=generate, validate=validate, propose=propose,
                      spec=spec,
                      config=ChallengerConfig(generated_per_round=generated,
                                              frontier_share=0.6, r0_floor=0.05,
                                              max_generate_attempts=10))


# -- the sampler in isolation ---------------------------------------------------


def test_the_sampler_mixes_seed_and_frontier():
    ch = _make_challenger()
    sampler = FrontierSampler(ch, seed=0)
    gen = _seed_tasks()[:2]
    sampler.install(gen)
    # with frontier_share=1 a sampler picks frontier tasks only
    ch.config = ChallengerConfig(frontier_share=1.0, max_frontier_pool=64)
    picked = sampler.pick(["s1", "s2"], 0)
    assert isinstance(picked, Task)
    assert picked.id in {t.id for t in gen}


def test_first_attempt_pass_drops_a_task_from_the_frontier():
    ch = _make_challenger()
    ch.config = ChallengerConfig(frontier_share=1.0, before_gap=0.05)
    sampler = FrontierSampler(ch, seed=0)
    task = Task(id="g1", prompt="p", meta={"gold": "1"})
    sampler.install([task])
    assert task.id in sampler._book
    sampler.record("g1", 1.0)                 # already passes: never frontier
    assert task.id not in sampler._book
    assert sampler.stats()["learned"] == 1


def test_the_sampler_measures_learnability_first_vs_last():
    ch = _make_challenger()
    ch.config = ChallengerConfig(frontier_share=1.0, before_gap=0.05,
                                 stuck_gap=0.0, max_attempts=3)
    sampler = FrontierSampler(ch, seed=0)
    task = Task(id="g9", prompt="p", meta={"gold": "9"})
    sampler.install([task])
    sampler.record("g9", 0.0)                 # fails at first
    assert sampler._book["g9"]["first"] == 0.0
    sampler.record("g9", 0.0)                 # still fails
    sampler.record("g9", 1.0)                 # learned
    report = sampler.sweep(0.999)
    assert report["learned"] >= 1
    assert report["r0"] > 0.0                 # first 0 -> last 1


def test_a_task_that_never_learns_is_dropped_as_stuck():
    ch = _make_challenger()
    ch.config = ChallengerConfig(frontier_share=1.0, before_gap=0.05,
                                 stuck_gap=0.0, max_attempts=2)
    sampler = FrontierSampler(ch, seed=0)
    task = Task(id="g5", prompt="p", meta={"gold": "5"})
    sampler.install([task])
    sampler.record("g5", 0.0)
    sampler.record("g5", 0.0)                 # never learns
    report = sampler.sweep(0.999)
    assert report["stuck"] >= 1
    assert task.id not in sampler._book


# -- the driver end to end --------------------------------------------------------


def test_solver_challenger_end_to_end(tmp_path):
    tasks = _seed_tasks()
    challenger = _make_challenger()
    result = curriculum_evolve(
        tasks, _reward, run=_run, propose=_propose,
        strategy=SingleSlot(initial_value="cap:5"),
        challenger=challenger, rounds=8, n_workers=2,
        repo_path=str(tmp_path / "repo"), seed=0)
    assert result.error is None
    assert result.outcomes().get("committed", 0) >= 1       # the solver learned
    assert result.curriculum is not None
    assert result.curriculum["generated_total"] >= 1
    assert result.curriculum["r0_mean"] > 0.0               # learnability is real
    # held-out is measured on seed tasks only: the challenger cannot touch it
    assert result.final_reward > 0.0


def test_the_challenger_evolves_its_spec_toward_hardness(tmp_path):
    tasks = _seed_tasks()
    challenger = _make_challenger(spec="10")
    result = curriculum_evolve(
        tasks, _reward, run=_run, propose=_propose,
        strategy=SingleSlot(initial_value="cap:5"),
        challenger=challenger, rounds=8, n_workers=2,
        repo_path=str(tmp_path / "repo"), seed=1)
    assert int(challenger.spec) > 10            # r0>0 -> difficulty rose


def test_invalid_generations_are_counted_not_crashed(tmp_path):
    tasks = _seed_tasks()
    challenger = _make_challenger(invalid=True)
    result = curriculum_evolve(
        tasks, _reward, run=_run, propose=_propose,
        strategy=SingleSlot(initial_value="cap:5"),
        challenger=challenger, rounds=8, n_workers=2,
        repo_path=str(tmp_path / "repo"), seed=0)
    assert result.error is None
    assert result.curriculum["invalid_total"] >= 1


def test_the_run_is_deterministic_under_a_seed(tmp_path):
    challenger = _make_challenger()
    a = curriculum_evolve(_seed_tasks(), _reward, run=_run, propose=_propose,
                          strategy=SingleSlot(initial_value="cap:5"),
                          challenger=challenger, rounds=8, n_workers=2,
                          repo_path=str(tmp_path / "a"), seed=7)
    b = curriculum_evolve(_seed_tasks(), _reward, run=_run, propose=_propose,
                          strategy=SingleSlot(initial_value="cap:5"),
                          challenger=_make_challenger(), rounds=8, n_workers=2,
                          repo_path=str(tmp_path / "b"), seed=7)
    assert a.curriculum["r0_mean"] == b.curriculum["r0_mean"]


def test_seed_tasks_are_still_sampled():
    # frontier_share < 1 must not starve the fixed curriculum
    ch = _make_challenger()
    ch.config = ChallengerConfig(frontier_share=0.5, generated_per_round=2,
                                 max_frontier_pool=64)
    sampler = FrontierSampler(ch, seed=0)
    sampler.install([Task(id="g1", prompt="p", meta={"gold": "1"})])
    picks = [sampler.pick(["s1", "s2", "s3"], i) for i in range(50)]
    assert any(isinstance(p, str) for p in picks)      # seed ids appear
    assert any(isinstance(p, Task) for p in picks)     # frontier tasks appear
