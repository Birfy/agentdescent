"""Tests for the task samplers (agentdescent.sampling).

A rollout is the expensive unit of work, so which task a worker picks matters.
These check the contract, the difficulty filter (zero-signal tasks get
down-weighted), and that both samplers drop into `evolve()` unchanged.
"""

from agentdescent.evolution import AppendRules, Task, evolve
from agentdescent.sampling import DifficultyWeighted, RoundRobin, TaskSampler


def test_both_satisfy_the_protocol_structurally():
    assert isinstance(RoundRobin(), TaskSampler)
    assert isinstance(DifficultyWeighted(), TaskSampler)


def test_round_robin_cycles_deterministically():
    s = RoundRobin()
    keys = ["a", "b", "c"]
    assert [s.pick(keys, i) for i in range(5)] == ["a", "b", "c", "a", "b"]


def test_round_robin_record_is_a_noop():
    s = RoundRobin()
    s.record("a", 1.0)                      # must not raise or change behaviour
    assert s.pick(["a", "b"], 0) == "a"


def test_difficulty_weighted_explores_untried_tasks_first():
    s = DifficultyWeighted()
    seen = set()
    for i in range(4):
        k = s.pick(["a", "b", "c", "d"], i)
        seen.add(k)
        s.record(k, 1.0)
    assert seen == {"a", "b", "c", "d"}     # every task tried before repeating


def test_difficulty_weighted_downweights_always_pass_tasks():
    """A task that always passes carries no gradient -> lose to one that fails."""
    s = DifficultyWeighted()
    for _ in range(6):
        s.record("solved", 1.0)             # always passes
        s.record("mixed", 1.0)              # ~50% pass -> maximal signal
        s.record("mixed", 0.0)
    picks = [s.pick(["solved", "mixed"], i) for i in range(10)]
    assert picks.count("mixed") > picks.count("solved")


def test_difficulty_weighted_downweights_never_pass_tasks():
    """A task that never passes is just as useless as one that always passes."""
    s = DifficultyWeighted()
    for _ in range(6):
        s.record("impossible", 0.0)
        s.record("mixed", 1.0)
        s.record("mixed", 0.0)
    picks = [s.pick(["impossible", "mixed"], i) for i in range(10)]
    assert picks.count("mixed") > picks.count("impossible")


def test_signal_peaks_at_half_and_vanishes_at_extremes():
    sig = DifficultyWeighted._signal
    assert sig(0.5) > sig(0.8) > sig(1.0)
    assert sig(0.5) > sig(0.2) > sig(0.0)
    assert sig(0.0) < 0.01 and sig(1.0) < 0.01


def test_pick_rejects_empty_keys():
    for s in (RoundRobin(), DifficultyWeighted()):
        try:
            s.pick([], 0)
        except (ValueError, IndexError):
            continue
        raise AssertionError(f"{type(s).__name__}.pick([]) should raise")


class _Agent:
    """Solves a task once its hint is in the artifact; only 'hard' tasks can fail."""

    def solve(self, rendered, task):
        if task.meta["hard"] and task.meta["hint"] not in rendered:
            return "no"
        return "yes"

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


def _tasks(n=20, n_hard=4):
    return [Task(id=f"t{i}", prompt="q", meta={"hard": i < n_hard, "hint": f"H{i}"})
            for i in range(n)]


REWARD = lambda t, o: 1.0 if o == "yes" else 0.0


def test_evolve_accepts_either_sampler_and_still_improves():
    for sampler in (RoundRobin(), DifficultyWeighted()):
        r = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                   rounds=8, n_workers=3, held_out_frac=0.3, task_sampler=sampler)
        assert r.error is None
        assert r.final_reward > 0.0


def test_default_sampler_is_round_robin_behaviour():
    """Omitting task_sampler must not change the existing deterministic default."""
    a = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
               rounds=6, n_workers=3, held_out_frac=0.3)
    b = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
               rounds=6, n_workers=3, held_out_frac=0.3, task_sampler=RoundRobin())
    assert a.state == b.state and a.final_reward == b.final_reward


def test_sampler_learns_from_recorded_outcomes_during_a_run():
    s = DifficultyWeighted()
    evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
           rounds=8, n_workers=3, held_out_frac=0.3, task_sampler=s)
    assert s.stats(), "the engine should have reported rollout outcomes"
    assert all(trials > 0 for _, trials in s.stats().values())


# --- select_hard: turning a saturated benchmark into one with headroom ---------

def test_select_hard_keeps_only_what_the_baseline_fails():
    from agentdescent.dataloader import select_hard
    items = list(range(20))
    hard = select_hard(items, lambda i: 1.0 if i % 2 == 0 else 0.0)
    assert hard == [i for i in items if i % 2]


def test_select_hard_returns_everything_when_nothing_fails():
    """An empty benchmark is worse than a saturated one -- say so by returning it."""
    from agentdescent.dataloader import select_hard
    items = list(range(10))
    assert select_hard(items, lambda i: 1.0) == items


def test_select_hard_caps_with_keep():
    from agentdescent.dataloader import select_hard
    items = list(range(20))
    assert select_hard(items, lambda i: 0.0, keep=3) == [0, 1, 2]


def test_select_hard_handles_an_empty_pool():
    from agentdescent.dataloader import select_hard
    assert select_hard([], lambda i: 0.0) == []


def test_select_hard_scores_concurrently():
    """It runs a whole baseline pass, so serial scoring would make it unusable."""
    import threading
    from agentdescent.dataloader import select_hard

    live, peak, lock = [0], [0], threading.Lock()

    def score(i):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        import time
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return 0.0

    select_hard(list(range(16)), score, concurrency=8)
    assert peak[0] > 1, "select_hard scored the pool serially"
