"""Bounding how long a round waits for its workers.

The aggregator is a round barrier, so a single hung rollout stalled the whole run
forever -- there was no way to bound the wait. `round_timeout` caps it and
abandons the stragglers (Python cannot cancel a thread, so their work continues in
the background but no longer holds up the round).
"""

import time

from agentdescent.evolution import AppendRules, Task, evolve


class _OneHangs:
    """Task t0 hangs; every other task is fast."""

    def __init__(self, hang=30.0):
        self.hang = hang

    def solve(self, rendered, task):
        if task.id == "t0":
            time.sleep(self.hang)
        return "yes" if task.meta["h"] in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return task.meta["h"]


def _tasks(n=12):
    return [Task(id=f"t{i}", prompt="q", meta={"h": f"H{i % 3}"}) for i in range(n)]


REWARD = lambda t, o: 1.0 if o == "yes" else 0.0


def test_round_timeout_abandons_a_hung_rollout():
    t0 = time.time()
    res = evolve(_tasks(), REWARD, agent=_OneHangs(hang=30.0), strategy=AppendRules(),
                 rounds=3, n_workers=4, max_concurrency=4, round_timeout=1.0)
    elapsed = time.time() - t0
    assert elapsed < 15.0, f"a 30s hang should not block the run; took {elapsed:.1f}s"
    assert len(res.history) == 3, "every round should still complete"
    assert res.error is None


def test_run_still_learns_from_the_workers_that_finished():
    res = evolve(_tasks(), REWARD, agent=_OneHangs(hang=20.0), strategy=AppendRules(),
                 rounds=3, n_workers=4, max_concurrency=4, round_timeout=1.0)
    assert res.state, "the fast workers' proposals should still be merged"
    assert res.final_reward > 0.0


def test_no_timeout_by_default_and_fast_runs_are_unaffected():
    """Default must not change behaviour for well-behaved rollouts."""
    class _Fast:
        def solve(self, rendered, task):
            return "yes" if task.meta["h"] in rendered else "no"

        def propose(self, rendered, task, output, reward):
            return task.meta["h"]

    a = evolve(_tasks(), REWARD, agent=_Fast(), strategy=AppendRules(),
               rounds=4, n_workers=3, max_concurrency=3)
    b = evolve(_tasks(), REWARD, agent=_Fast(), strategy=AppendRules(),
               rounds=4, n_workers=3, max_concurrency=3, round_timeout=30.0)
    assert a.state == b.state and a.final_reward == b.final_reward


def test_backend_errors_still_surface_through_the_timeout_path():
    """Abandoning stragglers must not swallow a real failure."""
    class _Dead:
        def solve(self, rendered, task):
            raise RuntimeError("backend is down")

        def propose(self, rendered, task, output, reward):
            return "x"

    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = evolve(_tasks(), REWARD, agent=_Dead(), strategy=AppendRules(),
                     rounds=2, n_workers=3, max_concurrency=3, round_timeout=5.0)
    assert res.error is not None and "backend is down" in res.error
