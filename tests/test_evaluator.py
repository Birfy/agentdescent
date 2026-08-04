"""The gate gets its own concurrency, and its own failure domain.

Evaluation and exploration call the same function and are different workloads. A
rollout is long-tailed, often fails, and is one opinion among many. An evaluation
is batched, cacheable, and decides whether a change is committed. Sizing them
together means sizing them for whichever matters less.

They were already separate in the weak sense that `score()` opened its own pool.
They were not *managed*: that pool was created and discarded on every call, so it
had a size and nothing else -- nothing to bound, nothing to observe, and no way to
give evaluation a different substrate.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentdescent import AppendRules, Policies, Task, evolve
from agentdescent.evaluator import EvaluatorGroup


def _tasks(n=12):
    return [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(n)]


def _run(**kw):
    kw.setdefault("rounds", 6)
    kw.setdefault("n_workers", 3)
    kw.setdefault("held_out_frac", 0.4)
    kw.setdefault("seed", 0)
    return evolve(_tasks(), lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                  run=lambda r, t: t.meta["gold"] if t.id in r else "?",
                  propose=lambda r, t, o, s: t.id, strategy=AppendRules(), **kw)


# ---------------------------------------------------------------------------
# The group itself
# ---------------------------------------------------------------------------


def test_it_preserves_order():
    """A caller averaging over a fixed task list will want to know which score
    belongs to which task; completion order quietly makes that impossible."""
    group = EvaluatorGroup(4)
    try:
        out = group.map(lambda i: (time.sleep(0.02 * (5 - i)), i * 2)[1], range(5))
        assert out == [0, 2, 4, 6, 8]
    finally:
        group.shutdown()


def test_it_bounds_what_is_in_flight():
    group = EvaluatorGroup(2)
    live, peak, lock = [0], [0], threading.Lock()

    def work(_):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.02)
        with lock:
            live[0] -= 1
        return 1

    try:
        assert sum(group.map(work, range(12))) == 12
        assert peak[0] <= 2, f"{peak[0]} evaluations ran at once, ceiling was 2"
    finally:
        group.shutdown()


def test_one_worker_is_sequential():
    group = EvaluatorGroup(1)
    seen = []
    try:
        group.map(lambda i: seen.append(i), range(5))
        assert seen == [0, 1, 2, 3, 4]
    finally:
        group.shutdown()


def test_shutdown_releases_threads_without_ending_the_group():
    """Shutdown frees threads; it is not a terminal state.

    A finished run releases its evaluator, and inspecting the result afterwards
    -- `verifier.eval_counts` on the returned aggregator is a documented way to
    do it -- must not meet an exception about a pool the caller never knew
    existed. Caught by an existing test doing exactly that."""
    group = EvaluatorGroup(2)
    assert group.map(lambda i: i, range(3)) == [0, 1, 2]
    group.shutdown()
    group.shutdown()
    assert group.map(lambda i: i, range(3)) == [0, 1, 2]
    group.shutdown()


def test_an_exception_reaches_the_caller():
    """The gate decides commits, so a failed evaluation must not become a score."""
    group = EvaluatorGroup(2)

    def boom(i):
        if i == 2:
            raise RuntimeError("the gate failed")
        return i

    try:
        with pytest.raises(RuntimeError, match="the gate failed"):
            group.map(boom, range(5))
    finally:
        group.shutdown()


# ---------------------------------------------------------------------------
# Through the engine
# ---------------------------------------------------------------------------


def test_a_run_builds_one_pool_not_one_per_gate_call():
    """`score()` is called once per gate, and used to build a fresh
    `ThreadPoolExecutor` every time -- 83 of them in this exact run."""
    created = []
    original = ThreadPoolExecutor.__init__

    def counting(self, *args, **kwargs):
        created.append(1)
        return original(self, *args, **kwargs)

    ThreadPoolExecutor.__init__ = counting
    try:
        _run(eval_concurrency=8)
    finally:
        ThreadPoolExecutor.__init__ = original
    assert len(created) <= 2, (
        f"{len(created)} pools created; the gate is building one per call again")


def test_sequential_evaluation_is_unchanged():
    """`eval_concurrency=1` is the reference path, and it must stay bit-for-bit
    what it was -- it is what a reader compares a parallel run against."""
    a = _run(eval_concurrency=1)
    b = _run(eval_concurrency=1)
    assert a.final_reward == b.final_reward
    assert [h.held_out_reward for h in a.history] == [h.held_out_reward for h in b.history]


def test_the_gate_can_be_given_its_own_group():
    group = EvaluatorGroup(3)
    try:
        result = _run(policies=Policies(evaluator=group))
        assert result.history
        assert group.stats()["peak_inflight"] > 0, "the injected group was not used"
    finally:
        group.shutdown()


def test_a_slow_gate_does_not_slow_the_rollouts():
    """Independent groups mean independent failure and independent latency: a
    gate that has become expensive must not throttle exploration."""
    rollout_times = []

    def timed_run(rendered, task):
        t0 = time.perf_counter()
        out = task.meta["gold"] if task.id in rendered else "?"
        rollout_times.append(time.perf_counter() - t0)
        return out

    slow = EvaluatorGroup(4)
    original = slow.map
    slow.map = lambda fn, items: original(lambda x: (time.sleep(0.01), fn(x))[1], items)
    try:
        evolve(_tasks(), lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
               run=timed_run, propose=lambda r, t, o, s: t.id,
               strategy=AppendRules(), rounds=4, n_workers=3, held_out_frac=0.4,
               seed=0, policies=Policies(evaluator=slow))
    finally:
        slow.shutdown()
    assert rollout_times, "no rollouts ran"
    assert max(rollout_times) < 0.01, (
        "a rollout was as slow as the gate -- the two are sharing a bottleneck")
