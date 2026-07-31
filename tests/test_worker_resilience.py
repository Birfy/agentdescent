"""How a worker responds to backend failures depends on whether it ever worked.

The two situations wear the same exception and want opposite responses:

* never succeeded -> misconfiguration (wrong key, dead endpoint). Retire fast so
  the run ends loudly rather than burning its whole budget.
* succeeded, then failing -> a transient (a rate limit). Retiring is actively
  wrong: every worker shares one backend, so shedding workers cannot relieve
  throttling and only guarantees the run dies.
"""
import itertools
import threading
import warnings

import pytest

from agentdescent.evolution import SingleSlot, Task, evolve

TASKS = [Task(id=str(i), prompt=str(i), meta={"gold": str(i)}) for i in range(8)]


def _evolve(run, **kw):
    return evolve(TASKS, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                  run=run, propose=lambda *a: None,
                  strategy=SingleSlot(initial_value="v"), asynchronous=True,
                  n_workers=3, rounds=40, held_out_frac=0.5, **kw)


def test_a_backend_that_never_works_ends_the_run_fast():
    def run(rendered, task):
        raise RuntimeError("401 unauthorized")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _evolve(run, max_seconds=60.0)
    assert res.error is not None and "401" in res.error
    assert res.retired_workers == 3


def test_an_intermittent_backend_does_not_kill_the_run():
    """The regression: a 429 storm used to retire every worker in seconds."""
    counter, lock = itertools.count(), threading.Lock()

    def run(rendered, task):
        with lock:
            i = next(counter)
        if i % 3:                      # 2 of every 3 rollouts fail
            raise RuntimeError("429 rate limited")
        return task.prompt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _evolve(run, max_seconds=12.0)
    # It must survive on the successes rather than retiring into silence.
    assert res.retired_workers == 0, "workers retired over a transient backend"
    # At this failure rate the *final measurement* may legitimately be impossible
    # (P(all held-out tasks succeed in one pass) ~ 1%). That is a different thing
    # from the run dying, and the report has to distinguish them.
    if res.error is not None:
        assert "final held-out scoring failed" in res.error, res.error


def test_a_worker_that_dies_after_working_warns_rather_than_going_quiet():
    """Not retiring must not mean not telling anyone."""
    state, lock = {"ok": True}, threading.Lock()

    def run(rendered, task):
        with lock:
            if state["ok"]:
                state["ok"] = False    # exactly one success, then the backend dies
                return task.prompt
        raise RuntimeError("500 backend gone")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _evolve(run, max_seconds=10.0, max_worker_errors=2)
    assert any("backing off, not retiring" in str(x.message) for x in w)


def test_retired_workers_survives_save_and_load():
    import os, tempfile
    from agentdescent.evolution import EvolutionResult

    def run(rendered, task):
        raise RuntimeError("dead")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _evolve(run, max_seconds=60.0)
    path = tempfile.mktemp(suffix=".json")
    try:
        res.save(path)
        assert EvolutionResult.load(path).retired_workers == res.retired_workers
    finally:
        os.path.exists(path) and os.unlink(path)


def test_the_merger_survives_a_transient_too():
    """It calls the backend every sweep, so it needs the tolerance workers have.

    Measured before the fix: against a backend refusing 1 call in 3, the run ended
    with 0 sweeps while the workers were still healthy -- one blanket try/except
    around the merger loop made it a single point of failure.
    """
    # Fail only on the merger's thread, to isolate merger resilience from the
    # workers' own fail-fast path. `propose` is the worker-exclusive call: the
    # merger *does* call run/reward, because scoring held-out runs the agent.
    worker_threads, lock, merger_calls = set(), threading.Lock(), itertools.count()

    def propose(rendered, task, output, reward_value):
        with lock:
            worker_threads.add(threading.get_ident())
        return f"tweak-{task.id}"

    def reward(task, output):
        with lock:
            is_merger = (worker_threads
                         and threading.get_ident() not in worker_threads)
            n = next(merger_calls) if is_merger else -1
        if is_merger and n < 4:      # an outage on the merger path that recovers
            raise RuntimeError("503 transient")
        return 0.5                   # never perfect, so propose keeps firing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve(TASKS, reward, run=lambda rendered, t: "wrong", propose=propose,
                     strategy=SingleSlot(initial_value="v"), asynchronous=True,
                     n_workers=2, rounds=30, held_out_frac=0.5, max_seconds=15.0)
    # Non-vacuity: the merger must actually have hit the outage and got past it,
    # otherwise this passes without ever exercising the recovery path.
    assert next(merger_calls) > 4, "the merger never reached the outage"
    assert res.history, "merger produced no sweeps against a transient backend"


def test_a_caller_bug_in_the_merger_still_propagates():
    """The merger's blanket except used to absorb ContractError as a backend outage."""
    from agentdescent.evolvable import ContractError

    class BadAggregator:
        def __init__(self, *a, **kw): pass
        def ingest(self, card): pass
        def step(self): raise ContractError("aggregator is broken")

    with pytest.raises(ContractError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evolve(TASKS, lambda t, o: 0.5, run=lambda r, t: t.prompt,
                   propose=lambda *a: "tweak",
                   strategy=SingleSlot(initial_value="v"), asynchronous=True,
                   n_workers=2, rounds=20, held_out_frac=0.5, max_seconds=10.0,
                   aggregator_factory=lambda *a, **kw: BadAggregator())


# --- the synchronous path, which had the same disease in a worse form -----------

def _sync(run, reward=lambda t, o: 0.5, **kw):
    return evolve(TASKS, reward, run=run, propose=lambda *a: "tweak",
                  strategy=SingleSlot(initial_value="v"), rounds=20,
                  n_workers=3, max_concurrency=3, held_out_frac=0.5, **kw)


def test_one_transient_does_not_end_a_sync_run():
    """It used to end it outright: `f.result()` re-raised the first worker's
    exception and the round loop broke. Measured: one 429 on call 5 turned a
    20-round run into 0 rounds."""
    counter, lock = itertools.count(), threading.Lock()

    def run(rendered, task):
        with lock:
            i = next(counter)
        if i == 5:
            raise RuntimeError("429 rate limited")
        return "wrong"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _sync(run)
    assert len(res.history) == 20, f"ran only {len(res.history)} rounds"
    assert res.error is None


def test_a_dead_backend_still_ends_a_sync_run_quickly():
    """Tolerating transients must not mean burning the whole budget on a dead key."""
    def run(rendered, task):
        raise RuntimeError("401 unauthorized")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _sync(run)
    assert res.error is not None and "401" in res.error
    assert len(res.history) == 0


def test_a_failing_held_out_score_does_not_escape_the_driver():
    """Per-round scoring runs the agent, so it is a backend call too.

    It sat outside the round's error handling and raised straight out of evolve(),
    discarding everything already committed.
    """
    counter, lock = itertools.count(), threading.Lock()

    def reward(task, output):
        with lock:
            i = next(counter)
        if 20 < i < 24:            # a blip during a round's held-out measurement
            raise RuntimeError("503 transient")
        return 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _sync(lambda rendered, t: "wrong", reward=reward)   # must not raise
    assert res.history, "the run produced nothing despite a recoverable blip"


def test_a_caller_bug_still_propagates_from_a_sync_worker():
    """Tolerating backend failures must not swallow contract violations."""
    from agentdescent.evolution import RewardContractError
    with pytest.raises(RewardContractError):
        _sync(lambda rendered, t: "wrong", reward=lambda t, o: "not a number")
