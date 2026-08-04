"""Offline tests for the efficiency-experiment helpers (no timing assertions)."""

import threading

from examples.efficiency import (
    constant_latency,
    heavy_tailed_latency,
    run_barrier,
    run_free,
)


def test_constant_latency():
    assert constant_latency(0.25)() == 0.25


def test_heavy_tailed_latency_values_and_spread():
    lat = heavy_tailed_latency(base=0.01, spike=10.0, p_spike=0.5, seed=1)
    seen = {round(lat(), 4) for _ in range(200)}
    assert seen == {0.01, 0.1}  # only the base and the spiked value occur


def _counting_rollout():
    """A rollout that only records that it happened.

    `run_barrier` / `run_free` used to take `Worker` objects and call
    `w.run(skill, base, tasks)`. The runtimes are adapters over the general
    engine now, so the microbenchmark takes the engine's rollout shape --
    `(rendered, task) -> output` -- and this counts instead of classifying."""
    calls = {"n": 0}
    lock = threading.Lock()

    def rollout(rendered, task):
        with lock:
            calls["n"] += 1
        return ""
    return rollout, calls


def test_barrier_and_free_do_equal_work():
    rounds, n = 5, 3

    barrier, barrier_calls = _counting_rollout()
    run_barrier(barrier, rendered="", task=None, n_workers=n, rounds=rounds)
    assert barrier_calls["n"] == rounds * n

    free, free_calls = _counting_rollout()
    run_free(free, rendered="", task=None, n_workers=n, rounds=rounds)
    assert free_calls["n"] == rounds * n, (
        "the two dispatch shapes must do the same work -- the experiment is "
        "about wall-clock, and a comparison of different amounts of work is not "
        "a comparison of dispatch")
