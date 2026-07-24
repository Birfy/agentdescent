"""Offline tests for the efficiency-experiment helpers (no timing assertions)."""

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


class _CountingWorker:
    def __init__(self):
        self.calls = 0

    def run(self, skill, base, tasks):
        self.calls += 1
        return None


def test_barrier_and_free_do_equal_work():
    rounds, n = 5, 3
    bw = [_CountingWorker() for _ in range(n)]
    run_barrier(bw, skill=None, base=None, tasks=None, rounds=rounds)
    assert sum(w.calls for w in bw) == rounds * n

    fw = [_CountingWorker() for _ in range(n)]
    run_free(fw, skill=None, base=None, tasks=None, rounds=rounds)
    assert sum(w.calls for w in fw) == rounds * n
    assert all(w.calls == rounds for w in fw)  # each worker did exactly `rounds`
