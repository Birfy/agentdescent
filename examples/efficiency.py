"""Experiments: parallel efficiency and asynchronous efficiency.

Two things the framework claims to buy you, measured in wall-clock on the
deterministic router domain with an injected per-rollout latency (models the
real cost of a rollout -- tool calls, HPC queues, LLM latency -- which is where
parallelism and asynchrony actually pay off).

    python -m examples.efficiency

**Experiment 1 -- parallel throughput scaling.** Run the async runtime with
N = 1, 2, 4, 8 workers for a fixed wall-clock and count rollouts. Throughput
(rollouts/sec) should scale with N; speedup = rate(N)/rate(1), efficiency =
speedup/N shows how close to linear it stays before lock contention bites.

**Experiment 2 -- async vs synchronous barrier, under a latency tail.** Isolates
the rollout-scheduling discipline: run the *same* fixed rollout budget two ways
-- (a) a synchronous round barrier where each round of N rollouts must wait for
the *slowest* before the next round starts, vs (b) barrier-free workers that
never wait for each other. With heavy-tailed latency the barrier pays
``E[max of N]`` per round while async pays ``E[latency]``; the ratio is the
tail-hiding speedup (the async-RL "partial rollout / no barrier" claim).
"""

from __future__ import annotations

import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from agentdescent.async_runtime import AsyncAgentDescent, AsyncConfig
from agentdescent.domains.router import make_task_universe
from agentdescent.ledger import Ledger


def constant_latency(seconds: float):
    return lambda: seconds


def heavy_tailed_latency(base: float, spike: float, p_spike: float, seed: int):
    rng = random.Random(seed)
    lock = __import__("threading").Lock()

    def lat():
        with lock:
            hit = rng.random() < p_spike
        return base * (spike if hit else 1.0)
    return lat


def _build(universe, n_workers, latency, seed, **cfg_kw):
    cfg = AsyncConfig(n_workers=n_workers, noise=0.1, worker_pause=0.0,
                      aggregator_interval=0.001, seed=seed, **cfg_kw)
    repo = tempfile.mkdtemp(prefix="agentdescent-eff-")
    sys = AsyncAgentDescent(repo, universe, config=cfg)
    for w in sys.workers:
        w.rollout_latency = latency
    return sys


# -- Experiment 1: parallel throughput scaling -------------------------------


def experiment_parallel(universe, seconds=2.0):
    print("=== Experiment 1: parallel throughput scaling (async runtime) ===")
    print(f"per-rollout latency = 6ms, wall-clock window = {seconds}s\n")
    print(f"{'workers':>8} {'rollouts':>9} {'rollouts/s':>11} {'speedup':>8} {'efficiency':>11}")
    base_rate = None
    for n in (1, 2, 4, 8):
        # target unreachable -> run the full window; huge async_ratio + stall
        # patience keep workers off the ledger so we measure pure rollout rate.
        sys = _build(universe, n, constant_latency(0.006), seed=1,
                     async_ratio=10_000, stall_patience=10_000,
                     target_accuracy=2.0, max_seconds=seconds)
        stats = sys.run()
        rate = stats.rollouts / stats.wallclock
        if base_rate is None:
            base_rate = rate
        speedup = rate / base_rate
        print(f"{n:>8} {stats.rollouts:>9} {rate:>11.0f} {speedup:>8.2f} "
              f"{speedup / n:>11.2f}")
    print()


# -- Experiment 2: async vs synchronous barrier ------------------------------


def _rollout_once(worker, skill, base, tasks):
    worker.run(skill, base, tasks)  # sleeps (latency) + does the cheap work


def run_barrier(workers, skill, base, tasks, rounds):
    """Barrier: each round runs N rollouts concurrently, then waits for the
    slowest before the next round -- wall-clock = sum of per-round max latency."""
    n = len(workers)
    start = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        for _ in range(rounds):
            futs = [ex.submit(_rollout_once, w, skill, base, tasks) for w in workers]
            for f in futs:              # <-- barrier: block on the slowest worker
                f.result()
    return time.time() - start


def run_free(workers, skill, base, tasks, rounds):
    """No barrier: each worker runs `rounds` rollouts back-to-back, never
    waiting for the others -- wall-clock = the busiest worker's own total."""
    def loop(w):
        for _ in range(rounds):
            _rollout_once(w, skill, base, tasks)
    threads = [threading.Thread(target=loop, args=(w,)) for w in workers]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.time() - start


def experiment_async(universe, n_workers=4, rounds=40):
    print("=== Experiment 2: async (no barrier) vs synchronous barrier ===")
    print(f"N = {n_workers} workers, heavy-tailed latency (4ms base, 12x spike @15%), "
          f"{rounds} rounds = {rounds * n_workers} rollouts each way\n")
    sys = _build(universe, n_workers, heavy_tailed_latency(0.004, 12.0, 0.15, seed=7),
                 seed=2, async_ratio=10_000, target_accuracy=2.0, max_seconds=1)
    snap = sys.ledger.snapshot(Ledger.DEV)
    skill, base, tasks = snap.get(sys.skill_id), snap.version, sys.scheduler.select().tasks
    total = rounds * n_workers

    bt = run_barrier(sys.workers, skill, base, tasks, rounds)
    ft = run_free(sys.workers, skill, base, tasks, rounds)

    print(f"{'mode':>16} {'wall-clock':>11} {'rollouts/s':>11} {'utilization':>12}")
    print(f"{'sync barrier':>16} {bt:>10.2f}s {total / bt:>11.0f} {ft / bt:>11.0%}")
    print(f"{'async (no barrier)':>16} {ft:>10.2f}s {total / ft:>11.0f} {'100%':>12}")
    print(f"\nasync speedup: {bt / ft:.2f}x  "
          f"(the barrier idles fast workers waiting for the tail every round)")
    print()


def main() -> None:
    universe = make_task_universe(seed=7)
    experiment_parallel(universe)
    experiment_async(universe)


if __name__ == "__main__":
    main()
