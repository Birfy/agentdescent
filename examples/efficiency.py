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


def _slow_rollout(latency):
    """The domain's rollout, plus the latency a real one would have.

    `time.sleep` releases the GIL, which is the whole point: it stands in for the
    network wait an agent rollout actually is, so the overlap being measured here
    is the overlap a real workload gets."""
    from agentdescent.domains.router import router_run

    def rollout(rendered, task):
        time.sleep(latency())
        return router_run(rendered, task)
    return rollout


def _build(universe, n_workers, latency, seed, **cfg_kw):
    cfg = AsyncConfig(n_workers=n_workers, noise=0.1, seed=seed, **cfg_kw)
    repo = tempfile.mkdtemp(prefix="agentdescent-eff-")
    return AsyncAgentDescent(repo, universe, config=cfg,
                             rollout=_slow_rollout(latency))


# -- Experiment 1: parallel throughput scaling -------------------------------


def experiment_parallel(universe, seconds=2.0):
    print("=== Experiment 1: parallel throughput scaling (async runtime) ===")
    print(f"per-rollout latency = 6ms, wall-clock window = {seconds}s\n")
    print(f"{'workers':>8} {'rollouts':>9} {'rollouts/s':>11} {'speedup':>8} {'efficiency':>11}")
    base_rate = None
    for n in (1, 2, 4, 8):
        # target unreachable -> run the full window; huge async_ratio + stall
        # patience keep workers off the ledger so we measure pure rollout rate.
        # `self_verify=False`: this counts dispatch, and a self-verify rollout
        # is a second sleep the counter does not see. The reference loop got its
        # before/after delta free, so leaving it on would halve a number that is
        # supposed to be comparable with the published table.
        sys = _build(universe, n, constant_latency(0.006), seed=1,
                     async_ratio=10_000, stall_patience=10_000, self_verify=False,
                     target_accuracy=2.0, max_seconds=seconds)
        stats = sys.run()
        # The window the experiment asked for, not a measured wall-clock. This
        # is what the description says ("a fixed wall-clock window"), and a
        # measured one also counts setup and the shutdown grace -- fixed costs
        # that fall hardest on the low-worker rows and read as superlinear
        # speedup. Measured both ways: 8 workers came out at 8.2-9.4x against
        # ~8.1x here.
        rate = stats.rollouts / seconds
        if base_rate is None:
            base_rate = rate
        speedup = rate / base_rate
        print(f"{n:>8} {stats.rollouts:>9} {rate:>11.0f} {speedup:>8.2f} "
              f"{speedup / n:>11.2f}")
    print()


# -- Experiment 2: async vs synchronous barrier ------------------------------


def run_barrier(rollout, rendered, task, n_workers, rounds):
    """Barrier: each round runs N rollouts concurrently, then waits for the
    slowest before the next round -- wall-clock = sum of per-round max latency."""
    start = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for _ in range(rounds):
            futs = [ex.submit(rollout, rendered, task) for _ in range(n_workers)]
            for f in futs:              # <-- barrier: block on the slowest worker
                f.result()
    return time.time() - start


def run_free(rollout, rendered, task, n_workers, rounds):
    """No barrier: each worker runs `rounds` rollouts back-to-back, never
    waiting for the others -- wall-clock = the busiest worker's own total."""
    def loop():
        for _ in range(rounds):
            rollout(rendered, task)
    threads = [threading.Thread(target=loop) for _ in range(n_workers)]
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
    # Dispatch shape only: one rendered artifact, one cluster, N concurrent
    # rollouts. Nothing here touches the ledger, which is the point -- the
    # question is what the *barrier* costs, not what a merge costs.
    from agentdescent.domains.router import RouterStrategy, cluster_tasks

    rollout = _slow_rollout(heavy_tailed_latency(0.004, 12.0, 0.15, seed=7))
    rendered = RouterStrategy().render({})
    task = cluster_tasks(universe, n_clusters=max(2, n_workers))[0][0]
    total = rounds * n_workers

    bt = run_barrier(rollout, rendered, task, n_workers, rounds)
    ft = run_free(rollout, rendered, task, n_workers, rounds)

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
