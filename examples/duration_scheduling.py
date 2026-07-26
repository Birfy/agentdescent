"""Duration-aware asynchronous scheduling.

Estimate a rollout's cost from the task's size, then schedule on that estimate.
Three parts, all deterministic (no LLM):

1. **Online estimation** -- a `DurationEstimator` learns ``seconds ≈ b + m·length``
   from observed rollouts; prediction error shrinks as it calibrates.
2. **Makespan-minimizing dispatch** -- with heavy-tailed task lengths, dispatch
   the longest-estimated tasks first (LPT) instead of round-robin, and the
   parallel makespan drops toward the optimal ``total/N``.
3. **Straggler checkpointing** -- in the live async runtime, a rollout that
   overruns its estimate is checkpointed to the ResumeQueue (partial rollout)
   instead of blocking a worker.

    python -m examples.duration_scheduling
"""

from __future__ import annotations

import random
import tempfile

from agentdescent.async_runtime import AsyncAgentDescent, AsyncConfig
from agentdescent.domains.router import make_task_universe
from agentdescent.scheduler import DurationEstimator, fifo_makespan, lpt_schedule


def heavy_tailed_lengths(n, seed):
    rng = random.Random(seed)
    return [int(rng.lognormvariate(6.0, 0.9)) for _ in range(n)]   # mean ~600, heavy tail


# -- Experiment 1: online duration estimation --------------------------------


def experiment_estimation(seed=1):
    print("=== 1. Online duration estimation (seconds ≈ b + m·length) ===")
    true_b, true_m = 0.05, 0.0008
    rng = random.Random(seed)
    est = DurationEstimator()
    print(f"{'rollouts':>9} {'pred MAE (s)':>13} {'learned b':>11} {'learned m':>11}")
    window = []
    for i in range(1, 301):
        length = int(rng.lognormvariate(6.0, 0.9))
        actual = max(0.0, true_b + true_m * length + rng.gauss(0, 0.02))
        window.append(abs(est.estimate(length) - actual))
        est.observe(length, actual)
        if i in (10, 30, 100, 300):
            mae = sum(window) / len(window)
            b, m = est.params
            print(f"{i:>9} {mae:>13.3f} {b:>11.3f} {m:>11.5f}")
            window = []
    print(f"(ground truth: b={true_b}, m={true_m})\n")


# -- Experiment 2: makespan -- LPT vs round-robin ----------------------------


def experiment_makespan(seed=2, n_workers=4, n_tasks=40):
    print("=== 2. Duration-aware dispatch: makespan on 4 workers ===")
    lengths = heavy_tailed_lengths(n_tasks, seed)
    weights = [0.05 + 0.0008 * L for L in lengths]   # estimated durations
    fifo = fifo_makespan(weights, n_workers)
    _, lpt = lpt_schedule(weights, n_workers)
    lower_bound = sum(weights) / n_workers
    print(f"{n_tasks} tasks (heavy-tailed length), {n_workers} workers\n")
    print(f"{'dispatch':>16} {'makespan (s)':>13} {'vs optimal':>11}")
    print(f"{'round-robin':>16} {fifo:>13.2f} {fifo / lower_bound:>10.2f}x")
    print(f"{'LPT (by estimate)':>16} {lpt:>13.2f} {lpt / lower_bound:>10.2f}x")
    print(f"{'optimal lower bound':>16} {lower_bound:>13.2f}\n")
    print(f"LPT speedup over round-robin: {fifo / lpt:.2f}x "
          f"(dispatching the tail early)\n")


# -- Experiment 3: straggler checkpointing in the async runtime --------------


def experiment_stragglers(seed=3):
    print("=== 3. Straggler checkpointing in the live async runtime ===")
    rng = random.Random(seed)
    lock = __import__("threading").Lock()

    def latency():                      # mostly short, 15% long-tail spikes
        with lock:
            spike = rng.random() < 0.15
        return 0.04 if spike else 0.004

    universe = make_task_universe(seed=7)
    cfg = AsyncConfig(n_workers=4, noise=0.1, worker_pause=0.0, async_ratio=10_000,
                      target_accuracy=2.0, max_seconds=2.0, duration_timeout_factor=3.0)
    with tempfile.TemporaryDirectory() as repo:
        sys = AsyncAgentDescent(repo, universe, config=cfg, estimator=DurationEstimator())
        for w in sys.workers:
            w.rollout_latency = latency
        stats = sys.run()
    b, m = sys.estimator.params
    print(f"rollouts={stats.rollouts}, learned base≈{b:.3f}s, "
          f"stragglers checkpointed to resume queue={stats.stragglers_checkpointed}")
    print("(overrunning rollouts are checkpointed instead of blocking a worker)\n")


def main() -> None:
    experiment_estimation()
    experiment_makespan()
    experiment_stragglers()


if __name__ == "__main__":
    main()
