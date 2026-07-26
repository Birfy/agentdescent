"""Asynchronous stage-orchestration demo (FlashEvolve-style).

Runs the barrier-free async runtime under each staleness policy
(Full / Guarded / Reflective) and prints throughput and staleness stats. Then
sweeps the ROLL Flash ``async_ratio`` lag budget to show the throughput vs
staleness trade-off.

    python -m examples.run_async
"""

from __future__ import annotations

import tempfile

from agentdescent.async_runtime import AsyncAgentDescent, AsyncConfig
from agentdescent.domains.router import make_task_universe
from agentdescent.staleness import get_policy


def main() -> None:
    universe = make_task_universe(seed=7)

    print("=== staleness policies (async_ratio=4) ===")
    print(f"{'policy':>11} {'dev_acc':>8} {'commits':>8} {'rollouts':>9} "
          f"{'stale':>6} {'refresh':>8} {'secs':>6}")
    for policy_name in ("full", "guarded", "reflective"):
        with tempfile.TemporaryDirectory() as repo:
            cfg = AsyncConfig(n_workers=6, async_ratio=4, noise=0.15,
                              target_accuracy=0.98, max_seconds=15.0, seed=1)
            sys = AsyncAgentDescent(repo, universe, config=cfg,
                                 staleness_policy=get_policy(policy_name))
            s = sys.run()
            print(f"{policy_name:>11} {s.final_dev_accuracy:>8.3f} {s.commits:>8} "
                  f"{s.rollouts:>9} {s.discarded_stale:>6} {s.forced_refreshes:>8} "
                  f"{s.wallclock:>6.1f}")

    print("\n=== async_ratio sweep (guarded policy) ===")
    print(f"{'async_ratio':>12} {'dev_acc':>8} {'stale':>6} {'refresh':>8} {'commits':>8}")
    for ratio in (0, 2, 4, 8):
        with tempfile.TemporaryDirectory() as repo:
            cfg = AsyncConfig(n_workers=6, async_ratio=ratio, noise=0.1,
                              target_accuracy=0.98, max_seconds=15.0, seed=2)
            sys = AsyncAgentDescent(repo, universe, config=cfg,
                                 staleness_policy=get_policy("guarded"))
            s = sys.run()
            print(f"{ratio:>12} {s.final_dev_accuracy:>8.3f} {s.discarded_stale:>6} "
                  f"{s.forced_refreshes:>8} {s.commits:>8}")


if __name__ == "__main__":
    main()
