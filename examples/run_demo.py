"""RQ1 (merge vs fork) end-to-end demo.

Runs the merge-based Concordia loop and the fork/archive baseline on the same
synthetic universe and prints the learning curve plus the final comparison.

    python -m examples.run_demo
"""

from __future__ import annotations

import tempfile

from concordia.domains.router import make_task_universe
from concordia.orchestrator import Concordia, run_fork_baseline


def main() -> None:
    universe = make_task_universe(n_keywords=24, tasks_per_keyword=12, seed=7)
    rounds = 40

    with tempfile.TemporaryDirectory() as repo:
        system = Concordia(repo, universe, n_workers=6, noise=0.15,
                           refresh_interval=2, seed=1)
        history = system.run(rounds=rounds)

        print(f"{'round':>5} {'dev_acc':>8} {'stable':>8} {'commit':>7} "
              f"{'fused':>6} {'stale':>6} {'confl':>6} {'oracle':>7}")
        for h in history:
            if h.round < 6 or h.round % 4 == 0 or h.round == rounds - 1:
                print(f"{h.round:>5} {h.dev_accuracy:>8.3f} {h.stable_accuracy:>8.3f} "
                      f"{h.committed:>7} {h.fused:>6} {h.discarded_stale:>6} "
                      f"{h.conflicts_dropped:>6} {h.oracle_used:>7}")

        merge_acc = system.final_accuracy()

    fork_acc = run_fork_baseline(universe, n_workers=6, noise=0.15, rounds=rounds, seed=1)

    print("\n=== RQ1: merge vs fork (same worker budget) ===")
    print(f"Concordia (merge) held-out accuracy : {merge_acc:.3f}")
    print(f"Fork/archive best-fork accuracy     : {fork_acc:.3f}")
    print(f"merge advantage                     : {merge_acc - fork_acc:+.3f}")


if __name__ == "__main__":
    main()
