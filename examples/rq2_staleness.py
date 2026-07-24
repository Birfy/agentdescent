"""RQ2 (staleness tolerance) sweep.

Sweeps the aggregator's staleness tolerance alpha over {0, 1, 5, infinity} and
reports final held-out accuracy and how many diffs were discarded for staleness.
The design hypothesis (design doc, RQ2): semantic-artifact space tolerates
staleness far better than parameter space, because a diff can be rebased and
cheaply re-verified where a gradient cannot.

    python -m examples.rq2_staleness
"""

from __future__ import annotations

import tempfile

from concordia.aggregator import AggregatorConfig
from concordia.domains.router import make_task_universe
from concordia.orchestrator import Concordia


def main() -> None:
    universe = make_task_universe(seed=7)
    print(f"{'alpha':>8} {'dev_acc':>8} {'discarded_stale':>16}")
    for alpha in [0, 1, 5, 10**6]:
        cfg = AggregatorConfig(alpha_head=alpha, alpha_tail=min(alpha, 1))
        with tempfile.TemporaryDirectory() as repo:
            system = Concordia(repo, universe, n_workers=6, noise=0.1,
                               refresh_interval=3, config=cfg, seed=2)
            history = system.run(rounds=40)
            total_stale = sum(h.discarded_stale for h in history)
            label = "inf" if alpha > 1000 else str(alpha)
            print(f"{label:>8} {system.final_accuracy():>8.3f} {total_stale:>16}")


if __name__ == "__main__":
    main()
