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

from agentdescent.aggregator import AggregatorConfig
from agentdescent.domains.router import make_task_universe
from agentdescent.orchestrator import AgentDescent


def rounds_to_converge(history, target: float = 0.999) -> str:
    """The first round whose dev accuracy reaches ``target``, or ``"-"``.

    Final accuracy saturates: the router domain reaches 1.000 under every
    configuration in this sweep, including alpha=0, so reporting only that made
    the whole experiment a flat line and the research question untestable. What a
    staleness budget actually buys is *cost to get there*, and that does vary.
    """
    for h in history:
        if h.dev_accuracy >= target:
            return str(h.round)
    return "-"


def main() -> None:
    universe = make_task_universe(seed=7)
    print("RQ2: does tolerating staleness cost anything, and does it buy anything?\n")
    print(f"{'alpha':>8} {'dev_acc':>8} {'rounds_to_1.0':>14} {'commits':>8} "
          f"{'discarded_stale':>16}")
    for alpha in [0, 1, 5, 10**6]:
        # Sweep BOTH bands. `Aggregator._alpha_for` picks alpha_head only for an
        # L1 artifact (blast_radius above governance.FAST_MAX), and the reference
        # RouterSkill is 0.2 -- so pinning alpha_tail to min(alpha, 1) meant three
        # of these four rows ran the identical configuration (effective alpha 1)
        # and the sweep silently reported one point three times.
        cfg = AggregatorConfig(alpha_head=alpha, alpha_tail=alpha)
        with tempfile.TemporaryDirectory() as repo:
            system = AgentDescent(repo, universe, n_workers=6, noise=0.1,
                               refresh_interval=3, config=cfg, seed=2)
            history = system.run(rounds=40)
            total_stale = sum(h.discarded_stale for h in history)
            commits = sum(h.committed for h in history)
            label = "inf" if alpha > 1000 else str(alpha)
            print(f"{label:>8} {system.final_accuracy():>8.3f} "
                  f"{rounds_to_converge(history):>14} {commits:>8} {total_stale:>16}")
    print("\nFinal accuracy saturates -- the domain is reachable from every")
    print("configuration here, so read the cost columns, not dev_acc. A domain")
    print("whose ceiling the system has to work for would separate them further.")


if __name__ == "__main__":
    main()
