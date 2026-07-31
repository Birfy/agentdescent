"""Customizable parallelism: DP / TP (and your own), driven through `evolve()`.

The parallelism *method* -- how a round of work is partitioned across workers --
is pluggable. Pick one of the paradigms, or implement the
:class:`~agentdescent.parallel.ParallelStrategy` protocol yourself.

    python -m examples.parallelism

All deterministic (no LLM). The task: fill a ``category -> rule`` artifact
correctly; a worker proposes for whichever category its task belongs to.

**This drives the real engine.** It used to hand-roll its own two drivers, and
they differed from `evolve()` in exactly the two places `evolve()` was broken:
they passed *artifact* keys to ``plan()`` where the engine passes *task ids*, and
they read ``WorkUnit.stage``, which the engine never did. So the example
demonstrated a TP that worked and a PP that worked, neither of which the engine
could actually run. Everything below now goes through `evolve()`.

PP is deliberately absent: it needs one artifact per stage and `evolve()` evolves
a single artifact, so `evolve(parallel=PipelineParallel(...))` now raises rather
than silently degrading to redundant DP. Its primitives remain usable on their
own -- see the last section.
"""

from __future__ import annotations

from agentdescent.aggregator import Aggregator
from agentdescent.evolution import KeyedRules, Task, evolve
from agentdescent.parallel import (
    DataParallel,
    PipelineChain,
    TensorParallel,
    WorkUnit,
)

CATEGORIES = ["acidbase", "kinetics", "thermo", "spectro"]
N_TASKS = 24


def category_of(task_id: str) -> str:
    """Which category a task's failure produces an edit for."""
    return CATEGORIES[int(task_id[1:]) % len(CATEGORIES)]


def tasks():
    return [Task(id=f"t{i}", prompt=f"question {i}") for i in range(N_TASKS)]


def reward(task, output):
    """Solved once the artifact carries a rule for this task's category."""
    return 1.0 if category_of(task.id) in (output or "") else 0.0


def run(rendered, task):
    return rendered


def propose(rendered, task, output, score):
    return f"{category_of(task.id)}: rule for {category_of(task.id)}"


class _Counting(Aggregator):
    """Counts proposals that actually reach the optimizer."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ingested = 0

    def ingest(self, card):
        self.ingested += 1
        super().ingest(card)


def measure(parallel, n_workers=4, rounds=8):
    """Run `evolve()` with this strategy and report what it delivered."""
    holder = {}

    def factory(ledger, verifier, audit, config, policy):
        holder["agg"] = _Counting(ledger, verifier, audit, config,
                                  staleness_policy=policy)
        return holder["agg"]

    proposals = {"n": 0}

    def counted_propose(rendered, task, output, score):
        proposals["n"] += 1
        return propose(rendered, task, output, score)

    res = evolve(tasks(), reward, run=run, propose=counted_propose,
                 strategy=KeyedRules(categories=CATEGORIES), parallel=parallel,
                 rounds=rounds, n_workers=n_workers, max_concurrency=n_workers,
                 self_verify=False, aggregator_factory=factory)
    return {
        "proposals": proposals["n"],
        "delivered": holder["agg"].ingested,
        "violations": res.outcomes().get("section-violation", 0),
        "keys": len(res.state),
        "reward": res.final_reward,
    }


# -- a user-defined strategy -------------------------------------------------


class BlockParallel:
    """Custom strategy: give each worker a contiguous *block* of the task list
    (good locality) instead of a round-robin shard."""

    name = "block"

    def plan(self, n_workers, round_index, keys):
        keys = list(keys)
        size = (len(keys) + n_workers - 1) // n_workers
        return [WorkUnit(worker=i, keys=keys[i * size:(i + 1) * size])
                for i in range(n_workers)]


def main() -> None:
    print("=== Customizable parallelism, through evolve() ===\n")
    print(f"{'strategy':>40} {'proposed':>9} {'delivered':>10} "
          f"{'out-of-section':>15} {'keys':>5} {'reward':>7}")

    variants = [
        ("DataParallel()", DataParallel()),
        ("BlockParallel()  [custom]", BlockParallel()),
        ("TensorParallel(4, keys=...)", TensorParallel(4, keys=CATEGORIES)),
        ("TensorParallel(4, keys=..., route=...)",
         TensorParallel(4, keys=CATEGORIES, route=category_of)),
    ]
    for label, strategy in variants:
        m = measure(strategy)
        print(f"{label:>40} {m['proposals']:>9} {m['delivered']:>10} "
              f"{m['violations']:>15} {m['keys']:>5} {m['reward']:>7.3f}")

    print("\nTP without `route=` hands each worker a data-parallel shard, so most of")
    print("what it proposes falls outside its own section -- rejected, and counted")
    print("in outcomes()['section-violation'] rather than silently dropped. With")
    print("`route=`, each worker only sees tasks whose edits land in its section, so")
    print("TP delivers everything DP does while keeping the merge a conflict-free")
    print("union. Implement ParallelStrategy.plan for your own method.\n")

    # -- PP: a standalone primitive, not an evolve() mode ---------------------
    chain = PipelineChain(["lit-review", "mol-engine", "hpc-submit"])
    stage_success = {"lit-review": False, "mol-engine": True, "hpc-submit": False}
    print("PipelineChain (standalone -- evolve() evolves ONE artifact, PP needs one")
    print("per stage, so evolve(parallel=PipelineParallel(...)) raises):")
    print(f"  stages          : {chain.stages}")
    print(f"  stage outcomes  : {stage_success}")
    print(f"  blame           : {chain.blame(stage_success)!r} "
          f"(earliest failing stage, not the one that surfaced the error)")
    print(f"  upstream of last: {chain.upstream_of('hpc-submit')}")


if __name__ == "__main__":
    main()
