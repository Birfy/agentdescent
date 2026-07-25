"""Customizable parallelism: DP / TP / PP (and your own).

The parallelism *method* -- how a round of work is partitioned across workers --
is pluggable. Pick one of the three classic paradigms, or implement the
:class:`~concordia.parallel.ParallelStrategy` protocol yourself.

    python -m examples.parallelism

All deterministic (no LLM). The toy task: fill a ``key -> value`` artifact
correctly; workers fix the keys they are assigned each round.
"""

from __future__ import annotations

from concordia.parallel import (
    DataParallel,
    PipelineParallel,
    TensorParallel,
    WorkUnit,
    section_of,
)

GOLD = {f"kw{i:02d}": f"label{i % 5}" for i in range(24)}
KEYS = list(GOLD)


# -- a shared-artifact driver (works for DP and TP) --------------------------


def run_shared(strategy, n_workers=4, rounds=8):
    """One shared artifact; each round the strategy assigns keys to workers, who
    fix any wrong ones. Tracks cross-worker collisions (two workers editing the
    same key in one round)."""
    state = {}
    collisions = 0
    for r in range(rounds):
        units = strategy.plan(n_workers, r, KEYS)
        edited_by = {}
        for u in units:
            for k in u.keys:
                if state.get(k) != GOLD[k]:
                    if k in edited_by:
                        collisions += 1
                    edited_by[k] = u.worker
        for k in edited_by:
            state[k] = GOLD[k]
    acc = sum(1 for k in KEYS if state.get(k) == GOLD[k]) / len(KEYS)
    return acc, len(state), collisions


# -- a pipeline driver (PP) --------------------------------------------------


def run_pipeline(strategy, n_workers=3, rounds=8):
    """A chain of stage-artifacts; a key is 'done' only if every stage is
    correct. Workers drive stages; downstream failure blames the earliest stage."""
    stages = list(strategy.stages)
    gold_stage = [{k: f"{s}:{GOLD[k]}" for k in KEYS} for s in stages]
    state = [dict() for _ in stages]
    for r in range(rounds):
        for u in strategy.plan(n_workers, r, KEYS):
            st = u.stage
            for k in u.keys:
                if state[st].get(k) != gold_stage[st][k]:
                    state[st][k] = gold_stage[st][k]
    done = sum(1 for k in KEYS if all(state[i].get(k) == gold_stage[i][k]
                                      for i in range(len(stages)))) / len(KEYS)
    # blame demonstration: a key correct downstream but broken upstream
    chain = strategy.chain()
    example = {stages[0]: False, stages[1]: True, stages[2]: True}
    return done, chain.blame(example)


# -- a user-defined strategy -------------------------------------------------


class BlockParallel:
    """Custom strategy: give each worker a contiguous *block* of the key-space
    (good locality) instead of a round-robin shard."""

    name = "block"

    def plan(self, n_workers, round_index, keys):
        keys = list(keys)
        size = (len(keys) + n_workers - 1) // n_workers
        return [WorkUnit(worker=i, keys=keys[i * size:(i + 1) * size])
                for i in range(n_workers)]


def main() -> None:
    print("=== Customizable parallelism (choose the method, or write one) ===\n")
    print(f"{'strategy':>22} {'final acc':>10} {'items':>6} {'collisions':>11}")

    for strat in (DataParallel(), TensorParallel(n_sections=4), BlockParallel()):
        acc, n, col = run_shared(strat)
        note = "static disjoint sections" if isinstance(strat, TensorParallel) else ""
        print(f"{strat.name + ' (' + type(strat).__name__ + ')':>22} "
              f"{acc:>10.3f} {n:>6} {col:>11}  {note}")

    pp = PipelineParallel(stages=["lit-review", "mol-engine", "hpc-submit"])
    done, blamed = run_pipeline(pp)
    print(f"\nPP (PipelineParallel): pipeline complete={done:.3f}, "
          f"3 stages, downstream-failure blame -> earliest failing stage = '{blamed}'")

    print("\nTP has 0 collisions by construction (disjoint sections); DP/block "
          "rotate or block the key-space. Implement ParallelStrategy.plan for your own.")


if __name__ == "__main__":
    main()
