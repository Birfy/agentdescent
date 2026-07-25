# Customizable parallelism (DP / TP / PP)

The parallelism *method* — how a round of work is partitioned across workers — is
**pluggable**. Pick one of the three classic paradigms, or implement the
`ParallelStrategy` protocol yourself. This is the design's DP/TP/PP mapping
(design spec §8) made selectable.

```bash
python -m examples.parallelism
```

Source:
[`examples/parallelism.py`](https://github.com/Birfy/concordia/blob/main/examples/parallelism.py)
· [`concordia/parallel.py`](https://github.com/Birfy/concordia/blob/main/concordia/parallel.py).

---

## The interface

A strategy answers one question: for round *r* with *N* workers, who works on
what?

```python
from typing import Protocol, Sequence, List
from concordia.parallel import WorkUnit

class ParallelStrategy(Protocol):
    name: str
    def plan(self, n_workers: int, round_index: int, keys: Sequence[str]) -> List[WorkUnit]: ...
```

A `WorkUnit(worker, keys, stage, section)` says which artifact keys (and, for
PP/TP, which stage/section) a worker owns that round.

## The three built-in paradigms

| Strategy | How work is partitioned | Recombination |
|---|---|---|
| **`DataParallel`** (DP) | same artifact; **tasks/keys sharded** across workers, rotating each round | diffs merged (fuse) |
| **`TensorParallel(n_sections)`** (TP) | one hot artifact **split into disjoint sections**; each worker owns a section | union — conflict-free *by construction* |
| **`PipelineParallel(stages)`** (PP) | artifacts form a **dependency chain**; each worker drives one stage | downstream failure blames the earliest failing stage |

```python
from concordia.parallel import DataParallel, TensorParallel, PipelineParallel

strategy = TensorParallel(n_sections=4)            # or DataParallel(), or ...
plan = strategy.plan(n_workers=4, round_index=0, keys=my_keys)
```

Running the example (fill a `key -> value` artifact; all converge, TP has zero
collisions by construction):

```
              strategy  final acc  items  collisions
     DP (DataParallel)      1.000     24           0
   TP (TensorParallel)      1.000     24           0   static disjoint sections
 block (BlockParallel)      1.000     24           0

PP (PipelineParallel): pipeline complete=1.000, 3 stages,
    downstream-failure blame -> earliest failing stage = 'lit-review'
```

## Writing your own

Implement `plan` and you have a new parallelism method — no other change:

```python
from concordia.parallel import WorkUnit

class BlockParallel:
    """Give each worker a contiguous block of the key-space (good locality)."""
    name = "block"
    def plan(self, n_workers, round_index, keys):
        keys = list(keys)
        size = (len(keys) + n_workers - 1) // n_workers
        return [WorkUnit(worker=i, keys=keys[i*size:(i+1)*size]) for i in range(n_workers)]
```

`isinstance(BlockParallel(), ParallelStrategy)` is `True` structurally — pass it
anywhere a strategy is accepted.

TP additionally provides [`TensorParallelMerge`](https://github.com/Birfy/concordia/blob/main/concordia/parallel.py)
(union + a consistency reviewer that rejects out-of-section edits), and PP
provides [`PipelineChain`](https://github.com/Birfy/concordia/blob/main/concordia/parallel.py)
(`blame` + counterfactual-replay pairs) — see [Concepts §7](concepts.md#7-parallel-paradigms-dp-tp-pp).
