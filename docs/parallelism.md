# Customizable parallelism (DP / TP / PP)

> **Plugs into [`evolve`](evolution.md) via** `parallel=DataParallel()` (or
> `TensorParallel` / `PipelineParallel` / your own).

The parallelism *method* — how a round of work is partitioned across workers — is
**pluggable**. Pick one of the three classic paradigms, or implement the
`ParallelStrategy` protocol yourself. This is the design's DP/TP/PP mapping
(design spec §8) made selectable.

```bash
python -m examples.parallelism
```

Source:
[`examples/parallelism.py`](https://github.com/Birfy/agentdescent/blob/main/examples/parallelism.py)
· [`agentdescent/parallel.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py).

---

## The interface

A strategy answers one question: for round *r* with *N* workers, who works on
what?

```python
from typing import Protocol, Sequence, List
from agentdescent.parallel import WorkUnit

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
from agentdescent.parallel import DataParallel, TensorParallel, PipelineParallel

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
from agentdescent.parallel import WorkUnit

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

TP additionally provides [`TensorParallelMerge`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py)
(union + a consistency reviewer that rejects out-of-section edits), and PP
provides [`PipelineChain`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py)
(`blame` + counterfactual-replay pairs) — see [Concepts §7](concepts.md#7-parallel-paradigms-dp-tp-pp).

## What each paradigm actually enforces in `evolve()`

`WorkUnit` carries three things — `keys` (which tasks), `section` (TP) and `stage`
(PP) — and the engine has to *honour* them for the paradigm to mean anything:

| | Enforced by `evolve()` | What that means |
|---|---|---|
| **DP** | ✅ `keys` | workers take disjoint task shards; diffs merge |
| **TP** | ✅ `section` | a worker's diff is **rejected if it touches a key outside its section**, which is what makes the union conflict-free. Without that check TP was only differently-sharded DP: every worker could edit the same hot key |
| **PP** | ❌ `stage` | ignored. `evolve()` evolves **one** `artifact_id`, so there is no artifact chain for stages to walk; `PipelineParallel` there only changes task sharding |

So `parallel=TensorParallel(n_sections=4)` genuinely gives tensor parallelism
through `evolve()`. Pipeline parallelism does not: its `blame` /
counterfactual-replay machinery lives in
[`PipelineChain`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py)
and is exercised by `examples/parallelism.py` and the tests, not by the engine.
Evolving a genuine dependency chain would need `evolve()` to take several
artifacts, which it does not.

!!! note "Out-of-section edits are dropped, not merged"
    A rejected TP proposal is simply not turned into evidence — the worker moves
    on. That is the intended semantics (the section owner will propose it), but it
    does mean a strategy whose proposals ignore sections will appear to make no
    progress under TP. Have `propose` target the keys the worker owns.

## `parallel=` vs the async runtime

`parallel=` decides **how one round's tasks are split** across workers; it is
orthogonal to **whether rounds have a barrier**:

* **Within a round** — `max_concurrency=n_workers` runs the split's workers
  *concurrently* (synchronous DP; the aggregator is the barrier).
* **Across rounds** — `evolve(asynchronous=True)` / [`async_evolve`](evolution.md#the-barrier-free-runtime-async_evolve)
  removes the barrier: workers keep producing under a lag budget while one merger
  aggregates.

!!! warning "The async path does its own sharding"
    `async_evolve` shards the train tasks round-robin across its worker threads and
    **ignores `parallel=`** — so DP is what you get, and `TensorParallel` /
    `PipelineParallel` have no effect there. `max_concurrency` is likewise a
    sync-path knob (async concurrency is `n_workers`). Use the synchronous path
    when you want a specific partitioning.

    Passing either one to `evolve(asynchronous=True)` now raises a
    `RuntimeWarning` naming it. It used to be dropped in silence, which is the
    worse failure: the run *looked* tensor-parallel and was plain DP.

So a run picks *both* a partition (`parallel=`) and a schedule (sync
`max_concurrency` vs barrier-free `asynchronous`). See
[Parallelism & async](evolution.md#parallelism-async-the-frameworks-core).
