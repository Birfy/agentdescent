# Customizable parallelism (DP / TP / CP)

> **Plugs into [`evolve`](evolution.md) via** `parallel=DataParallel()` (or
> `TensorParallel` / `ClusterParallel` / your own).

The parallelism *method* — how a round of work is partitioned across workers — is
**pluggable**. Pick a paradigm, or implement the `ParallelStrategy` protocol
yourself. This is the design's DP/TP mapping (design spec §8) made selectable.

!!! warning "PP is a standalone primitive, not an `evolve()` mode"
    `evolve()` evolves a **single** `artifact_id`; pipeline parallelism needs one
    artifact per stage. `evolve(parallel=PipelineParallel(...))` therefore
    **raises** — it used to be accepted and quietly ignored, handing every worker
    the whole task list (strictly worse than the DP default, with no signal). The
    PP machinery is still available directly as
    [`PipelineChain`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py)
    — stage ordering, `blame`, counterfactual-replay pairs.

```bash
python -m examples.parallelism
```

Source:
[`examples/parallelism.py`](https://github.com/Birfy/agentdescent/blob/main/examples/parallelism.py)
· [`agentdescent/parallel.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py).

---


## Where rollouts run

`parallel=` decides how a round's work is **split**. Where each piece then
**runs** — threads here, supervised worker processes, hosts later — is a separate
plane with its own page: [Where rollouts run](execution.md). It covers the
`rollout(spec) -> Result` seam, why a rollout has to be describable as data
before it can leave the process, and what `policies=Policies(executor=...)`
does and does not accept.

## The interface

A strategy answers one question: for round *r* with *N* workers, who works on
what?

```python
from typing import Protocol, Sequence, List
from agentdescent import WorkUnit

class ParallelStrategy(Protocol):
    name: str
    def plan(self, n_workers: int, round_index: int, keys: Sequence[str]) -> List[WorkUnit]: ...
```

A `WorkUnit(worker, keys, stage, section)` says which artifact keys (and, for
PP/TP, which stage/section) a worker owns that round.

There is one **optional** method:

```python
    def observe(self, unit: WorkUnit, task_id: str, score: float) -> None: ...
```

`evolve()` calls it after every rollout, when the strategy defines it. `plan`
alone is a pure function of `(n_workers, round_index, keys)` — enough to *shard*,
not enough to *schedule*, because a strategy had no way to learn anything from
the rollouts it dispatched. That is why UCB over task clusters (design §5.2,
L-task) lived only in the reference runtime's `TaskScheduler` and was
inexpressible here. `DataParallel` and `TensorParallel` do not define it and are
unaffected; [`ClusterParallel`](#clusterparallel-ucb-over-the-task-tail) is what
uses it.

## `ClusterParallel` — UCB over the task tail

DP shards tasks round-robin, which spends as much on a cluster that teaches
nothing as on one that still fails. `ClusterParallel` groups the tasks and
**leases whole clusters**, ordered by UCB with a difficulty filter — clusters
that are all-pass or all-fail carry no gradient and are down-weighted (the GRPO
zero-advantage argument), while clusters with little evidence keep an exploration
bonus.

```python
from agentdescent import ClusterParallel, evolve

evolve(tasks, reward, agent=agent,
       parallel=ClusterParallel(cluster_of=lambda task_id: task_id.split("-")[0]))
```

`cluster_of` maps a **task id** to a cluster id — by source, by topic, by
difficulty band, by whatever axis your tail actually runs along. A worker is
handed one cluster whole, so it sees a coherent slice of the distribution rather
than one task from each.

Two honest differences from the reference `TaskScheduler`, both in what feeds the
estimate:

* it is told a rollout's **reward**, not the before/after delta of the diff that
  rollout produced. That delta needs `self_verify`, which the directory entry
  points turn **off** because it doubles the cost of every proposal — a scheduler
  depending on it would silently stop learning exactly there. `1 - score` is the
  room a cluster still has;
* it learns per **task**, not per lease, so an estimate moves once per rollout
  rather than once per round.

The exploration constant stays the textbook `1.4`. [`DifficultyWeighted`](sampling.md)
uses `0.2` because a sweep at *task* granularity measured 1.4 as worse than
round-robin; that sweep has not been repeated at cluster granularity, where there
are far fewer arms and each carries many tasks, and transplanting a number
measured elsewhere is how a default stops meaning anything.

!!! note "It composes with `task_sampler`, it does not replace it"
    `ClusterParallel` decides **which cluster** a worker gets;
    [`task_sampler`](sampling.md) decides **which task inside it** the worker
    rolls out next. Two granularities of the same idea, and they share
    `stats.difficulty_weight`.

## The classic two: DP and TP

| Strategy | How work is partitioned | Recombination |
|---|---|---|
| **`DataParallel`** (DP) | same artifact; **tasks/keys sharded** across workers, rotating each round | diffs merged (fuse) |
| **`TensorParallel(n_sections, keys=, route=)`** (TP) | one hot artifact **split into disjoint sections**; each worker owns a section | union — conflict-free *by construction* |

`TensorParallel` needs two things beyond the section count:

* **`keys=`** — the artifact's key space, which is what the sections partition.
  `evolve()` fills it in from the strategy when the strategy declares one
  (`KeyedRules` does), so you rarely pass it by hand.
* **`route=`** — `task_id -> artifact key`, so each worker is handed exactly the
  tasks whose edits land in its own section. Optional but wanted: without it a
  worker gets a data-parallel shard and most of what it proposes falls outside its
  own section.

```python
from agentdescent import DataParallel, TensorParallel

strategy = TensorParallel(n_sections=4, keys=CATEGORIES, route=category_of)
plan = strategy.plan(n_workers=4, round_index=0, keys=task_ids)   # TASK ids
```

!!! danger "`plan()` receives task ids, not artifact keys"
    This is the distinction TP got wrong. `plan()` is handed the round's **task
    ids**; the **section** is about the *artifact*. They used to be conflated —
    `plan` filtered task ids through `section_of` while `evolve()` enforced the
    section against the artifact keys the resulting diff wrote — two unrelated key
    spaces, so **75–88% of every worker's proposals were discarded** with no
    report at all.

Running the example, through `evolve()`:

```
                                strategy  proposed  delivered  out-of-section  keys  reward
                          DataParallel()         4          4               0     4   1.000
               BlockParallel()  [custom]        14         14               0     4   1.000
             TensorParallel(4, keys=...)         8          4               4     4   1.000
  TensorParallel(4, keys=..., route=...)         4          4               0     4   1.000
```

The third row is TP being honest: half of what those workers proposed was for a
section they do not own, so it was rejected **and counted**. The fourth row routes
tasks to their section owner, so TP delivers everything DP does while keeping the
merge a conflict-free union.

## Writing your own

Implement `plan` and you have a new parallelism method — no other change:

```python
from agentdescent import WorkUnit

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
(union + a consistency reviewer that rejects out-of-section edits) and
`assign_key_sections` (a balanced **partition** of a declared key space — unlike
`section_of`, which is a hash bucket and can leave a section owning nothing). PP
provides [`PipelineChain`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/parallel.py)
(`blame` + counterfactual-replay pairs) — see [Concepts §7](concepts.md#7-parallel-paradigms-dp-tp-pp).

## What each paradigm actually enforces in `evolve()`

`WorkUnit` carries three things — `keys` (which tasks), `section` (TP) and `stage`
(PP) — and the engine has to *honour* them for the paradigm to mean anything:

| | Enforced by `evolve()` | What that means |
|---|---|---|
| **DP** | ✅ `keys` | workers take disjoint task shards; diffs merge |
| **CP** | ✅ `keys` + `observe` | workers take whole clusters, leased by UCB; every rollout's reward feeds back into the lease order |
| **TP** | ✅ `section` | a worker's diff is **rejected if it touches a key outside its section**, which is what makes the union conflict-free — and every rejection is counted as `section-violation` in [`result.outcomes()`](evolution.md) |
| **PP** | ⛔ refused | `evolve()` raises. It evolves **one** `artifact_id`, so there is no artifact chain for stages to walk |

`evolve()` also validates the TP pairing **before the first rollout**, because an
incompatible one used to be discovered a silently-dropped diff at a time:

* a strategy with **no declared key space** (`AppendRules` content-addresses its
  keys, so a proposal's section is unpredictable) is refused, naming the fix;
* `n_sections` greater than the number of keys is refused — a section owning
  nothing means a worker that can never commit. `SingleSlot` has exactly one key,
  so it cannot be tensor-parallelised at all.

!!! note "Out-of-section edits are rejected — and reported"
    A rejected TP proposal is not turned into evidence; the worker moves on and
    the section owner will propose it instead. That is the intended semantics, but
    it means a strategy whose proposals ignore sections spends rollouts for
    nothing. `result.outcomes()["section-violation"]` is how you see it — without
    that count, a TP run discarding most of its work looked exactly like one whose
    reflector had nothing useful to say, and those need opposite fixes. Pass
    `route=` to remove the waste entirely.

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
    **ignores `parallel=`** — so DP is what you get, and `TensorParallel` has no
    effect there. `max_concurrency` is likewise a
    sync-path knob (async concurrency is `n_workers`). Use the synchronous path
    when you want a specific partitioning.

    Passing either one to `evolve(asynchronous=True)` raises a `RuntimeWarning`
    naming the ignored argument, so a run never silently behaves differently from
    how it reads.

So a run picks *both* a partition (`parallel=`) and a schedule (sync
`max_concurrency` vs barrier-free `asynchronous`). See
[Parallelism & async](evolution.md#parallelism-async-the-frameworks-core).
