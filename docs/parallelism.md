# Customizable parallelism (DP / TP)

> **Plugs into [`evolve`](evolution.md) via** `parallel=DataParallel()` (or
> `TensorParallel` / your own).

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


## Where rollouts run — the execution plane

`evolve()` runs rollouts in threads, and for this workload that is the measured
right answer: a rollout is almost entirely waiting on a model, and the numbers
above are **7.1x** on I/O against **1.0x** on CPU. Nothing here is an attempt to
make rollouts faster with processes.

What processes buy is different and unavailable any other way:

* **fault isolation** — the code being evolved is model-authored, so a segfault
  or an OOM is ordinary. In a thread it takes the run with it;
* **capacity beyond one machine**, and **heterogeneous workers**, later.

```python
from agentdescent import ProcessExecutor, Ref, RolloutSpec, ThreadExecutor
```

### Work has to be describable as data first

The wall is not the executor. Measured on this package:

| passed to `evolve()` | crosses a process? |
|---|---|
| `Task`, `Diff`, `EvidenceCard`, `AppendRules` | yes |
| `rewards.last_number()` | **no** — `Can't pickle local object` |
| `reflector(model)`, `LLMAgent(...)` | **no** |
| `run=lambda rendered, task: ...` | **no** |

Every factory in the package returns a closure, and so does every example in
these docs. So a process pool fails on the first submit no matter how good the
pool is, and the fix is a way to *describe* the work:

```python
Ref("agentdescent.rewards:last_number", {"gold_key": "gold"})
Ref("agentdescent.runners:code_runner", {"entrypoint": ["python", "main.py"]})
Ref("agentdescent.evolution:reflector",              # references nest
    {"complete": Ref("agentdescent.agents:claude", {"model": "..."})})
```

The worker resolves these against **its own** copy of the code. `cloudpickle`
would send the closure instead, which is less work here and worse afterwards: it
executes the sending process's code on the receiving side, so a version skew
becomes a wrong answer rather than an import error.

`resolve()` runs whatever it imports, so across a boundary it *is* the boundary:
targets are restricted to an allowlist (`agentdescent.*` by default) and config
to JSON scalars. Widen it deliberately — that is the moment to think about who
can write to the queue.

### Why not `ProcessPoolExecutor`

1. **One worker dying abruptly breaks the whole pool** (`BrokenProcessPool`) and
   every in-flight task with it. Fault isolation built on something that fails as
   a unit is not fault isolation — and this is the entire reason for processes
   here;
2. `max_tasks_per_child` is 3.11+; this package supports 3.9;
3. it has no notion of a sandbox, so no way to say "this needs an environment
   with fingerprint X, wait for one";
4. its default start method is `fork` on Linux, and this engine is threaded — a
   `fork` from a process holding locks in other threads produces a child holding
   locks nothing will release.

`ProcessExecutor` is persistent workers, a bounded task queue and a supervisor
that decides on its own when a worker is gone.

!!! warning "`spawn` re-imports `__main__`"
    A script that builds a `ProcessExecutor` at module level builds one again in
    every child, which builds one in every grandchild. The machine fills with
    processes and nothing reports, so it reads as *slow* rather than as a fault.
    Put the run behind `if __name__ == "__main__":`. Building one inside a worker
    is refused outright.

### Status

The executors are usable directly and are covered by tests, including the
assertion `ProcessPoolExecutor` cannot pass: killing one of three workers leaves
the others producing and re-dispatches the dead one's task.

They are **not yet wired into `evolve()`'s round loop**. Doing that now would
mean writing it twice, once in each of the two loops that
[#57](https://github.com/Birfy/agentdescent/issues/57) exists to unify — so it
waits for that, rather than being duplicated and then de-duplicated.

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

## The three built-in paradigms

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
