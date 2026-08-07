# Where rollouts run — the execution plane

> **Plugs into [`evolve`](evolution.md) via** `policies=Policies(executor=...)`.

[Parallelism](parallelism.md) decides **how a round's work is split**. This page
is about **where each piece of it then runs** — the seam between the engine's
control plane and the process that performs a rollout.

`evolve()` runs rollouts in threads, and for this workload that is the measured
right answer: a rollout is almost entirely waiting on a model, and
[the numbers](efficiency.md#threads-and-the-gil-is-this-really-parallel) are **5.8x** on
I/O against **1.1x** on CPU. Nothing here is an attempt to make rollouts faster
with processes.

What processes buy is different and unavailable any other way:

* **fault isolation** — the code being evolved is model-authored, so a segfault
  or an OOM is ordinary. In a thread it takes the run with it;
* **capacity beyond one machine**, and **heterogeneous workers**, later.

```python
from agentdescent import ProcessExecutor, Ref, RolloutSpec, ThreadExecutor
```

## The seam is one rollout, not one round

```
      round body                          executor
  ┌────────────────────┐            ┌────────────────────┐
  │ pick a task        │            │                    │
  │ render the artifact│──spec────► │  run(rendered,task)│
  │                    │            │  reward(task, out) │
  │ output → diff      │◄──Result───│                    │
  │ hand to aggregator │            │                    │
  └────────────────────┘            └────────────────────┘
     stays in-process                  free to move
```

Everything either side of the rollout — choosing the task, turning an output into
a diff, ingesting evidence — reads or writes state that has to stay in this
process. The rollout does not. So the seam is `rollout(spec) -> Result`, one at a
time, and that is what lets `run` move elsewhere without moving the control plane
with it.

`Result` reports rather than raises: one bad rollout is evidence, not the end of
a run. It carries a `kind` (`ok` · `model` · `infrastructure` · `caller`) because
the round body has to tell a backend transient — which must not stop the run —
from a broken caller contract, which must.

## Work has to be describable as data first

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

!!! note "Secrets do not travel in a spec"
    A spec is pickled, logged, cached and journalled, so one carrying an API key
    leaks it into all four. `SandboxSpec` carries variable **names**; the values
    are read on the far side.

## What `evolve()` can and cannot hand an executor

`evolve()` builds a `ThreadExecutor` and gives it the `run` and `reward` you
passed, **as callables**. In this process nothing needs describing, and resolving
a `Ref` per rollout would rebuild a model client every time.

That is also the limit. A closure has no name, so `evolve()` cannot turn your
`run=` into a `Ref` — the spec it builds carries a reference that *raises when
resolved* and says why. So:

| `policies=Policies(executor=...)` | what happens |
|---|---|
| omitted | a `ThreadExecutor` sized from `eval_concurrency` |
| a `ThreadExecutor` you built | accepted; `evolve()`'s `run`/`reward` are attached to it and **win** over any passed to its constructor |
| anything without `attach_actors` (e.g. `ProcessExecutor`) | **refused at build time**, naming the fix |
| any of the above under `async_evolve()` / `asynchronous=True` | **refused**: the barrier-free loop has no executor seam |

!!! warning "`async_evolve()` does not take an executor"
    Its worker calls `eng.run` directly — there is no seam to route through yet.
    Both engines shared one list of honoured `Policies` fields, so `executor` was
    declared supported for a loop that never read it: accepted, then dropped. That
    is the single outcome [`require_supported`](evolution.md) exists to prevent,
    and it got sharper once a supplied executor started working under `evolve()`,
    because flipping `asynchronous=True` would silently stop honouring it. It now
    raises `NotImplementedError` naming the field.

!!! warning "Why the refusal, rather than a best effort"
    It used to be a best effort, and the spec named `agents:echo` /
    `rewards:contains` as stand-ins. Every rollout then failed on an argument-count
    mismatch, the gate still scored the artifact from `evolve()`'s own actors, and
    the run returned `rollouts=0` with a plausible `final_reward` and **no
    exception**. A wrong answer in the shape of a right one is worse than a
    refusal.

To run rollouts in processes today, describe them yourself and drive the
executor directly:

```python
if __name__ == "__main__":                      # required — see the warning below
    specs = [RolloutSpec(rendered=artifact.render(), task=t,
                         run=Ref("agentdescent.runners:code_runner",
                                 {"entrypoint": ["python", "main.py"]}),
                         reward=Ref("agentdescent.rewards:last_number"))
             for t in tasks]
    with_executor = ProcessExecutor(4)
    for result in with_executor.map_rollouts(specs):
        ...
    with_executor.shutdown()
```

## Why not `ProcessPoolExecutor`

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
that decides on its own when a worker is gone. Four decisions in it are worth
knowing, because each replaced something that looked reasonable:

| decision | why |
|---|---|
| `spawn`, never `fork` | the engine is threaded; a forked child inherits locks nothing will release, and the symptom is an occasional hang rather than an error |
| liveness is `is_alive()`, not a heartbeat | a worker inside a rollout cannot answer, and a rollout can legitimately take ten minutes. The heartbeat that remains catches *alive but wedged*, so its timeout is **longer** than any real rollout |
| results queue unbounded, task queue bounded | back-pressure belongs on work going in. A full results queue blocks the worker in `put`, so it never takes another task, so the supervisor waits forever |
| re-dispatch reuses the lease id | a worker only *presumed* dead may still finish. The lease id is what lets the caller drop the second answer instead of putting two cards for one task into the evidence pool |

!!! warning "`spawn` re-imports `__main__`"
    A script that builds a `ProcessExecutor` at module level builds one again in
    every child, which builds one in every grandchild. The machine fills with
    processes and nothing reports, so it reads as *slow* rather than as a fault.
    Put the run behind `if __name__ == "__main__":`. Building one inside a worker
    is refused outright.

## Recovery is at task granularity

The supervisor notices a worker is gone — dead, or alive and wedged past
`hang_timeout` — and sends its task somewhere else under the same lease id. A
task whose worker dies repeatedly is given up on after two re-dispatches and
reported as an infrastructure failure rather than retried forever.

Two counters make the policy visible, and the second is the one worth watching:

```python
r.redispatched          # tasks sent out again after a worker was presumed lost
r.duplicates_dropped    # answers that arrived for a task already answered
```

Dropping a duplicate is correct and invisible, so without the count an
over-eager re-dispatch policy looks exactly like a well-tuned one — it is simply
paying twice.

!!! note "Not partial-rollout resume"
    Resuming a half-finished rollout would need `run(rendered, task) -> output` to
    become an inspectable conversation, and that opaque contract is what lets any
    agent be plugged in at all. [`ResumeQueue`](async.md) stays the turn-level
    primitive it always was, unwired, rather than being repurposed as a
    task-level channel because it happens to be a queue.

## The gate has its own concurrency

Rollouts and evaluations are different workloads that call the same function. A
rollout is long-tailed, often fails, and is one opinion among many. An evaluation
is batched, cacheable, and decides whether a change is committed — losing one
costs the decision, not a little evidence.

So they are sized separately: `max_concurrency` bounds rollouts,
`eval_concurrency` bounds the [evaluation group](verifier.md#the-evaluation-group).
Sizing them together means sizing them for whichever matters less.

## Related

* [Parallelism (DP / TP)](parallelism.md) — how a round's work is *split*
* [Sandboxes](sandboxes.md) — the environment a rollout runs *in*
* [Async](async.md) — whether rounds have a barrier at all
* [The verifier](verifier.md) — the evaluation group and its cache
