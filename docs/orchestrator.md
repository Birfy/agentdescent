# The reference orchestrator and the reference domain

*Modules:* [`agentdescent.orchestrator`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/orchestrator.py)
· [`agentdescent.worker`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/worker.py)
· [`agentdescent.domains.router`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/domains/router.py)
· *API:* [`AgentDescent`, `RoundStat`, `run_fork_baseline`](api.md#the-reference-orchestrator), [`Worker`](api.md#the-worker)

[`evolve()`](evolution.md) is the entry point you build on. `AgentDescent` is the
one the **research claims were measured with**: the same loop, wired to a
deterministic in-process domain so the whole parallel/async machinery runs with
no model, no network, and no API key.

```
TaskScheduler --lease--> Workers --diff + evidence--> EvidenceBuffer
      --> Aggregator (staleness / conflict / fusion / accept) --CAS--> Ledger
      --broadcast--> Workers        ;  AuditScheduler audits merges
```

If you only want to *use* the framework, you never touch this page. If you want
to know whether its central claim is true, this is where you check.

## Why a synthetic domain exists at all

The interesting behaviour — staleness, conflict resolution, fusion, statistical
acceptance, merge-versus-fork — is a property of the **aggregation system**, not
of any particular model. Testing it through an LLM would make every result a
measurement of that LLM, on top of being slow, expensive and non-deterministic.

`domains/router.py` is the smallest task with the right structure:

* the skill is a `keyword -> label` table; the optimal skill maps every keyword
  to its gold label, and a fresh skill knows nothing;
* two workers fixing **different** keywords produce complementary diffs → fusion
  should win;
* a noisy worker proposing the **wrong** label for a keyword produces a
  contradiction → conflict resolution must drop it.

That is exactly the structure needed to show that merging concurrent diffs beats
forking them, and it runs in milliseconds.

```python
from agentdescent.domains.router import make_task_universe, RouterSkill, router_eval
```

`RouterSkill` is a hand-written [`Evolvable`](data-model.md) — the worked example
to copy when your artifact is not a flat `{key: value}` dict.

!!! note "`RouterTask` is not `Task`"
    The domain's own task type is `RouterTask(text, label, keyword)`. It is
    aliased to `Task` inside that module for backwards compatibility, which is a
    genuine collision with [`agentdescent.evolution.Task`](evolution.md) —
    disjoint fields, no relationship. Prefer `RouterTask` in new code.

## `AgentDescent` — the merge-based loop

```python
from agentdescent import AgentDescent
from agentdescent.domains.router import make_task_universe

system = AgentDescent(repo_path, make_task_universe(seed=7),
                      n_workers=6, noise=0.15, refresh_interval=3, seed=0)
stats = system.run(rounds=40)          # -> List[RoundStat]
```

Each `RoundStat` carries `round`, `dev_accuracy`, `stable_accuracy`,
`committed`, `fused`, `discarded`, `conflicts`, `oracle_used` — the learning
curve plus the reason it has that shape.

Staleness arises **naturally** rather than being injected: workers refresh their
ledger snapshot only every `refresh_interval` rounds, so between refreshes their
`base_version` lags the head and the [staleness policy](staleness.md) has real
work to do.

## `run_fork_baseline` — the control

```python
from agentdescent import run_fork_baseline

best = run_fork_baseline(universe, n_workers=6, noise=0.15, rounds=40, seed=0)
```

The same workers, the same budget, the same noise — but each keeps its own
private artifact and none of them merge. This is the DGM-style archive/fork
strategy, and it is the control for the framework's central claim: N workers
merging should beat N workers forking on an equal budget.

Measured on the reference domain (see [results](results.md) and
[`examples/run_demo.py`](https://github.com/Birfy/agentdescent/blob/main/examples/run_demo.py)):

```
AgentDescent (merge) held-out accuracy : 1.000
Fork/archive best-fork accuracy        : 0.379
merge advantage                        : +0.621
```

```bash
python -m examples.run_demo        # reproduces both numbers, no API key
```

## `Worker` — rollout and propose

```python
Worker(worker_id="w0", gold=gold_table, noise=0.15, max_ops=4, seed=0)
```

A worker holds a **snapshot** of the ledger, runs tasks against it, and turns
observed failures into a diff plus an [evidence card](data-model.md). It never
mutates the ledger — all mutation goes through the
[aggregator](aggregator.md), which is the sole optimizer.

In a real deployment "propose" is a model reflecting on a trajectory. Here it is
a deterministic corrector with tunable `noise`, which is what makes the
aggregator's two hard paths reachable on demand: raise `noise` to generate
contradictions, spread the tasks to generate complements.

`rollout_latency` is an optional callable returning seconds. It models the real
cost of a rollout — tool calls, queue time, model latency — and is what makes
parallelism and asynchrony observable in wall-clock in the
[efficiency experiments](efficiency.md).

`max_ops` keeps a worker's diffs inside the aggregator's
[trust region](aggregator.md) so the baseline never wins or loses by emitting
one enormous diff.

## Relationship to `evolve()`

| | `AgentDescent` | [`evolve()`](evolution.md) |
|---|---|---|
| artifact | `RouterSkill` (fixed) | any [`Strategy`](strategies.md) |
| actor | `Worker` (deterministic, noisy) | your agent or model |
| purpose | measuring the system | using the system |
| needs a model | no | usually |

Both drive the same ledger, aggregator, verifier, scheduler and governance. When
a claim on the [results page](results.md) says "measured", it was measured
here — with the deterministic actor — so the number is about the merge machinery
and not about a model's mood.
