# The reference orchestrator and the reference domain

*Modules:* [`agentdescent.orchestrator`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/orchestrator.py)
· [`agentdescent.domains.router`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/domains/router.py)
· *API:* [`AgentDescent`, `RoundStat`, `run_fork_baseline`](api.md#the-reference-orchestrator)

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
                      n_workers=6, noise=0.15, refresh_interval=2, seed=0)
stats = system.run(rounds=40)          # -> List[RoundStat]
```

Each `RoundStat` carries `round`, `dev_accuracy`, `stable_accuracy`,
`committed`, `fused`, `discarded`, `conflicts`, `oracle_used` — the learning
curve plus the reason it has that shape.

Staleness arises **naturally** rather than being injected: workers refresh their
ledger snapshot only every `refresh_interval` rounds (default 2), so between refreshes their
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

## The rollout, and where the noise comes from

There is no `Worker` class any more. `AgentDescent` and `AsyncAgentDescent` are
**adapters over [`evolve()`](evolution.md) and [`async_evolve()`](async.md)** —
they describe this domain in the vocabulary those speak and let them run it. The
loop they used to own is gone, and with it the second implementation
[`architecture.md`](architecture.md) called a known wart.

What a rollout *is* has not changed: classify a cluster against the current
table, turn the failures into a diff of up to `max_ops` keyword fixes, attach an
[evidence card](data-model.md). It is now expressed as the three callables the
engine takes:

```python
from agentdescent.domains.router import (
    RouterStrategy, cluster_tasks, router_propose, router_reward, router_run)

train, held = cluster_tasks(universe, n_clusters=6)
evolve(train + held, router_reward, run=router_run,
       propose=router_propose(universe.gold, noise=0.15, seed=0),
       strategy=RouterStrategy())
```

`propose` is a deterministic corrector with tunable `noise`, which is what makes
the aggregator's two hard paths reachable on demand: raise `noise` to generate
contradictions, spread the tasks to generate complements. `max_ops` keeps a
diff inside the aggregator's [trust region](aggregator.md), so the baseline
never wins or loses by emitting one enormous diff.

!!! note "Three things the translation does not preserve exactly"
    Listed in `agentdescent/domains/router.py` rather than left to be found:
    `before_after_delta` and `evidence_eval` are measured over the whole cluster
    rather than the failing subset, and **noise is per proposal, not per
    worker** — the general engine has one `propose` for every worker and, by
    design, no worker identity to branch on. `run_fork_baseline` keeps per-fork
    noise, because a fork *is* one actor for its whole run.

    Both paths still converge on the same table and merge still beats fork,
    which is what the [results](results.md) actually claim.

`rollout=` replaces `Worker.rollout_latency`: pass a callable that sleeps before
doing the domain's work, and parallelism becomes observable in wall-clock. That
is how the [efficiency experiments](efficiency.md) inject latency, and how the
resilience tests inject failures.

## Relationship to `evolve()`

| | `AgentDescent` | [`evolve()`](evolution.md) |
|---|---|---|
| artifact | `RouterStrategy` (fixed) | any [`Strategy`](strategies.md) |
| actor | a deterministic, noisy corrector | your agent or model |
| purpose | measuring the system | using the system |
| needs a model | no | usually |

Both drive the same ledger, aggregator, verifier, scheduler and governance. When
a claim on the [results page](results.md) says "measured", it was measured
here — with the deterministic actor — so the number is about the merge machinery
and not about a model's mood.
