# Async — removing the round barrier

*Modules:* [`agentdescent.async_evolve`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/async_evolve.py)
· [`agentdescent.async_runtime`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/async_runtime.py)
· *API:* [`async_evolve`](api.md#barrier-free-evolution), [`AsyncAgentDescent`, `AsyncConfig`, `AsyncStats`](api.md#the-async-orchestrator)

The synchronous loop runs a barrier: every worker steps, then one
`aggregator.step()` fires, then the next round begins. The barrier is what makes
a run reproducible and easy to reason about — and it is also what makes the whole
round wait for its slowest rollout.

```python
result = evolve(tasks, reward, agent=agent,
                asynchronous=True, async_ratio=3, max_seconds=120)
```

Two stages — *rollout + propose* (the workers) and *aggregate + commit* (the
merger) — become independent threads connected by a thread-safe
`EvidenceBuffer`. A worker keeps producing evidence while the merger is still
working through the previous batch, so the pipeline overlaps instead of stalling.

!!! note "What the GIL does and does not cost"
    Python threads give no CPU parallelism. But rollouts are network-bound (the
    GIL is released during I/O), the pipeline overlap is real, and the
    concurrency-control machinery — CAS, buffer locks, per-diff staleness — is
    exactly the code a genuinely parallel process or host pool needs. Nothing
    here is a simulation of concurrency; it is concurrency at the wrong
    granularity for CPU work and the right one for agent work.

## `async_ratio` — the lag budget

A worker refreshes its ledger snapshot only once the head has drifted more than
`async_ratio` versions ahead of it. That single number is the throughput /
staleness trade:

| `async_ratio` | behaviour |
|---|---|
| small (1–2) | near-synchronous: few stale diffs, workers resync often |
| **3** (default) | the measured sweet spot on the reference domain |
| large (8+) | highly asynchronous: many stale diffs for the [staleness policy](staleness.md) to rebase or discard |

The two knobs are one decision. A tight staleness tolerance with a large lag
budget discards everything (`outcomes()` fills with `all-stale` and nothing
commits); a tight lag budget re-introduces the waiting you removed the barrier to
avoid. `result.forced_refreshes` is the mismatch showing itself.

## `evolve(asynchronous=True)` vs `async_evolve()`

They are the same engine; the first delegates. The wrapper exists so that
switching costs one argument, but three of `evolve()`'s parameters have no
meaning without a barrier and it **says so** rather than dropping them silently:

| argument | what happens under `asynchronous=True` |
|---|---|
| `parallel=` | ignored — the async runtime shards data-parallel across its own workers |
| `max_concurrency=` | ignored — concurrency *is* `n_workers` |
| `round_timeout=` | ignored — there is no barrier to bound; use the backend's own timeout |
| `rounds=` | **reinterpreted** as a budget of `rounds × n_workers` worker rollouts |
| `max_seconds=None` | **becomes 20.0 seconds**, where it means "no limit" on the sync path |

Each of those emits a `RuntimeWarning`. The last two are the sharp ones: flipping
one boolean turns an unbounded run into a 20-second one, and a partial artifact
with `error=None` and a populated `history` is indistinguishable from a converged
one. **Check `result.stop_reason`** — `"target_reward"` is convergence,
`"max_seconds"` / `"max_iters"` is a budget expiry.

Call `async_evolve(max_iters=...)` directly when you want an exact rollout count.

```python
from agentdescent import async_evolve

result = async_evolve(tasks, reward, agent=agent,
                      n_workers=6, async_ratio=3,
                      max_seconds=120, max_iters=200)
```

## What the async path adds

Beyond the barrier removal, three signals only it can report:

| field | meaning |
|---|---|
| `result.forced_refreshes` | workers forced to resync because the pipeline stalled — cards arriving, nothing committing |
| `result.stragglers` | rollouts that overran their predicted duration by `straggler_factor` (needs a [`duration_estimator=`](duration-scheduling.md)) |
| `result.retired_workers` | workers that gave up after repeated backend failures |

`retired_workers` deserves attention: a run can finish **cleanly** at a fraction
of its requested concurrency, so `error` stays `None` while throughput quietly
drops. Check it to tell a fast run from a lucky one.

`stall_patience` (default 50) is how many empty merger sweeps pass before a
worker is forced to refresh; `shutdown_grace` is how long the runtime waits for
in-flight rollouts when it stops.

## `AsyncAgentDescent` — the reference runtime

`async_evolve` is the general engine. `AsyncAgentDescent` is the research
orchestrator it grew out of: it runs the same barrier-free pipeline over the
[reference domain](orchestrator.md#why-a-synthetic-domain-exists-at-all), with no LLM involved,
which is what makes the parallelism claims testable offline.

```python
from agentdescent import AsyncAgentDescent, AsyncConfig
from agentdescent.domains.router import make_task_universe

cfg = AsyncConfig(n_workers=6, async_ratio=4, noise=0.12,
                  target_accuracy=0.95, max_seconds=15.0, seed=1)
stats = AsyncAgentDescent(repo_path, make_task_universe(seed=7),
                          config=cfg,
                          staleness_policy=get_policy("reflective")).run()

print(stats.rollouts, stats.commits, stats.discarded_stale,
      stats.final_dev_accuracy, stats.final_stable_accuracy)
```

`AsyncStats` also carries `sweeps`, `fused`, `conflicts_dropped`,
`stragglers_checkpointed`, `oracle_used`, `wallclock` and a `timeline` of
`(rollout, accuracy)` pairs — the raw material behind the
[measured results](results.md) and
[`examples/run_async.py`](https://github.com/Birfy/agentdescent/blob/main/examples/run_async.py).

## Choosing sync or async

| you want | use |
|---|---|
| reproducibility, a clean per-round trace, a paper table | synchronous (`max_concurrency=n_workers` for the speedup) |
| maximum throughput, long or uneven rollouts | `asynchronous=True` |
| both, to compare | run each and read `result.history` — but see the reinterpretation table above before comparing lengths |

`len(result.history)` is **not** comparable across the two: on the async path a
`RoundInfo.round` is a merger-sweep index, not a round.
