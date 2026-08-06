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
| **3** (default) | the shipped default; the reference domain is measured at 4, [here](concepts.md#34-async_ratio-roll-flash-the-global-lag-budget) |
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
| `rounds=` | **reinterpreted** as a budget of `rounds × n_workers` worker rollouts (silent, and exact, once `max_rollouts=` says the budget outright) |
| `max_seconds=None` | **becomes 20.0 seconds**, where it means "no limit" on the sync path |

Each of those emits a `RuntimeWarning`. The last two are the sharp ones: flipping
one boolean turns an unbounded run into a 20-second one, and a partial artifact
with `error=None` and a populated `history` is indistinguishable from a converged
one. **Check `result.stop_reason`** — `"target_reward"` is convergence,
`"max_seconds"` / `"max_iters"` / `"max_rollouts"` / `"max_calls"` is a budget
expiry.

Say the rollout budget outright rather than deriving it from `rounds`. Either
spelling works and both silence the reinterpretation warning:

```python
from agentdescent import async_evolve, evolve

evolve(tasks, reward, agent=agent, asynchronous=True,
       n_workers=6, max_seconds=120, max_rollouts=200, max_calls=400)

async_evolve(tasks, reward, agent=agent,
             n_workers=6, async_ratio=3,
             max_seconds=120, max_iters=200, max_calls=400)
```

`max_calls` is the second half of an equal-budget comparison: two configurations
matched on rollouts still differ in model spend whenever one asks for more
proposals per rollout, and a rollout that solves its task never proposes at all —
so calls are not a fixed multiple of rollouts and cannot be derived from them.
Both bounds are checked as each rollout lands here, so the barrier-free path
overshoots only by what was already in flight.

## How the pipeline holds together

Three properties are worth knowing before you tune anything:

* **The lag budget bounds un-merged work, not just version drift.** A worker will
  not pile up more than `async_ratio` candidates ahead of the merger. That matters
  at **cold start**: before the first commit the head has not advanced, so a
  version-only budget cannot engage, and workers would flood the buffer while the
  merger is busy with the first slow held-out evaluation. Gating on pending intake
  prevents it.
* **There is exactly one merger,** so it is the only writer — there are no CAS
  conflicts on this path, and a custom
  [`aggregator_factory`](aggregator.md) sees only already-rebased cards. Every
  optimizer that works synchronously works here unchanged.

    Two consequences follow from "only writer", and both used to be paid for
    rather than used:

    * **The head is published, not fetched.** A worker measures its drift against
      a version the merger publishes after each sweep, not against the ledger. It
      used to read the ledger on *every rollout*, and a ledger read is a
      `git checkout` behind a process-wide file lock plus an RLock the whole run
      queues on — so the cost of asking "am I far enough behind?" grew with the
      concurrency it exists to support (measured: 46 reads for a 21-rollout run,
      22 after). The published head can lag by one sweep, which delays a refresh
      by at most one rollout; the refresh itself still takes a real snapshot.
    * **The staleness denominator is split.** This gate sees every card and
      forwards only the survivors, and `Aggregator` then counts those survivors
      on the same meter — so each survivor was counted as "considered" twice, and
      `result.stale_rate()` came out at roughly half the truth (20 cards
      reporting `stale_considered = 40`). Each side now counts what only it can
      see: the discards here, the survivors there.
* **A backpressure guard forces a global sync if the pipeline stalls** (evidence
  arriving, nothing committing). Without it a mismatched `async_ratio > alpha`
  livelocks under Guarded: workers propose against a snapshot too old to accept,
  every card is discarded, the head never moves, so the lag budget never triggers
  a refresh either. `stall_patience=` tunes it; `result.forced_refreshes` counts
  how often it fired.

`self_verify` controls whether a worker, after producing a diff, re-runs its own
trajectory with the diff applied to record a local before/after signal
(`before_after_delta`, which the staleness gate's cheap re-verify uses). Ports
that score only the *candidate* on held-out — [EvoSkill](algo-evoskill.md),
whose repo evaluates the child on the validation set and never re-runs the sampled
task — pass `self_verify=False` to skip that extra rollout. So do the
[directory entry points](directory-evolution.md), where it would double the agent
calls per proposal.

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

`stall_patience` (default 50) is how many merger sweeps may pass **with cards in
them and nothing committing** before every worker is forced to refresh — a sweep
that had no evidence to merge is neither progress nor a stall, and is not
counted. `shutdown_grace` is how long the runtime waits for in-flight rollouts
when it stops.

## `AsyncAgentDescent` — the reference runtime

`async_evolve` is the general engine. `AsyncAgentDescent` is the research
orchestrator it grew out of: it runs the same barrier-free pipeline over the
[reference domain](orchestrator.md#why-a-synthetic-domain-exists-at-all), with no LLM involved,
which is what makes the parallelism claims testable offline.

```python
from agentdescent import AsyncAgentDescent, AsyncConfig, get_policy
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
