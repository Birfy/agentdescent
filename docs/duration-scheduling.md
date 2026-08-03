# Duration-aware scheduling

> **Belongs to the async paths**, not synchronous [`evolve`](evolution.md), which
> has no `duration_estimator=` parameter at all and bounds a slow rollout with
> `round_timeout=` instead. Pass a `DurationEstimator` to `AsyncAgentDescent`
> (reference runtime) or to [`async_evolve`](async.md) (the general one, where it
> reports `result.stragglers`). The `DurationEstimator` / `lpt_schedule`
> primitives are usable on their own too.

Agentic rollouts are heavy-tailed and their cost correlates with task size. This
module **estimates a rollout's duration from the task's length**, then uses that
estimate for asynchronous scheduling — dispatching to minimize makespan and
detecting stragglers. It's the concrete machinery behind the design's
**L-traj** (trajectory-duration) long tail (design spec §5.1).

```bash
python -m examples.duration_scheduling
```

Source: [`examples/duration_scheduling.py`](https://github.com/Birfy/agentdescent/blob/main/examples/duration_scheduling.py).

---

## 1. Estimate duration from task size

`DurationEstimator` fits `seconds ≈ intercept + slope · length` **online**, by
least squares, as real rollout durations arrive — the constant isn't known a
priori, so it calibrates itself:

```python
from agentdescent import DurationEstimator

est = DurationEstimator()
est.observe(cost=len(task.prompt), seconds=measured)   # after each rollout
predicted = est.estimate(cost=len(next_task.prompt))   # before the next
```

It recovers the true law quickly (prediction MAE shrinks; learned parameters
converge to ground truth):

```
 rollouts  pred MAE (s)   learned b   learned m
       10         0.117       0.056     0.00080
       30         0.015       0.051     0.00080
      300         0.017       0.050     0.00080
(ground truth: b=0.05, m=0.0008)
```

---

## 2. Dispatch by estimate to minimize makespan

Given a batch of tasks with estimated durations, dispatch the **longest first**
(LPT) to the least-loaded worker instead of round-robin. Dispatching the tail
*early* keeps one long rollout from defining the whole batch's wall-clock:

```python
from agentdescent import lpt_schedule, fifo_makespan

weights = [est.estimate(len(t.prompt)) for t in tasks]
assignment, makespan = lpt_schedule(weights, n_workers)   # near-optimal
baseline = fifo_makespan(weights, n_workers)              # round-robin
```

On 40 heavy-tailed tasks over 4 workers:

```
        dispatch  makespan (s)  vs optimal
     round-robin          8.23       1.28x
LPT (by estimate)          6.49       1.01x
optimal lower bound          6.44

LPT speedup over round-robin: 1.27x
```

LPT lands within **1% of the optimal** `total/N` lower bound (its worst-case
guarantee is 4/3); round-robin is 28% over.

---

## 3. Straggler checkpointing in the async runtime

Pass a `DurationEstimator` to `AsyncAgentDescent` and it becomes duration-aware: it
times every rollout, calibrates the estimator, and **records any rollout that
overran `duration_timeout_factor × its estimate`** into the `ResumeQueue`. The
record is written once the rollout returns, so this measures the tail rather than
cutting it short — see the warning below.

```python
from agentdescent import AsyncAgentDescent, AsyncConfig
from agentdescent import DurationEstimator

sys = AsyncAgentDescent(repo, universe,
                     config=AsyncConfig(duration_timeout_factor=3.0),
                     estimator=DurationEstimator())
stats = sys.run()
stats.stragglers_checkpointed        # overrunning rollouts DETECTED (see the note below)
```

```
rollouts=727, learned base≈0.006s, stragglers detected=108
```

This is the *detection* half of the design's partial-rollout mechanism (§5.1),
driven by a live estimate: a rollout predicted to be short but running long is
identified and counted, so it shows up in the stats instead of silently setting
the pace. The design's other half — setting it aside and resuming it against the
latest ledger — is where the mechanism stops:

!!! warning "Straggler *resume* is not implemented"
    A rollout that overruns its predicted cost is flagged into `ResumeQueue` and
    counted — that part is real, and it is what keeps a straggler from silently
    defining the round's wall-clock in the reported stats. But the rollout is not
    interrupted (the flag is recorded *after* it returns), the queued item carries
    no continuation state, and **nothing pops the queue**. True turn-level
    checkpoint-and-resume would need a rollout contract that exposes its turns;
    the engine's `run(rendered, task) -> output` is opaque. What actually prevents
    one slow rollout from stalling the rest today is removing the barrier — see
    [the async runtime](evolution.md#the-barrier-free-runtime-async_evolve), which
    the [efficiency experiments](efficiency.md) measure at ~2.8x over a sync
    barrier under heavy-tailed latency.

---

## 4. The audit scheduler — allocating oracle budget

The same module holds the other scheduler: the one that decides **which merge
decisions are worth checking with ground truth**, given a budget that runs out.

```
priority(d) = blast_radius(d) * uncertainty(d) / trust(artifact)
```

High blast radius means a mistake is expensive; high uncertainty means the cheap
layers do not know; low trust means the cheap layers have been *wrong here
before*. The aggregator submits its own merge decisions here, which closes the
audit loop over the optimizer itself — "submits", not "enqueues": with the
default `collect=False` the priority is computed and returned without a queue
being built (see below).

```python
from agentdescent import AuditScheduler

audit = AuditScheduler()
audit.submit(diff_id, artifact_id, blast_radius=0.6, uncertainty=0.1)
audit.force_oracle(blast_radius, artifact_id)   # -> bool
audit.update_trust(artifact_id, oracle_agreed=True)
```

### Trust has to be measurable for free

`force_oracle` fires on `blast_radius > FAST_MAX` **or** `trust < 0.75`, and
trust rises by `+0.25` when the cheap layer agreed with the oracle, halving when
it did not.

That signal must be obtainable without spending oracle budget, or the rule is
circular — and it was: the only writer of trust sat inside the `force_oracle`
branch, so for any artifact below the L1 boundary the condition could never
become true and the audit never ran at all. Measured on the default
`blast_radius=0.2`: `oracle_calls_used == 0` for a whole run, trust pinned at its
initial 1.0.

The fix costs nothing: the [verifier](verifier.md)'s `eval_counts` already scored
base and candidate on the full held-out set for the acceptance test, so comparing
that verdict with the cheap layer's is free and happens on every merge.

### The boundary lives in one place

`force_oracle` reads `FAST_MAX` from [`governance`](governance.md) rather than
re-deriving it. A third hand-written threshold here (`>= 0.5`) meant an artifact
at 0.4 was L1 by governance and audited like an L2 skill: never.

### The queue is opt-in, because nothing drains it

```python
AuditScheduler()                  # computes priorities, queues nothing (default)
AuditScheduler(collect=True)      # keeps the heap for an out-of-band auditor
```

`force_oracle` and the trust update are the parts in the engine's path, and both
work either way. The priority *queue* has no consumer in the shipped runtimes —
so maintaining a heap plus a periodic rebuild, once per merge decision, bought
nothing and cost work on the merge path. It is now off unless asked for.

With `collect=True` the heap is capped at `MAX_QUEUED` (4096), sheds the
lowest-priority tail, and reports what it shed in `audit.dropped`.
