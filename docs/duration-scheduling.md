# Duration-aware scheduling

> **Belongs to the async runtime**, not synchronous [`evolve`](evolution.md):
> pass a `DurationEstimator` to `AsyncAgentDescent` for straggler checkpointing. The
> `DurationEstimator` / `lpt_schedule` primitives are usable on their own too.

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
times every rollout, calibrates the estimator, and **checkpoints any rollout
that overruns `duration_timeout_factor × its estimate`** to the `ResumeQueue`
(partial rollout) instead of letting it block a worker.

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

This is the design's partial-rollout mechanism (§5.1) driven by a live estimate:
a rollout predicted to be short but running long is set aside and resumed against
the latest ledger, rather than defining the wall-clock of its worker.

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
