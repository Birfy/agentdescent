# Staleness — diffs proposed against a version that moved

*Module:* [`agentdescent.staleness`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/staleness.py)
· *API:* [`StalenessPolicy`, `FullStaleness`, `GuardedStaleness`, `ReflectiveStaleness`, `get_policy`](api.md#staleness-policies)

This is the problem parallel self-improvement has and serial self-improvement does
not. A worker reads version 4, spends a minute on a rollout, and proposes a diff
— but three other workers merged in the meantime and the head is now version 7.
Is the proposal still good?

```python
eta = vv_staleness(head, base_version)     # 0 = fresh, larger = the world moved
```

Throwing every stale diff away wastes the expensive part of the run (the
rollout). Accepting every one of them merges changes justified by a state that no
longer exists. The policy is where you choose.

```python
evolve(tasks, reward, agent=agent, staleness_policy=get_policy("guarded"))
```

| policy | `eta == 0` | `0 < eta <= alpha` | `eta > alpha` |
|---|---|---|---|
| `full` | accept | accept | accept |
| **`guarded`** (default) | accept | **rebase** + re-verify | discard |
| `reflective` | accept | rebase | **rebase** — `alpha` is ignored |

`full` is maximum throughput and the reference orchestrator's baseline. `guarded`
is the AReaL bounded-staleness discipline expressed over diffs. `reflective` is
FlashEvolve's top tier: every stale diff gets a replay on the current head, and is
discarded only if the improvement no longer holds — the highest recovery of
otherwise-wasted proposals, paid for in cheap-eval work.

A **contract-breaking** diff is the exception in all three: once stale it is
discarded rather than rebased, because a cross-contract rebase costs more than
re-proposing.

`alpha` is the tolerance, and it widens for a **cold** artifact — one that few
diffs touch — because the odds that an unrelated merge invalidated this proposal
are lower. The [aggregator](aggregator.md) supplies it.

## The trade-off, measured

From [`examples/rq2_staleness.py`](https://github.com/Birfy/agentdescent/blob/main/examples/rq2_staleness.py)
and [`examples/run_async.py`](async.md) on the reference domain, at the same
`async_ratio`:

* **Guarded discards more work than Reflective** — that is the claim under test,
  and it holds regardless of machine speed.
* **Reflective spends fewer rollouts** to reach the same accuracy, because a
  rebased card is a rollout it did not have to repeat.
* Recovering that work never leaves Reflective *behind* Guarded on final
  accuracy.

The counts are in `result.outcomes()` as `all-stale`, and in the async runtime's
`AsyncStats.discarded_stale`.

## `StaleAction` — the three outcomes

```python
class StaleAction(Enum):
    ACCEPT   # merge it as proposed
    REBASE   # cheap re-verification, then merge against the new head
    DISCARD  # settle the evidence card back into the pool
```

`DISCARD` is not deletion. The [evidence card](data-model.md#evidencecard-the-gradient-metadata)
goes back to the pool: the diff no longer applies, but the *observation* that
justified it is still true, and the pool is what a later round draws on.

## Writing your own

```python
from agentdescent import StaleAction

class MyPolicy:
    name = "patient"

    def decide(self, eta: int, alpha: int, contract_breaking: bool) -> StaleAction:
        if contract_breaking and eta > 0:
            return StaleAction.DISCARD
        return StaleAction.ACCEPT if eta == 0 else StaleAction.REBASE

evolve(tasks, reward, agent=agent, staleness_policy=MyPolicy())
```

Anything with a `name` and `decide(eta, alpha, contract_breaking) -> StaleAction`
works. `get_policy(name)` resolves the three built-ins, which is what the
examples' `--policy` flags use.

## Its relationship to `async_ratio`

The [lag budget](async.md#async_ratio-the-lag-budget) bounds how far ahead the
workers may run; the staleness policy decides what to do with whatever slips
through anyway. They have to be set together:

* tolerance too tight for the lag budget → everything is discarded, and
  `result.outcomes()` fills with `all-stale` while nothing commits;
* lag budget too tight → the workers idle at a barrier you were trying to remove.

`result.forced_refreshes` is the symptom of the mismatch: cards arriving, nothing
committing, workers forced to resync. A non-zero count means the two knobs
disagree.
