# Fusion policies — merging what survived

*Module:* [`agentdescent.fusion`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/fusion.py),
[`agentdescent.defaults`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/defaults.py)
· *Contract:* `FusionPolicy.select(artifact, diffs) -> (diff, applied, fused)`

After conflict resolution the surviving diffs are pairwise non-contradicting,
so their union always builds. The question is what to do when values *did*
contest, and whether anyone pays a ranking evaluation.

## Implemented

| Policy | Rule | Reach for it when |
|---|---|---|
| `DefaultFusion` | union of complementary diffs (`ops.update`), straight to the gate; with `tournament=True` it first ranks every single against the union on the cheap layer | the default. The tournament is the only instrument that answers "does merging average the improvements away?" — a per-workload diagnostic, not a tax |
| `ReflectiveFusion(complete)` | asks a model to write the union of *contested* values — one model call, one gate evaluation, **no ranking of anything**; falls back to `DefaultFusion` when synthesis fails | text-valued keys where dropping a contradiction loses real work. Measured 52% cheaper in model calls on a matched workload. **Not** for code or strict-JSON values: the synthesized value bypasses the strategy's validator |

```python
evolve(tasks, reward, agent=agent, n_workers=4,
       policies=Policies(**reflective_merge(completion)))
```

`reflective_merge` returns the `fusion` + `conflict` pair because
`ReflectiveFusion` installed alone is a no-op: `DefaultConflict` has already
dropped the contradictions it exists to merge.

The method runner applies exactly this split: text-valued artifacts
get reflective merge, code/JSON-valued artifacts keep `DefaultConflict` —
see the [matrix overview](matrix-overview.md).

## The deep dive: when a dictionary update cannot merge

`fuse_diffs` is `ops.update()`. That is right when two workers touched
**different** keys, and useless when they touched the same one: the last writer
wins, so `DefaultFusion` declines to build a fused candidate at all. For an
artifact held in **one key** that is every round — GEPA's `InstructionSlot`
records `contested = 0` for a whole run.

```python
from agentdescent import Policies, evolve
from agentdescent.fusion import reflective_merge

evolve(tasks, reward, agent=agent, n_workers=4,
       policies=Policies(**reflective_merge(completion)))
```

A model writes one value keeping what each proposal contributed, for the keys the
diffs actually disagree on. Keys they agree on stay the plain union — a model
asked to merge values that do not disagree can only make them worse.

**It is asked for a union of deltas, not for a rewrite.** "Write one version that
keeps every improvement" invites a fresh composition that happens to cover the
same ground, and there is no way to check whether it did. The prompt
(`fusion.MERGE_PROMPT`) instead says:

```
Several independent improvements were made to the same text, each fixing a
different failure. Produce their UNION.

CURRENT
--- {the value the workers started from} ---

PROPOSAL 1 --- {worker 0's whole rewrite} ---
PROPOSAL 2 --- {worker 1's} ---
PROPOSAL 3 --- {worker 2's} ---

Do this:
1. For each proposal, work out what it CHANGED relative to CURRENT.
2. Output CURRENT with every one of those changes applied together.
```

That is an operation whose result can be checked, and it needs only what a
`FusionPolicy` receives — the current value and the competing ones. Fusion never
sees the evidence cards, so "what was each proposal fixing" is not available and
deriving the deltas from CURRENT is what makes it unnecessary rather than
missing.

Verified on `GLM-5.2` with three real GEPA-style rewrites of one instruction:

| | |
|---|---|
| CURRENT | *Answer the question using the given context.* |
| worker 0 | + *for comparison questions, verify the attribute for BOTH entities* |
| worker 1 | + *reply with the shortest correct form, no explanation* |
| worker 2 | + *for yes/no questions reply with exactly 'yes' or 'no'* |
| **union** | all three survive into one instruction |
| `fuse_diffs` on the same input | keeps **one**, the other two are lost |

Four things it refuses to do, each because the obvious version would mislead: it
will not accept an answer that merely repeats one of its inputs (that is not a
merge, and would enter as a duplicate); it will not accept one over `max_chars`
(the synthesised value reaches the ledger without passing the trust region, which
filters *cards*); it will not commit a **partial** union when one contested key
fails (that would ship some workers' contributions and silently drop the rest,
which looks like success); and a dead backend falls back rather than raising —
fusion sits on the commit path of every round.

!!! danger "There is no tournament on this path, and that is the trade"
    The union goes **straight to the acceptance gate**. Measured on a workload
    where both paths reach the same final quality: **55 model calls against
    114** — 52% cheaper, because ranking every candidate was the largest cost in
    a merge and nothing here ranks anything.

    | | |
    |---|---|
    | **acceptable** | the union can commit while being worse than the best single would have been. It can still never commit a **regression** — the gate scores it on the full held-out set and runs the Beta test and the regression guard, untouched |
    | **gone** | `best_single_score`, and with it the answer to *"does merging just average the improvements away?"* Nothing scores a single, so every trial is `ranked=False`, `fusion_stats()` reports them as `unranked`, and `win_rate` is `None` |

    A union that was never compared has not won anything, and the statistics
    cannot pretend otherwise. To **measure** whether merging helps, use the
    shipped `DefaultFusion` on a multi-key artifact instead.

!!! warning "It needs two policies, which is why `reflective_merge` returns a pair"
    Conflict resolution is **step 2** and fusion is **step 3**, so `DefaultConflict`
    has already dropped the losing side of every contradiction before a fusion
    policy runs. `ReflectiveFusion` alone is handed a single diff on exactly the
    workloads it was written for, and correctly declines to merge it with itself.
    `KeepContradictions` is the partner that leaves them for step 3.

The one path that still ranks is the fallback: a dead backend, an empty or
oversized answer, or one that merely repeats an input falls through to
`DefaultFusion` rather than losing the round's work — counted as
`synthesis_failed`, apart from `contradiction`, which means no model was asked.

---

