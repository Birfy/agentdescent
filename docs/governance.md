# Governance — L0 frozen, L1 slow, L2 fast

*Module:* [`agentdescent.governance`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/governance.py)
· *API:* [`Layer`, `classify`, `assert_mutable`, `FROZEN_IDS`, …](api.md#governance)

Not everything an agent can change should change at the same speed. A phrasing
tweak to one skill is recoverable; a change to the harness that runs every task
is not, and a change to the thing that *judges* the changes is a different
category altogether.

```python
evolve(tasks, reward, agent=agent, blast_radius=0.2)   # L2 — a skill
evolve(tasks, reward, agent=agent, blast_radius=0.6)   # L1 — a harness
```

| layer | what lives there | cadence | how it merges |
|---|---|---|---|
| **L2 fast** | local skills, prompts, few-shot blocks | hours | full async merge on held-out score |
| **L1 slow** | harness, context policy, tool router, learned verifier | days | serialised, and **every merge passes the oracle** |
| **L0 frozen** | the oracle, the audit budget, merge permissions, safety constraints | human | the loop may read it, never write it |

## The L1/L2 boundary is measured, not declared

```python
FAST_MAX = 0.30       # blast_radius <= 0.30 -> L2, above -> L1
```

`blast_radius` is an estimate of how much of the task surface an artifact
touches. That makes the boundary a property of *behaviour*, not of a label: a
skill triggered by every task is pulled into the slow layer automatically, while
a harness patch that only affects one task cluster can ride the fast layer.

There is exactly one threshold, and it is here. It used to be re-derived from raw
floats in two other places with a *different* value (`> 0.5`), so an artifact at
0.4 was L1 by governance and treated as L2 everywhere it mattered — it got the
cold-artifact staleness tolerance and no oracle audit at all. `classify()` is now
the single definition, and both the aggregator and the
[audit scheduler](duration-scheduling.md#4-the-audit-scheduler-allocating-oracle-budget) call it.

## L0 is a list, not a threshold

```python
FROZEN_IDS = frozenset({"oracle", "audit_budget", "merge_permissions",
                        "safety_constraints"})
```

This is the one deliberately hand-labelled taxonomy in the system, and the
inconsistency is the point. Nothing about a blast radius can tell you that an
artifact *is the oracle* — that is a structural fact, not a measured one. A
verifier that learns to pass itself is exactly what an estimated layer would fail
to catch.

The names are reserved end to end: `evolve(artifact_id="oracle")` is refused up
front, and `assert_mutable` guards every merge:

```python
from agentdescent import GovernanceError, assert_mutable

assert_mutable(artifact)      # raises GovernanceError if the artifact is L0
```

## Freezing *paths*, not just artifacts

`FROZEN_IDS` freezes a whole artifact by id, which cannot express "this skill may
evolve, but not its test suite". When the artifact is a
[directory](directory-evolution.md), that distinction is the difference between a
measured improvement and a self-graded one:

```python
FileTree(files, frozen=["tests/**", "eval/**", "SAFETY.md"])
```

This is L0 in the file world, and it is enforced **twice** — only the second is a
security boundary:

1. the proposal filter stops the *reflector* from editing those files;
2. the runner overlays the **pristine** copies after materialisation, and the
   test gate is invoked from outside the tree — so candidate code cannot pass by
   rewriting `conftest.py` at run time either.

Without both, the shortest path to a high score is to weaken the thing measuring
it.

## What L1 actually costs

For `blast_radius > 0.30`, `AuditScheduler.force_oracle` returns `True` on every
merge, so the [verifier](verifier.md)'s oracle scores base and candidate before
the commit, and a candidate that does not beat the base on ground truth is
rejected with `oracle-rejected` in
[`result.outcomes()`](evolution.md#why-did-nothing-commit).

The surprising part is that this is **free in agent calls**: the oracle scores
the same artifact on the same held-out set that the acceptance test just scored,
and the engine's evaluation cache serves it. L1 spends the `oracle_budget`
counter, not the model.

## `L1SerialGate` — a primitive, not a path

"At most one L1 diff in evaluation at a time" is a design requirement, and the
shipped runtimes satisfy it *by construction*: every merge decision runs on one
thread — the round barrier in `evolve()`, the single merger thread in
[`async_evolve` and `AsyncAgentDescent`](async.md).

`L1SerialGate` is what would enforce it once merges run concurrently across
processes or hosts. It is tested in isolation for that day. Treat it as a
primitive you may need, not as something currently in the path — the docstring
says so too, because a gate that looks wired in and is not is worse than no gate.

## Choosing a blast radius

| you are evolving | pass | because |
|---|---|---|
| an instruction, a playbook, one skill | `0.2` (default) | local, recoverable, merge on held-out |
| a skill *directory* | `0.2` | same, still local |
| an agent definition, a subagent folder, a tool router | `0.6` | it is a harness: oracle-gate every merge |
| agent code | `0.6` + `frozen=["tests/**"]` | as above, plus it executes |
| a learned verifier | `0.6`, and never name it `oracle` | it judges; audit it hard |

`classify(artifact)` prints which layer you actually landed in — every
[algorithm port](self-evolution-examples.md) does this at startup (the eight benchmark-faithful ports; the MethodPolicy runner sets `blast_radius=0.6` directly), which is how
you tell a configuration mistake from a result.
