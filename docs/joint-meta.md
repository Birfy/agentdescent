# Joint multi-slot evolution — search the whole decision plane

*Module:* [`agentdescent.joint_meta`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/joint_meta.py)
· *Issue:* #388 (P7)

Every other rung of the meta stack evolves **one** slot: `meta_run` /
`continuous_meta_run` fix a `slot` and search its value, `meta_population_run`
runs several rules of one slot as islands, and `portfolio_report` only *moves
budget* between slots after the fact. `joint_meta_run` searches several slots
**together**: the artifact is a bundle `{slot: value}`, each outer rollout
installs all the slots and runs the inner search, and the reward is the inner
held-out curve. An effect that only appears when two rules change together —
invisible to two independently evolved rules — is visible here.

```python
from agentdescent import (SourceSlot, joint_meta_run, joint_problem,
                          priority_selection)

specs = {
    "selection": priority_selection(),
    "task_sampler": SourceSlot(initial_value=task_sampler_seed),
}
train = {"algotune": joint_problem(tasks, reward, rounds=4, n_workers=2)}
val = {"algotune-val": joint_problem(val_tasks, reward, rounds=4, n_workers=2)}

report, result = joint_meta_run(
    train, val, slots=list(specs), specs=specs,
    propose=offline_propose, seeds=range(4), val_seeds=range(4, 8),
    isolate=True)
print(report.verdict, report.joint_gain)     # e.g. adopt, +0.4
print(report.slot_gains)                     # per-slot marginal held-out gain
print(report.isolated)                       # slots rolled back as untrusted
```

## The artifact: a bundle

`PoliciesBundle` is both a `Strategy` (for the outer `evolve()`) and a
spec-like object (for validation and the reflector). The state is
`{slot: rendered value}` and the key space is the slot names, so the
aggregator's existing rules apply with no new machinery:

* a proposal names one or more slots as `=== <slot> ===` blocks;
* edits to **different** slots union-merge (a coherent multi-slot change);
* two edits to the **same** slot contradict and are resolved on held-out score;
* a bare proposal (no header) is accepted only when exactly **one** slot's gate
  compiles it — otherwise it is ambiguous and refused.

## Isolating untrusted rules

`joint_meta_validate` attributes the joint change per slot: each slot's
**marginal** held-out gain is its edit against the seed bundle with every other
slot held at the seed. A slot whose marginal gain is below `min_gain` is
**untrusted** — the joint gain may be positive on the others' backs, but this
slot did not pay for itself on held-out problems.

* `isolate=True` (default): the untrusted slot is rolled back to its seed value
  and the corrected bundle re-validated — adopted only if the corrected gain
  still clears `min_gain`.
* `isolate=False`: any untrusted slot rejects the whole bundle (fail-closed).

## The archive is unchanged

An adopted bundle writes **one ordinary `meta-rule` record per slot** (with its
own `problem_descriptor`), so `_archived_rules`, `exam_archive` and
`continuous_meta_run` keep seeding and examining it with no special case. Pass
`store=` and the next run seeds its bundle from the archive's best per-slot
rules instead of each spec's raw initial value.

## Isolation contract

As with `meta_run`, `val_problems` and `val_seeds` must be isolated from the
training set: overlapping problem names and overlapping seed sets both raise.
