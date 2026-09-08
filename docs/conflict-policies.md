# Conflict policies — contradicting diffs

*Module:* [`agentdescent.defaults`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/defaults.py),
[`agentdescent.fusion`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/fusion.py)
· *Contract:* `ConflictPolicy.resolve(artifact, cards) -> (survivors, dropped)`

Two diffs contradict when they write the same key with different values.
Overlap alone is not a conflict — two workers proposing the same rule
collapse to one. This step runs **before** fusion, which is why the choice
here decides what fusion ever gets to see.

## Implemented

| Policy | Rule | Reach for it when |
|---|---|---|
| `DefaultConflict` | drop the contradicting loser, keep whichever scores better (PCGrad-style) | the default; single-key artifacts where one value must win |
| `KeepContradictions` | pass contradictions through untouched, for fusion to resolve | **only as a pair** with `ReflectiveFusion` — installed alone it changes nothing, installed with it the contradictions are merged instead of dropped |
| `AdvantageConflict` | break the tie by group-relative advantage instead of raw score | group-standardised evidence exists and raw scores are noisy across task clusters |

Use `reflective_merge(completion)` to install the `KeepContradictions` +
`ReflectiveFusion` pair correctly — see [fusion policies](fusion-policies.md).

```python
from agentdescent import Policies, evolve
from agentdescent.advantage import AdvantageConflict

evolve(tasks, reward, agent=agent, policies=Policies(
    conflict=AdvantageConflict(margin=0.5)))    # falls back to DefaultConflict
```

`AdvantageConflict()` wraps `DefaultConflict` when given no `inner`. That rule
scores a tie on the verifier, which is built inside the engine and which a
caller cannot supply — so the aggregator hands it over through the policy's
optional `bind(verifier)` hook when the policy is installed, and the wrapper
forwards it. Nothing here has to see a verifier. Before this only fusion was
bound, and a wrapped default conflict rule could be installed only from inside
an `aggregator_factory`. See
[acceptance policies](acceptance-policies.md#installing-a-policy-the-two-optional-hooks)
for the hooks themselves.
