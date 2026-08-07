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
