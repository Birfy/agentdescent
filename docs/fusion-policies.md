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
