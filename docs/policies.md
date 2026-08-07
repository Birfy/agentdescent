# Choosing policies — the decision plane

*Module:* [`agentdescent.policies`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/policies.py)

Every decision `evolve()` makes is an object you can swap, and they all travel
in one argument:

```python
from agentdescent import Policies, evolve
from agentdescent.selection import Beam
from agentdescent.sampling import DifficultyWeighted
from agentdescent.advantage import AdvantageAcceptance
from agentdescent.fusion import reflective_merge

evolve(tasks, reward, agent=agent, policies=Policies(
    selection=Beam(4),
    task_sampler=DifficultyWeighted(),
    acceptance=AdvantageAcceptance(inner=my_gate),
    **reflective_merge(completion),
))
```

Two guarantees make the bundle trustworthy. **Nothing is silently ignored**:
each engine declares what it honours, and `Policies.require_supported` raises
on anything else — a custom acceptance rule either runs or refuses loudly.
**`None` means today's behaviour**: `Policies()` and passing nothing are the
same run, so adding a policy never changes an existing measurement.

## Which seam is my mechanism?

An algorithm's distinctive mechanism lives in exactly one of three layers:

1. **The artifact's shape** — what a proposal *is* — belongs in the
   [strategy](strategies.md), not a policy. Append-only memory, a keyed
   library, a single replaced slot.
2. **A decision the engine already makes** — which parent, which task, which
   prior, merge or drop — belongs in a `Policies` field. This page's table.
3. **Pure actor text** — prompts and tools — belongs in the definition and
   needs no seam at all.

| Field | The decision | Page | Shipped implementations |
|---|---|---|---|
| `task_sampler` | which task the next rollout spends | [Task sampling](sampling.md) | `RoundRobin`, `DifficultyWeighted` |
| `selection` | which candidate the next batch starts from | [Candidate selection](selection.md) | `SingleHead`, `Beam`, `ParetoFrontier`, `Archive`, `MCTS` (+ examples-level `BinaryTournament`, `SoftMixed`) |
| `proposal` | how rollout evidence becomes proposals | [Proposal policies](proposal-policies.md) | protocol only — write your own |
| `conflict` | what happens to contradicting diffs | [Conflict policies](conflict-policies.md) | `DefaultConflict`, `KeepContradictions`, `AdvantageConflict` |
| `fusion` | whether and how survivors merge | [Fusion policies](fusion-policies.md) | `DefaultFusion`, `ReflectiveFusion` |
| `acceptance` | whether the merged candidate commits | [Acceptance policies](acceptance-policies.md) | `DefaultAcceptance`, `AdvantageAcceptance`, `StableDistanceAcceptance` |
| `promotion` | when dev reaches stable | [Promotion policies](promotion-policies.md) | `DefaultPromotion` |
| `staleness` | what a lagging diff is worth | [Staleness policies](staleness.md) | `Full`, `Guarded`, `Reflective` |

The remaining fields are machinery, not algorithm: `verifier`, `ledger`,
`eval_cache`, `executor`, `evaluator`, `sandbox_*` — see
[the verifier](verifier.md), [the ledger](ledger.md), and
[execution](execution.md).

## When one decision is not enough

Some optimizers change *how the decisions compose* — a population with its own
admission rule, per-instance score rows, an archive driving parent switches.
That is the [`aggregator_factory` exit](aggregator-factory.md): replace (or
subclass) the whole optimizer while every policy above stays available to the
replacement. Rule of thumb: **swap a field first; reach for the factory when
your mechanism needs state the pipeline does not keep.**
