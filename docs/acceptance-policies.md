# Acceptance policies — whether a candidate commits

*Module:* [`agentdescent.defaults`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/defaults.py),
[`agentdescent.advantage`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/advantage.py)
· *Contract:* `AcceptancePolicy.accept(ctx: MergeContext) -> AcceptDecision`

The gate at the end of the merge pipeline: given the merged candidate's
held-out measurement, its history, and its prior, decide commit or refuse —
and say *which kind* of refusal (`AcceptDecision.category`), because "the gate
says it doesn't help" and "it never reached the gate" need opposite fixes.

## Implemented

| Policy | Rule | Reach for it when |
|---|---|---|
| `DefaultAcceptance` | Beta posterior on the improvement, annealed risk `base_delta`, plus a regression guard on the full held-out set | the default; statistical acceptance for noisy rewards |
| `AdvantageAcceptance(inner, strength)` | shifts the prior by the candidate group's standardised advantage (`GroupAdvantage`), then defers to `inner` — wraps, never replaces, so the Beta test and regression guard survive | GRPO-shaped methods (R-Zero here); group evidence should nudge, not override |
| `StableDistanceAcceptance` | penalises candidates that drift far from the `stable` branch — the KL-to-reference analogy with `stable` as the reference | long runs where dev can wander; distance as regularisation |
| `AcceptAnyCompiling` (examples/godel_agent) | accepts everything the validator let through | faithfulness ablations of gateless methods; measuring what the gate is worth |
| `StrictImprovement` (examples/skillopt) | commit only a strict full-held-out improvement — no Beta draw, no annealing | SkillOpt's deliberately harsher gate; small clean val sets where strictness is the algorithm |

Related knobs that are *not* policies: `TrustRegion` / `AdaptiveTrustRegion`
cap diff size before cards ever reach the gate.

```python
evolve(tasks, reward, agent=agent, policies=Policies(
    acceptance=AdvantageAcceptance(DefaultAcceptance(0.5, 64, 4000))))
```
