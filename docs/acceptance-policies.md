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
    acceptance=AdvantageAcceptance()))          # wraps the shipped gate
```

`AdvantageAcceptance()` and `StableDistanceAcceptance()` wrap the shipped
`DefaultAcceptance` when given no `inner`. Its three thresholds
(`base_delta`, `anneal_half_life`, `accept_samples`) are left unset and the
aggregator fills them from the run's `AggregatorConfig` when the policy is
installed — so the wrapped gate and `agg_config=` cannot disagree, which they
silently could when the example above read
`DefaultAcceptance(0.5, 64, 4000)`. Pass a value to pin one:
`AdvantageAcceptance(DefaultAcceptance(base_delta=0.3))` keeps `0.3` and takes
the other two from the run. `DefaultAcceptance.from_config(cfg)` is the fully
pinned form.

## Installing a policy: the two optional hooks

A policy may need two things only the engine has: the verifier, for anything
that ranks, and the aggregator's config, for anything that reads a threshold.
When a policy is installed the aggregator offers both through two *optional*
methods — `bind(verifier)` and `configure(config)` — and every shipped wrapper
forwards them to its `inner` rule. A policy with neither is left alone. A
shipped default used **without** having been installed (driven by hand, outside
`evolve()`) raises `PolicyUnboundError` naming the missing piece rather than
failing on a `None` in the middle of a merge; call the hook yourself, or use
`install_policy(policy, verifier, config)` from `agentdescent.aggregator`.

## What the default knows that a replacement must be told

**Acceptance reads the full held-out set, never the cheap layer.**
`MergeContext` carries both (`base_counts` vs `base_cheap`) because the
regression guard once read the cheap one, which `cheap_eval_tasks` sub-samples
— so a four-task sample could veto a commit the full-set test had just
approved. Ranking may use the cheap numbers; deciding to commit may not.
