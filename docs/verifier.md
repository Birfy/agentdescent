# The verifier — rule, learned, oracle

*Module:* [`agentdescent.verifier`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/verifier.py)
· *API:* [`ThreeLayerVerifier`, `VerifierBudget`](api.md#the-verifier)

The [aggregator](aggregator.md) needs to score candidates at two very different
price points, and conflating them is what makes a merge-based loop unaffordable:

| layer | used for | cost | budget |
|---|---|---|---|
| **rule** | ranking candidates against each other | a small subset | none |
| **learned** | the same, plus an uncertainty estimate | a small subset | none |
| **oracle** | ground truth, before a high-impact commit | the full held-out set | capped |

Ranking happens constantly — every conflict resolution, every entry in a fusion
tournament. Committing happens rarely. Paying oracle prices for ranking is the
default mistake, and it is expensive in exactly the case that matters:

!!! danger "`eval_fn` runs your agent"
    On an LLM workload every held-out evaluation is a full sweep of real model
    calls. `cheap_eval_tasks=None` (the default) pins the cheap layer to the
    *whole* held-out set, so ranking N candidates costs N full sweeps. Set
    `evolve(cheap_eval_tasks=4)` and ranking becomes cheap. Both gates that decide
    a *commit* — the Beta-posterior acceptance test and the regression guard
    beside it — read the full held-out set, so this trades ranking precision and
    nothing else. The [directory entry points](directory-evolution.md) default it to 4
    for this reason.

## The three methods that matter

```python
verifier.cheap_eval(artifact)     # 0.5 * rule + 0.5 * learned -- ranking
verifier.eval_counts(artifact)    # (successes, failures) on the FULL held-out set
verifier.oracle_eval(artifact)    # ground truth, spends budget
```

`eval_counts` is what feeds the Beta-posterior acceptance test, and it never
sub-samples: the acceptance decision has to rest on an honest sample size, or the
posterior is confident about noise.

!!! note "It also feeds the regression guard, and that used to be a real hole"
    The aggregator refuses a candidate that scores *worse* than the incumbent even
    when the posterior likes it. That guard read the **cheap** layer until
    recently, so with `cheap_eval_tasks=4` a four-task sample could veto a commit
    the full-set test had just approved — while three source comments and two doc
    pages promised sub-sampling could not touch commit safety. It now reads the
    full-set rates `eval_counts` has already produced.

## The sample is fixed, and that is a correctness property

The cheap layers score a **stable** subset, drawn once per size:

```python
ThreeLayerVerifier(eval_fn, held_out, rule_subset=8)
```

A fresh draw per call would score candidate A on `{1,3,5}` and candidate B on
`{2,4,6}` and call the difference a winner. The aggregator compares candidates
head to head — `_resolve_conflicts` pits two diffs against each other,
`_tournament` ranks every candidate — so like-for-like comparison is not a nicety.
It also defeats the evaluation cache, which memoises per `(artifact, task)`.

Overfitting to that fixed subset is bounded by the acceptance test, which never
sub-samples.

## The oracle budget is a real cap

```python
evolve(..., oracle_budget=200)
```

Once spent, `oracle_eval` falls back to the cheap layer rather than spending
money it was told not to spend. Note that this only saves anything when
`cheap_eval_tasks` makes the cheap layer genuinely cheaper — the two knobs go
together, and setting `oracle_budget` alone does nothing.

!!! tip "The oracle gate is usually free"
    For an [L1 artifact](governance.md) every merge is forced through the oracle
    — and it costs no extra agent calls. `oracle_eval` and `eval_counts` call the
    same `eval_fn` over the same held-out set, and the engine's evaluation cache
    is keyed on `(rendered artifact, task id)`, so the second call is served from
    the first. L1 spends the *counter*, not the model. Evolving a harness is not
    more expensive than evolving a skill.

## Trust, and why it has to be measurable for free

The [audit scheduler](duration-scheduling.md#4-the-audit-scheduler-allocating-oracle-budget) prioritises
oracle spending by `blast_radius * uncertainty / trust`, where trust is "how
often does the cheap layer agree with the full held-out set".

That signal must be obtainable **without** spending oracle budget, or it is
circular — and it was: `force_oracle` fired on low trust, and the only writer of
trust sat inside that branch, so for any artifact below the threshold the
condition could never become true and the audit never ran at all. Measured on the
default `blast_radius=0.2`: `oracle_calls_used == 0` for a whole run, trust
pinned at its initial 1.0.

The fix is free: `eval_counts` already scored base and candidate on the full set
for the acceptance test, so comparing that verdict with the cheap layer's costs
nothing and happens on every merge.

## Bringing your own

`ThreeLayerVerifier` is a reference implementation, not a requirement. It takes
one function:

```python
ThreeLayerVerifier(eval_fn=lambda artifact, tasks: artifact.score(tasks),
                   held_out=held_out_tasks,
                   rule_subset=4,
                   budget=VerifierBudget(oracle_calls_remaining=200))
```

The reference aggregator calls **four** methods, so a substitute needs all four —
building to the three above raises `AttributeError` from inside the merge, after
the run has already spent its rollouts:

```python
cheap_eval(artifact) -> float                 # ranking
learned_eval(artifact) -> (score, uncertainty) # the audit priority's uncertainty term
eval_counts(artifact) -> (successes, failures) # the acceptance test, full set
oracle_eval(artifact) -> float                 # ground truth, spends budget
```

An [`aggregator_factory`](aggregator.md#replacing-aggregator_factory-aggregatorprotocol)
receives the verifier, so a custom optimizer that does not want an audit gate can
ignore whichever of these it never calls.

A learned verifier is itself an evolvable artifact — and one that must never
evolve itself. That is what the [L0 frozen layer](governance.md#l0-is-a-list-not-a-threshold)
is for: an artifact that can rewrite the thing that judges it is exactly what an
*estimated* governance layer would fail to catch.
