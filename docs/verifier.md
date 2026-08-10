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


## The evaluation group

Evaluation and exploration call the same function and are different workloads. A
rollout is long-tailed, frequently fails, and is one opinion among many — losing
one costs a little evidence. An evaluation is batched, cacheable, and decides
whether a change is committed — losing one costs the decision. Sizing them
together means sizing them for whichever matters less.

```python
from agentdescent import Policies, evolve
from agentdescent.evaluator import EvaluatorGroup

evolve(tasks, reward, agent=agent,
       n_workers=8,                                    # exploration
       policies=Policies(evaluator=EvaluatorGroup(4)))  # the gate
```

`eval_concurrency=` still works and builds the group for you; the injectable one
exists so evaluation can be bounded, observed (`group.stats()`) and eventually
given a different substrate from rollouts.

!!! note "It used to be a pool per call"
    `score()` is called once per gate — each round's held-out measurement and,
    far more often, every per-candidate comparison — and built a fresh
    `ThreadPoolExecutor` each time: **83 of them in a six-round run over twelve
    tasks**. A pool created per call has a size and nothing else: nothing to
    bound, nothing to observe, nothing to replace.

## The evaluation cache

Evaluation is the expensive half of a run —
[measured](efficiency.md#the-other-axis-eval_concurrency), the same work takes
3.6 s against 1.2 s at `eval_concurrency` 1 and 8, and every one of those seconds
is a real rollout. So evaluations are memoised, keyed on **what the artifact
renders to**, the task id, and the **environment's fingerprint**.

```python
from agentdescent import FileCache, Policies, evolve

evolve(tasks, reward, agent=agent,
       policies=Policies(eval_cache=FileCache("~/.cache/agentdescent")))
```

Three things the key and the cache have to get right, each of which was wrong at
some point:

* **Not the state — what it renders to.** `eval_one` passes only `render()` to
  `run`, so two states that render identically cannot score differently. Keying
  on state made a strategy carrying bookkeeping beside the artifact (ADAS keeps a
  design's name and rationale) re-evaluate the whole held-out set because a label
  changed.
* **Single-flight.** A plain dictionary checks, releases its lock, then computes,
  so N concurrent callers for one uncomputed key all miss and all compute — which
  is wasteful in exactly the case caching exists for. The first caller computes
  and the rest wait for that result.
* **The environment is part of the identity.** A `code_runner` score depends on
  what `setup_cmd` installed, the python minor version, whether there was a
  network. Sharing a cache across images without the fingerprint means one
  environment's measurement answers another environment's question — and that
  number is what the commit gates read.

`FileCache` is a directory, so two processes on one machine stop paying twice for
the same gate without a server between them. A network backend is the same
protocol and belongs with the cross-machine work that would justify running one.

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

!!! tip "The oracle gate is free, because it is not a second measurement"
    For an [L1 artifact](governance.md) every merge is forced through the oracle,
    and `ThreeLayerVerifier`'s oracle scores **exactly** the set `eval_counts`
    scores — same `eval_fn`, same held-out set. So the aggregator reuses the
    full-set rates it has already measured for the acceptance test instead of
    asking for them again. `ThreeLayerVerifier.oracle_shares_full_set` is what
    says so; a substitute whose oracle is a genuinely independent measurement
    leaves it undefined and keeps being called.

    The verdict is identical either way. Evolving a harness is not more expensive
    than evolving a skill, and with the shipped verifier an L1 run now reports
    `oracle_calls_used == 0` — the audit ran, nothing had to be bought. Use
    `AuditScheduler.audits` to ask whether the gate opened.

!!! danger "Why reuse, and not just a saving"
    `oracle_eval` degrades to `rule_eval` when the budget runs out, and
    `rule_eval` is the **sub-sample**. So an exhausted budget silently turned the
    audit gate into a sub-sample veto — measured, a candidate that took the
    full-set rate from 0.5 to 1.0 was reported `oracle-rejected` because a
    two-task sample scored both sides at 0.5.

    That contradicted the two promises above it on this page: sub-sampling trades
    ranking precision and never decides a commit. The merge path no longer
    reaches the fallback. If you bring your own verifier, either keep
    `oracle_eval` exact or set `oracle_shares_full_set` — an oracle that quietly
    gets cheaper must not hold a veto.

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

There is also one **optional** attribute, read with a default so a substitute
that omits it is unaffected:

```python
oracle_shares_full_set = True    # oracle_eval scores the same set eval_counts does
```

Set it when both are the same measurement, and the aggregator will reuse the
rates it already has rather than asking twice. Leave it out when your oracle is
genuinely independent — then it is called, and it must stay exact: an
`oracle_eval` that gets cheaper under budget pressure holds a veto over commits.

An [`aggregator_factory`](aggregator.md#replacing-aggregator_factory-aggregatorprotocol)
receives the verifier, so a custom optimizer that does not want an audit gate can
ignore whichever of these it never calls.

A learned verifier is itself an evolvable artifact — and one that must never
evolve itself. That is what the [L0 frozen layer](governance.md#l0-is-a-list-not-a-threshold)
is for: an artifact that can rewrite the thing that judges it is exactly what an
*estimated* governance layer would fail to catch.
