# Sparse audit — a cheap verifier against ground truth

*Module:* [`agentdescent.audit`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/audit/__init__.py)
· *API:* [`AuditedReward`](api.md#the-sparse-audit-layer)
· [`GoldAnswer`, `DeferredOracle`](api.md#audit-oracle-sources)
· [`AuditStore`](api.md#the-audit-store)

The loop optimises whatever `reward` returns. When that reward is a *fact* about
the output — an exact match, a unit test, a checker — there is nothing to audit.
When it is an **agent judging the output**, or a learned scorer, or a heuristic,
the loop is optimising a proxy, and a proxy can be wrong in a *direction*:
systematically generous about a kind of answer it happens to like.

Nothing inside the loop can notice. Every gate reads the same proxy, so a change
that games it is indistinguishable from a change that improves. This module adds
the only thing that can notice — occasionally asking someone who knows.

!!! info "This is not the default path"
    With the shipped [`ThreeLayerVerifier`](verifier.md) the acceptance gate and
    the oracle are literally the same call (`eval_fn` on the full held-out set),
    so there is no bias between them to estimate. This page is for the case where
    **your `reward` is itself a proxy** and truth lives somewhere more expensive.

## The two sources

| | cheap verifier | oracle |
|---|---|---|
| what it is | an agent judging the output, a learned scorer, a heuristic | a gold answer, a checker, a wet-lab experiment, a human |
| how often | every rollout | a sampled few percent |
| when it answers | now | now, or next week |
| type | `(task, output) -> float` | `(task, output) -> float` |

The last row is the whole design. Both are
[`Reward`](evolution.md)-shaped, so pairing them is a
matter of calling both on the same output rather than of building a second
evaluation stack.

## Why it hooks the reward, not the verifier

[`ThreeLayerVerifier`](verifier.md) looks like the natural home for an oracle. It
is the wrong one, for three reasons that all point the same way:

1. **Its layers score `(artifact, tasks) -> float`** — an aggregate over a task
   list. Estimating a bias needs *paired* observations, and an aggregate has
   already averaged the pairing away.
2. **A verifier-level oracle is asked for a fresh measurement**, so `f` and `Y`
   would come from two different rollouts. Their difference would then carry
   rollout variance on top of the bias, and no amount of sampling separates them
   again. At the reward level both score the **same output**.
3. **The merge path calls `full_eval` synchronously.**
   [`Aggregator._audit`](aggregator.md) will happily block a merger on it — which
   is fine for a checker and catastrophic for an experiment that returns on
   Thursday.

Hooking the reward instead puts the audit *below* the verifier, so the aggregator,
the tournament and both commit gates are untouched. Two properties follow from
the shape of the code rather than from anyone remembering them:

- **Enabling the audit cannot change the run.** `AuditedReward.__call__` returns
  the verifier's score. The oracle's answer is written to disk and goes nowhere
  near a gate.
- **The audit cannot block the loop.** Submitting is decoupled from resolving.

## Gold answers — truth that returns now

```python
from agentdescent import evolve, AuditedReward, GoldAnswer, AuditStore

def llm_judge(task, output):          # cheap, biased, runs on every rollout
    return grade_with_a_model(task.prompt, output)

def exact_match(task, output):        # truth, runs on ~10% of them
    return 1.0 if output.strip() == task.meta["gold"] else 0.0

audited = AuditedReward(
    llm_judge,
    oracle=GoldAnswer(exact_match),
    store=AuditStore("runs/audit.jsonl"),
    sample_rate=0.1,
    version_extra={"model": "judge-v3", "prompt_sha": "9f21ac"},
)

result = evolve(tasks, audited, agent=agent, rounds=20)
```

Afterwards, the paired observations are on disk and in memory:

```python
from agentdescent.audit import summarise

print(summarise(audited.calibration_set()))
# {'n_records': 41, 'n_resolved': 41, 'mean_residual': 0.13}
```

A positive `mean_residual` means the judge scored outputs **higher** than the
truth did — the direction that makes a loop accept changes that improved nothing.

## Experiments — truth that returns next week

```python
from agentdescent import AuditedReward, DeferredOracle, AuditStore

audited = AuditedReward(simulator_score, oracle=DeferredOracle(),
                        store=AuditStore("runs/audit.jsonl"), sample_rate=0.05)
evolve(tasks, audited, agent=agent)
```

`DeferredOracle.submit` writes the question down and returns immediately. The
record keeps the **output**, because that is what the experiment is run against,
and the store is JSONL on disk, so answering happens in another process on
another day:

```python
from agentdescent import AuditStore, resolve_from_mapping

store = AuditStore("runs/audit.jsonl")
for rec in store.pending():
    print(rec.record_id, rec.output)      # go and measure these

resolve_from_mapping(store, {"a1b2c3d4...": 0.83, "e5f6a7b8...": 0.11})
```

[`NullOracle`](api.md#audit-oracle-sources) is the same thing with nobody
queued yet: the questions still accumulate, on units drawn by a known
probability rule rather than by whoever happened to look.

## The three fields that cannot be reconstructed later

An audit is a probability sample or it is an anecdote. Three things have to be
recorded **at the time**, because no amount of later care recovers them:

| field | why |
|---|---|
| `inclusion_prob` | The probability this unit had of being drawn. Without it the sample weights nothing and estimates nothing. The sampling policy may have changed since. |
| `verifier_version` | Which verifier produced the score. A correction estimated for one verifier says nothing about the next, and the failure is silent — the numbers still compute. |
| `output` | What the oracle scores. Re-running the artifact gives a different output and folds rollout variance into the residual. |

`verifier_version` is derived from the verifier's source by
[`verifier_fingerprint`](api.md#audit-records). **Pass `version_extra`
whenever the verifier is an agent** — an LLM judge's behaviour lives in its
prompt and its model id, neither of which appears in the source of the function
that calls it, so without it an edited prompt keeps the old fingerprint and the
old correction goes on being applied to a different instrument.

## Two pools, never mixed

Every audited unit is assigned a `Purpose`:

- **`CALIBRATION`** — estimating the verifier's bias. Never shown to whoever
  edits the verifier.
- **`IMPROVEMENT`** — diagnosing and fixing the verifier. Never enters a
  calibration set.

The split is not bookkeeping. A label used to *change* the verifier cannot also
*calibrate* it: the change was chosen to make those very units agree, so the
residual measured on them is optimistically biased by construction. The
resulting estimate is confidently wrong rather than noisy, and it errs towards
"the verifier is honest" — the direction that does no visible damage for weeks.

`AuditStore.for_calibration` asserts the separation rather than filtering for it,
because a query condition is one careless edit away from being widened and an
assertion is not.

## Sampling

`sample_rate` is the flat case. When the residual varies across the score range —
it usually does; a judge is most wrong near its own decision boundary — spend the
budget where the variance is:

```python
audited = AuditedReward(
    llm_judge,
    oracle=GoldAnswer(exact_match),
    stratify=lambda task, output, score: (
        "boundary" if 0.45 < score < 0.55 else "clear"),
    rates={"boundary": 0.5, "clear": 0.02},
)
```

The inclusion draw is seeded **per unit**, from
`(seed, verifier_version, task_id, output)`, not taken from a shared stream.
Evaluation runs on up to `eval_concurrency` threads, and a shared stream would
make inclusion depend on which thread arrived first — so the same run replayed
would audit a different sample and `sampler_seed` would document nothing.
[`ThreeLayerVerifier.learned_eval`](verifier.md) seeds per-artifact for the same
reason.

## Estimating the bias

[`residual_bias`](api.md#audit-estimation) turns resolved records into
`Delta = E[f - Y]` with a 95% interval:

```python
from agentdescent import residual_bias

print(residual_bias(audited.calibration_set()))
# {'n': 214, 'delta': 0.147, 'ci': (0.089, 0.206), 'se': 0.030,
#  'f_mean': 0.71, 'y_mean': 0.56, 'disagree': 0.19}
```

It is the **Hájek** (inclusion-probability-weighted) mean with a percentile
bootstrap. Weighted from the start because an unweighted mean is correct only
while every unit shares one inclusion probability — it is silently wrong the day
someone raises the rate on the boundary stratum, and plausible either way.
Bootstrap rather than a `t` interval because the residual of two binary scores
takes three values with most of its mass at zero, and because the estimator is a
ratio.

`disagree` is worth reading beside `delta`: a small bias spread over every unit
and a large bias on a few units give the same mean and call for different fixes.

### What is deliberately not here

**Prediction-powered inference.** PPI additionally borrows strength from the
*unlabelled* verifier scores — units `f` scored that the oracle never saw — using
a cross-fitted coefficient, per-stratum weights and a `t` quantile. That earns
its keep when labels are scarce and unlabelled scores are plentiful; it also has
several ways to be subtly wrong that a coverage test does not catch, so it
belongs in a module with its own golden vectors and mutation tests.

The two are a baseline and a refinement, not alternatives: same point estimate in
expectation, wider interval. Any PPI estimator added later is **measured against
this one** rather than trusted over it.

`summarise` is neither — a raw *unweighted* mean for eyeballing a run, named so
nobody wires it into a gate.
