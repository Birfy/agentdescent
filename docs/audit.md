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

## Improving the verifier — and the trap in it

Calibration corrects the verifier's mean error. The other question is whether the
verifier can be made less wrong in the first place, and
[`agentdescent.audit.diagnose`](api.md#diagnosing-the-verifier) is for that.

!!! danger "Do not optimise the verifier against `delta_hat`"
    A mean can be driven to zero by adding errors in the *opposite* direction,
    and that is not an improvement. Measured on a real 177-pair HotpotQA audit,
    two obviously-correct hard rules — reject an answer that echoes the question,
    reject one far shorter than the reference — produced this:

    | | before | after |
    |---|---|---|
    | `delta` | +0.175 | **+0.051** (−71%) |
    | `sigma` | 0.381 | **0.417** (up) |
    | disagreement | 0.175 | 0.175 (unchanged) |
    | false-negative rate | — | **22.4%** |

    Eleven corrections and eleven fresh mistakes. The bias fell because the
    errors now cancel, not because the verifier learned anything.

    **`sigma` is the target here.** `delta` is what the calibrator already
    handles, and optimising the thing that is already handled breaks the thing
    that is not.

### Sorted by what it would take to fix

```python
from agentdescent.audit import classify_disagreements, reference_classifier

report = classify_disagreements(
    store.for_improvement(version),          # never the calibration pool
    reference_classifier(normalise, lambda ctx: ctx.gold,
                         spec_gap_when=..., ambiguous_when=...),
    context={task.id: task for task in tasks},
)
print(report.to_markdown())
```

| kind | fix | cost |
|---|---|---|
| `FORMATTING` | normalise both sides | nothing; cannot introduce a judgement |
| `SPEC_GAP` | a hard rule | cheap — and the one most likely to look free |
| `AMBIGUOUS` | **none** | a second oracle would disagree too |
| `JUDGMENT` | a better judge — prompt, model, thinking | expensive, and last |

The order is the point: the first two need no model and no training, and are
usually most of the residual. A diagnosis that jumps to "the judge needs to be
smarter" is skipping the cheap majority.

`Direction` is recorded beside `Kind` because a fix that trades `OVER` errors for
`UNDER` ones looks like progress in every summary that omits it — which is
exactly what the table above is.

### The floor

`report.floor_sigma` is what `sigma` would be if every non-ambiguous
disagreement were fixed perfectly. It is not zero, and chasing below it is not a
plan to improve the verifier — it is a plan to redefine correctness.

From the same audit, gold `'Robert Erskine Childers DSC'` against an answer of
`'Robert Erskine Childers'`, or `'from 1986 to 2013'` against `'1986 to 2013'`:
the judge said these were right, exact match said they were wrong, and the judge
has the better case. Driving those out means training the judge *into* exact
match, which is what having a judge was supposed to avoid.

### Measure a fix on the answers it was not aimed at

```python
from agentdescent.audit import evaluate_fix

got = evaluate_fix(labelled, my_rule, context=tasks_by_id)
print(got.to_markdown())      # helps / does not help, by sigma
```

`evaluate_fix` scores a proposed change against **every** labelled pair, not the
disagreements it targets. Restricted to its targets, the two-rule fix above
removes eleven errors, breaks nothing, and cuts both the disagreement rate and
the bias by 35% — a clean win by every number a person reaches for. On the whole
set it also breaks eleven correct judgements.

The rule was not a bad rule. It was a rule nobody had measured against the
answers it was not aimed at.

`FixReport.helps` reads `sigma_after < sigma_before` and nothing else, for the
reason at the top of this section.

## Allocation — where the budget should go

A flat rate spends the budget where the *units* are. What sets the width of the
correction is where the verifier is *unreliable*, and those are different places:
a layer the verifier gets right every time contributes nothing to the interval no
matter how many of its units you label.

[Neyman allocation](api.md#audit-allocation) says it exactly — `n_h ∝ W_h · sd_h`,
the layer's population share times the standard deviation of the **residual**
`f − Y` within it:

```python
from agentdescent.audit import (AuditPolicy, boundary_stratifier,
                                observed_weights, plan_audit, resid_sd_from)

policy = AuditPolicy(enabled=True, target_halfwidth=0.03)
plan = plan_audit(
    policy,
    weights=observed_weights(store, version),      # counts from the last run
    resid_sd=resid_sd_from(previous_ppi_result),   # where it was unreliable
    expected_units=40_000,
)

audited = AuditedReward(
    llm_judge, oracle=GoldAnswer(exact_match), store=store,
    stratify=boundary_stratifier(threshold=0.5, width=policy.boundary_width),
    rates=plan.rates, sample_rate=plan.default_rate,
    calibration_fraction=policy.calibration_fraction,
)
```

`target_halfwidth` is a specification rather than a wish: under Neyman
allocation `se = Σ(W_h·sd_h) / √n`, so the total label budget follows from the
half-width directly — and halving the half-width costs four times the labels.

!!! danger "`resid_sd`, not `sd(Y)`"
    They are different quantities, both plausible here, and the wrong one
    produces a plan that is merely *suboptimal* — so it survives review. `sd(Y)`
    sends the budget to whichever layer has the most variable outcome; `sd(f−Y)`
    sends it to the layer where the verifier is least trustworthy. A layer whose
    outcome swings wildly but which the verifier tracks perfectly deserves almost
    no labels at all.

### Rates, not a chosen set of units

The plan this implements had the sampler take a generation's units and hand back
which ones to send. That shape does not fit the tap, which sees one
`(task, output)` at a time and decides on the spot with a draw seeded from the
unit itself — which is what makes inclusion independent of thread scheduling.

The allocation survives the translation intact: `n_h` units out of an expected
`W_h · N` is an inclusion probability of `n_h / (W_h · N)`. So this plans
**rates**, the tap keeps deciding per unit, and the allocation is the same one.
The cost is that rates are set from an *expected* population, so realised counts
land near the plan rather than on it.

### Three decisions worth knowing about

**Floors are applied after the allocation, never before.** When a layer is raised
to its floor, the remaining budget is still split by Neyman rather than scaled
down proportionally. `min_dominant` defaults to `MIN_N_DOMINANT` — the same
number as the coverage warning, because they are the same fact, and letting them
drift apart is how a floor stops meaning anything.

**A layer with no history is over-sampled, not under-sampled.** Missing residual
sds are filled at the *largest measured* one. The asymmetry decides it:
under-sampling a layer nobody has measured keeps it unmeasured, which is
self-perpetuating; over-sampling costs budget once and self-corrects the moment
there is a real number. (An earlier version filled with the constant `1.0`, which
against a layer measured at `0.4` handed the unmeasured one 2.5× the allocation
for no reason but the units the constant happened to be written in.)

**A stratum the plan never saw gets a rate of zero.** Not a small default:
sampling it would put units into the estimate under an inclusion probability
nobody chose, and that is the one field that cannot be reconstructed afterwards.

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

`summarise` is neither an estimator nor this one — a raw *unweighted* mean for
eyeballing a run, named so nobody wires it into a gate.

## Prediction-powered inference — the refinement

`residual_bias` uses only the audited units. Most of a run's units were scored by
the verifier and never sent to the oracle, and
[`ppi_mean_stratified`](api.md#prediction-powered-inference) puts those to work:

```python
from agentdescent.audit import Stratum, ppi_mean_stratified

result = ppi_mean_stratified([
    Stratum("accepted", weight=0.55, f_lab=..., y_lab=..., f_unlab=...),
    Stratum("boundary", weight=0.15, f_lab=..., y_lab=..., f_unlab=...),
    Stratum("rejected", weight=0.30, f_lab=..., y_lab=..., f_unlab=...),
])
result.theta, result.ci, result.se, result.gain_factor
```

Per stratum the estimate is

```
theta = lam * mean(f_unlab)  +  mean(y_lab - lam * f_lab)
```

At `lam = 0` this is the labelled-only mean — so **a useless verifier costs
nothing**, which is what makes it safe to switch on. At `lam = 1` it is the
classical PPI rectifier. In between, `lam` minimises

```
Var = lam² · Var(f_unlab) / N  +  Var(y - lam·f) / n
```

`gain_factor` reports `Var(labelled-only) / Var(this)` — the factor by which the
oracle budget was effectively multiplied. **`gain_factor → 1` means the verifier
carries no usable signal**, and the answer is a better verifier, not a bigger
audit.

!!! warning "`weight` is the population share, not the sample share"
    They differ by exactly the amount stratification was introduced to create.
    Using the sample share turns a stratified sample back into a simple one; on
    the test workload coverage falls from 0.94 to **0.01**.

### Four things that are easy to get wrong here

| | why it bites |
|---|---|
| `lam` fitted on the labels it is applied to | The residuals look smaller than they are. Reported SE comes in ~7% low and coverage drops to 0.91. Fixed by K-fold cross-fitting — **run the coverage test before touching `_lambda_crossfit`**. |
| a `z` quantile instead of `t` | At 40–100 labels per stratum the normal quantile is 1–2% too small, and a 2%-narrow interval covers 93% while claiming 95%. |
| fewer than `MIN_N_DOMINANT` (80) labels in the heaviest stratum | Coverage is about 0.92, not 0.95 — the skew of a binary outcome at small n. Not a bug, so it is a warning and a locked test rather than a fix. |
| dropping `lam² · Var(f_unlab) / N` | One missing term. The estimate does not move, the interval looks normal, coverage falls ~3pp. Nothing but a replication study finds it. |

### How it is guarded

`residual_bias` is the baseline PPI has to beat, and the two are checked against
each other rather than one being trusted over the other:

- **Coverage** — 400 replications must cover at the nominal rate, and the
  reported SE must match the actual spread of the estimates to within 7%.
- **Mutations** — three deliberately wrong estimators, whose coverage must
  collapse: ignoring stratum weights (0.94 → 0.01), imputing `f` as truth
  (→ 0.00), dropping the unlabelled variance term (→ 0.91). They test the
  *coverage suite*, not the estimator: a coverage run that passes at 0.95 proves
  nothing until you know it would fail at 0.50.
- **Golden vectors** — six recorded inputs spanning the regimes (useless
  verifier, perfect verifier, no unlabelled units, weights carrying the answer,
  thin dominant stratum), exact to 1e-12. Coverage moves a point under any small
  change and cannot separate a refactor from a regression; these can.
  Regenerate deliberately with `python -m tools.gen_audit_ppi_golden`.

Every test that checks *bias* uses a one-directional perturbation. Symmetric
noise is unbiased on balanced binary outcomes, so a suite built from symmetric
flips passes against an estimator that has no idea what it is doing.

## The calibrator — from a store to a correction

[`Calibrator`](api.md#the-calibrator) is the join. It reads the store, assembles
the strata, runs the estimator, and hands back the one thing the acceptance gate
needs:

```python
from agentdescent.audit import Calibrator

cal = Calibrator(store)
r = cal.current(audited.verifier_version)

if not r.is_stale:
    p_true = p_hat - r.delta_hat        # subtract the bias
    var_true = var_p + r.se ** 2        # and carry its uncertainty
```

`delta_hat` is `E[f] - E[Y]`. Note which half is estimated: **`E[f]` is not**.
The tap saw every unit the run scored — audited or not — so the population mean
of `f` is a count, not a sample statistic. Only `E[Y]` is estimated, which is why
`delta_se` and `se` are the same number here, and would stop being so the day
someone computes `E[f]` from a subsample.

### Stale is an answer, not a failure

Every way this can fail returns a **stale** rectification rather than a number or
an exception, because the caller is a merge decision and a merge decision has to
be made:

| situation | what comes back |
|---|---|
| fewer than `min_labels` resolved calibration labels | stale — "a very wide interval" and "we do not know yet" are different claims |
| a verifier version never audited | stale |
| `mark_stale()` was called because the verifier changed | stale, carrying the previous numbers so a log can say what was withheld |
| the run converged — both scorers saturated, no variance anywhere | stale, and **not** a correction of zero: a converged run has no evidence about the verifier either way |

A stale rectifier means widen, not correct: the gate multiplies its variance by
`STALE_INFLATION` and commits less, rather than the same amount with more
confidence.

### Two things it will not do, because the alternative is quiet

**It never reads the improvement pool.** Labels used to *edit* the verifier were
chosen to make those units agree with it, so a bias estimated on them reads as
more honest than the truth. The test asserts this exactly rather than within a
tolerance: 400 flattering labels added to a store must not move `delta_hat` by a
single bit.

**It merges thin strata rather than dropping them.** A dropped stratum removes
its units from the population the estimate describes, so the answer silently
becomes "the bias among units we sampled enough of" — a different question, and
a flattering one when the thin stratum is where the verifier is worst. Both
halves move together: the records *and* their unlabelled moments, pooled with
Chan's parallel form. Adding the variances instead would be wrong by exactly the
between-group term, in the direction that makes an interval too narrow.

### What the store keeps, and what it does not

The estimator's dependence on the unlabelled half is exactly three numbers per
stratum — a count, a mean and a variance. So the store keeps a
[Welford](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
accumulator per `(verifier_version, stratum)` and **never stores an unlabelled
score**. At a 1% sampling rate that is the difference between three numbers and
a hundred thousand.

Welford rather than a running sum of squares, because the naive form subtracts
two large nearly-equal numbers and can return a small *negative* variance, which
propagates as a `nan` through the interval instead of failing where it happened.
Snapshots go into the same JSONL under a `kind` key, reconciled last-wins like
the records; a crash loses at most `FLUSH_EVERY` observations, which moves a
stratum mean by about 1e-4.
