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
    and that is not an improvement. Measured on a real 177-pair HotpotQA audit by
    `scripts/audit_diagnose.py`, which re-derives every number below offline:

    | rule | fixed | broke | `sigma` | `delta` | false negatives | |
    |---|---|---|---|---|---|---|
    | **A** answer echoes the question | 12 | 11 | 0.381 → **0.410** | −74% | 0% → **22.4%** | *does not help* |
    | **B** answer far shorter than the gold | 10 | 0 | 0.381 → **0.324** | −32% | 0% → 0% | helps |
    | **A + B**, as anyone would ship them | 19 | 11 | 0.381 → **0.362** | −97% | 0% → **22.4%** | helps |

    **Rule A cuts the bias by three quarters and makes the verifier worse.**
    Twelve corrections, eleven fresh mistakes: the mean falls because the errors
    now cancel, and the spread — which is what the acceptance gate's variance is
    built from — goes up.

    **And bundled with a rule that works, it passes.** B alone is a clean win, so
    the bundle's `sigma` improves and the bundle "helps" — while still containing
    A and still rejecting 22.4% of correct answers. A bundle launders whatever is
    in it, so measure one rule at a time.

    **`sigma` is the target throughout.** `delta` is what the calibrator already
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
disagreements it targets. Restricted to its targets, rule A above removes twelve
errors and breaks nothing — a clean win by every number a person reaches for. On
the whole set it also breaks eleven correct judgements and the residual goes up.

The rule was not a bad rule. It was a rule nobody had measured against the
answers it was not aimed at.

`FixReport.helps` reads `sigma_after < sigma_before` and nothing else, for the
reason at the top of this section.

Two rates come out of it and they are not the same number:

| | denominator | on rule A |
|---|---|---|
| `breakage_rate` | judgements that were **right** | 7.5% |
| `false_negative_after` | **answers** that were right | 22.4% |

They coincide only when the verifier errs in both directions equally, which is
never the interesting case. They were one field called `false_negative_rate`
until real data put 7.5% and 22.4% side by side.

## The scorecard — before a new verifier replaces the old one

A verifier is the instrument every other number in a run is measured with, so
changing it invalidates the run's history in a way nothing in the run can see.
[`verifier_scorecard`](api.md#the-verifier-scorecard) is the card that has to be
filled in at that moment:

```python
from agentdescent.audit import verifier_scorecard, rescan

card = verifier_scorecard(cal.current(new_version), fresh_labels,
                          previous=cal.current(old_version),
                          previous_records=old_labels,
                          rescan_report=rescan(old_labels, new_verifier, tasks),
                          cost=Cost(verifier_seconds=0.4, oracle_seconds=9.1))
print(card.to_markdown())      # card.ship is False if anything blocks
```

The plan's top row is `delta_hat`, "the only real target: it should fall". That
is the row rule A wins, so the card leads with `sigma` instead and reports
`delta_hat` below it with the reason attached.

| row | goal | blocks? |
|---|---|---|
| `sigma` | lower | **yes** — a rise means new errors in the opposite direction |
| false-negative rate | lower | **yes**, at a bound you set; it is a policy dial, not a measurement |
| `delta_hat` | — | never scored: the one metric a change can improve by breaking things |
| disagreement | lower | no |
| `gain_factor` | higher | no — approaching 1 means *change the verifier*, not audit harder |
| seconds per decision | lower | **yes** above `max_cost_ratio` of an oracle call |

`blockers` is the whole verdict. A weighted total would let a large fall in the
metric that lies buy a small rise in the one that does not, which is the exact
trade the card exists to refuse.

### The rescan, and what it is not

```python
report = rescan(old_records, new_verifier, tasks_by_id,
                pairs=[(base_sig, cand_sig), ...])
```

The plan describes replaying the Ledger — "the verifier is cheap and the
artifacts are all stored, so this sweep is nearly free". The Ledger stores
artifact *states*. The outputs those artifacts produced — the things a verifier
scores — were never kept, so re-deciding a past merge means re-running the agent
over both sides' held-out sets, which is a whole run's worth of rollouts and the
expensive half.

What *is* nearly free is re-scoring the outputs the **audit store** kept. Those
are a probability sample, so the numbers come with an `n` and are weighted by
`inclusion_prob`. Weaker than the plan's claim, and honest. It still answers the
question the row exists for: on the Phase 0 records, the A+B bundle disagrees
with the shipped judge on 17% of stored outputs and **reverses the ordering of
2 of 10 artifact pairs** — above the 10% alarm, meaning the run's recorded
history was scored by an instrument that would no longer say the same thing.

`sigma_shift` sits next to `mean_shift` for a reason: a verifier that moved half
its scores up by 0.5 and half down by 0.5 has a mean shift of exactly zero and
has rescored the run from end to end.

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

r.delta_hat      # E[f] - E[Y]: how generous the verifier is, on average
r.se             # how well that average is pinned down
r.resid_sd       # how *scattered* the error is around it  <- the one that matters
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

`resid_sd` is not in the plan, and it turns out to be the term the gate is mostly
made of. Why is the next section.

## Spending it — the gate

[`RectifiedAcceptance`](api.md#spending-the-correction) is the only place in the
package where the audit changes an outcome. It wraps whatever acceptance rule
the run already uses:

```python
from agentdescent.audit import RectifiedAcceptance
from agentdescent.policies import Policies

gate = RectifiedAcceptance(calibrator=cal, verifier_version=audited.verifier_version)
evolve(tasks, reward=audited, policies=Policies(acceptance=gate), ...)
```

`enabled=False` returns `inner.accept(ctx)` on the untouched context — not
"behaves the same as", *the same call* — which is what makes it safe to switch on
during a production run.

### The plan's formula is aimed at the wrong term

Phase 4 of the plan says:

```python
p_true = p_hat - delta_hat
var_true = var_p + r.se ** 2
```

Both halves are slightly off, and the Phase 0 audit says by how much. On 177
HotpotQA pairs, with the shipped gate reading 32 held-out tasks at 0.688:

| term | value | share of the binomial term (0.00671) |
|---|---|---|
| `resid_sd ** 2 / n` | 0.00454 | **68%** |
| `se(delta) ** 2` | 0.00082 | 12% |

The plan carries the 12% term and omits the 68% one.

**Why the correction itself mostly does not matter.** `delta_hat` is one number
subtracted from *both* sides of a comparison, so it cancels out of `cand - base`
exactly. So does its standard error. A gate asking "is this candidate better
than that one" is almost immune to a verifier that is uniformly generous — which
is good news, and is why this module reads less like a correction than expected.

What does not cancel is the verifier's *disagreement* with the truth on each
side's own held-out set. `mean(f) - delta` estimates `mean(Y)` with variance
`resid_sd ** 2 / n`, independently on each side, and the gate has been spending
that as evidence. Correcting a bias the gate never suffered from while ignoring
the noise it did is the shape of the mistake worth naming.

!!! warning "\"A positive `delta_hat` should mean fewer commits\" is half true"
    It is the plan's acceptance criterion, and it holds only above a rate of a
    half. Since `delta_hat` cancels out of the comparison, its only route to the
    verdict is the Beta spread `p(1-p)` — and subtracting it moves rates
    *towards* a half when they were above it and *away* when they were below.
    Measured over 600 random pairs on 32 held-out tasks:

    | measured rates | plain | `delta_hat` = 0.175 | plus `resid_sd` = 0.38 |
    |---|---|---|---|
    | 0.55 – 0.85 | 0.608 | 0.542 | 0.450 |
    | 0.10 – 0.35 | 0.602 | **0.685** | 0.257 |

    The criterion was reaching for `resid_sd`, which lowers the rate in both
    regimes because it is uncertainty rather than a shift.

`delta_hat` is still applied, for two smaller reasons that are real:

* **The variance scale.** A Beta posterior's spread is `p(1-p)`. At a measured
  0.90 that is 0.09; at the true 0.73 it is 0.20 — the gate is 2.2× overconfident
  about a difference in either direction.
* **The rates in the refusal.** `held-out regression 0.812 -> 0.781` is read by a
  person, and two numbers that are both 0.17 too high are two wrong numbers.

`se(delta)` is carried, **once** rather than twice, under a name that says what
it stands in for: `Adjustment.drift`, the allowance for `delta` not actually
being the same on both sides. It would not be, if a candidate shifted its outputs
into a stratum where the verifier is more generous — which is the failure this
whole package exists to catch. `se` is not an estimate of that drift; it is the
only number to hand of roughly the right size, and `drift_allowance=` takes a
better one.

### How the doubt is applied

By **discounting the counts**. A rate measured by a noisy proxy over `n` tasks is
worth some smaller number of oracle-scored tasks, and `discount_for` solves for
exactly that number:

```
p(1-p) / n_eff  =  p(1-p) / n + extra_var
kappa           =  p(1-p) / (p(1-p) + n * extra_var)
```

Scaling `(successes, failures)` by `kappa` leaves the rate untouched, so the
regression guard, `observed_delta`, and the artifact's own prior all see exactly
what they saw before; only the Beta test's confidence moves. And because the
change is in the *context* rather than in the rule, every acceptance policy gets
it, not just the shipped one.

On the Phase 0 numbers `kappa ≈ 0.60`: **32 tasks judged by that LLM judge carry
the information of 19 judged by exact match.** A candidate scoring 0.625 → 0.750
commits on the first reading and does not commit on the second.

!!! note "The gate's prior is not discounted"
    An earlier version solved for a *posterior* variance instead, which made the
    achievable widening depend on how many commits an artifact already had:
    past about forty, the prior alone was narrower than the target and the audit
    could not make the gate doubt its verifier at all — silently, since the
    arithmetic returned a number either way. The prior is separate evidence and
    still speaks; it is just not evidence the verifier produced.

Clipping is one-directional on purpose. Shifting a rate out of `[0, 1]` and
clipping it back shrinks the gap between the two sides, never widens it, so the
correction's failure mode is a candidate that does not commit.

### When there is nothing to apply

| situation | what the gate does |
|---|---|
| `enabled=False` | `inner.accept(ctx)`, same object in and out |
| rectification is stale | keeps the rates, spends `1 / STALE_INFLATION` of the evidence |
| no calibrator and no rectification | the same — "unmeasured" is not "unbiased" |
| `resid_sd` is missing | **stale**, because the missing term is the one the variance is mostly made of |
| `inflate_when_stale=1.0` | pass through, for a run migrating onto the audit |

A refusal says which of these applied, and — one extra Monte-Carlo draw, on
refusals only — whether the audit is what caused it:

```
refused by the audit -- audit: delta_hat +0.175; sigma_eps 0.381; evidence x0.61
```

The draw is seeded per candidate, so re-running the gate on the same context
returns the same number: the attribution is a fact about that decision, not a
coin flip near the threshold.

### The instrument has to hold still

A correction estimated for one verifier says nothing about the next, and nothing
in the arithmetic notices. [`VerifierWatch`](api.md#spending-the-correction)
withdraws the calibration when the verifier may have moved:

```python
watch = VerifierWatch(cal,
                      fingerprint=lambda: verifier_fingerprint(judge, extra=prompt),
                      artifact_ids=["judge_prompt"],
                      key_globs=["rubric.*"])
watch.check()                       # exact, and blind to an evolving prompt
watch.on_merge(artifact, diff)      # heuristic, and not blind to it
```

`check()` compares fingerprints, which is exact and blind to the case that
matters most: a verifier whose *prompt* the loop is evolving has the same module,
qualname and source. Fold the prompt into the fingerprint (`extra=`) and it stops
being blind.

**Nothing is watched by default**, which is the right default for a run whose
verifier is a fixed function and exactly the wrong one for a run that evolves its
own judge. A false positive costs one recompute; a false negative is the failure
the package exists to prevent. Name too much rather than too little.

## Can it order things at all?

Everything above measures how *far* the verifier is from the truth. The
acceptance gate does exactly one thing, and it is not that: it decides whether a
candidate is better than a baseline. A verifier can be badly wrong on every
number on this page and order every comparison correctly — add 0.2 to every
score and nothing the gate decides changes — and it can be close on all of them
and still pick the wrong winner.

[`rank_agreement`](api.md#ordering-agreement) is the row for that, and the
scorecard carries it.

On the Phase 0 audit, five artifacts from one run:

| artifact | n | mean `f` | mean `Y` |
|---|---|---|---|
| `bab6bec25105a4c1` | 49 | **0.714** | 0.408 |
| `26ecdd4b7262776b` | 32 | 0.625 | 0.375 |
| `0bf0b7ead111b97e` | 32 | 0.594 | **0.469** |
| `434f372098a00a9f` | 32 | 0.188 | 0.062 |
| `4aab01c9a0d25067` | 32 | 0.000 | 0.000 |

**Eight of ten pairs are ordered the same way; two are reversed.** The artifact
the verifier ranks first is third by ground truth, and one reversal is on an
apparent **12-point** improvement — the size of gap the gate commits on. `Δ` and
`resid_sd` both say "this judge is generous". Neither says "it picks the wrong
winner in one comparison out of five".

!!! danger "Unit-level Kendall τ is nearly uninformative here — and it is the number people ask for"
    Same data: 4753 concordant pairs, **zero** discordant, `τ_b = 0.681`. That
    is not evidence of good ordering; it is a restatement of the bias being
    one-directional. Two units are discordant only when the verifier prefers one
    and the truth prefers the other, and when every error runs the same way
    (`f > Y`, never `f < Y`) no such pair exists. **A verifier that answered 1.0
    to everything scores zero discordant pairs too.**

    `RankReport.one_directional` flags it, and the markdown says it in place.

Two more things the report is careful about:

**A reversal on a gap smaller than the gate's noise costs nothing** — the gate
refuses both candidates there. `report.above(gap)` gives the agreement among the
pairs the gate would actually have acted on. On this data the sub-0.05 reversal
drops out and the 12-point one does not.

**Artifacts from one run are a lineage, not independent draws.** Five of them
make ten pairs, and the report deliberately offers no interval: a binomial
interval on 2-of-10 spans 0.03 to 0.56 and would be wrong about the dependence
on top of that. A reversal is a reason to go and look, not a rate.

The scorecard row **does not block** for the same reason.

## Watching it over generations

One rectification says how biased the verifier is now. A sequence of them says
whether the loop is *finding* the verifier's blind spots — a `delta_hat` walking
steadily upward is a population drifting into whatever the proxy likes, and no
single measurement shows it.

```python
from agentdescent.audit import DriftMonitor

monitor = DriftMonitor()
for generation in generations:
    monitor.observe(calibrator.recompute(version), label=generation.name)
print(monitor.report.to_markdown())
```

!!! danger "Not a test per generation"
    At α = 0.05 that alarms once every twenty generations *when nothing is
    wrong*, by construction. Measured over two thousand runs of a hundred
    in-control generations each:

    | | alarms per 100 generations | clean runs that alarm |
    |---|---|---|
    | a two-sided test per generation | 4.95 | **99.3%** |
    | this chart (λ=0.2, L=3) | 0.27 | 16.2% |

    An operator who has seen five false alarms does not act on the sixth. `L`
    is not a per-generation significance level; it sets the average run length
    between false alarms.

Two things differ from the textbook chart, both because the inputs are estimates
rather than measurements:

**The limits are recursive.** The closed form assumes every point has the same
standard error; here each `delta_hat` arrives with its own, which grows and
shrinks with how many labels that generation bought. So
`Var(z) = λ²·se² + (1−λ)²·Var(z_prev)`, carried forward exactly — the band widens
after a noisy generation and narrows after a well-audited one.

**Overlapping label sets invalidate the chart, and are the default.**
`Calibrator` recomputes from the whole store, so consecutive rectifications share
most of their labels, are strongly positively correlated, and the true spread of
`z` is *wider* than the recursion says — the limits are too tight and the chart
alarms on a verifier that never moved. `DriftMonitor` reads `Rectification.covers`,
notices, and says so instead of charting silently. Feed it one rectification per
generation, computed on that generation's own labels, and the chart is valid.

`gain_factor` is watched differently, because it has no standard error and there
is nothing to put limits around: it is smoothed and compared against a threshold
near 1. Below it the verifier no longer predicts the truth well enough to borrow
from, and the signal says the thing the number implies — **replace the verifier**;
more labels only pay for what it stopped contributing.

## From another process

Truth may take days. The process that dispatched a record is long gone when a
wet-lab result or a human review comes back, so
[`agentdescent.audit.service`](api.md#the-audit-out-of-process) takes a **path**
and returns JSON. These are also the `audit_*` tools on the
[MCP server](plugins.md) (`agentdescent mcp`).

| verb | what it answers |
|---|---|
| `audit_status` | the rectifier in force, what it rests on, what is outstanding |
| `audit_pending` | the units waiting on an oracle, for a person or an experiment rig |
| `audit_resolve` | file one result — **refuses to overwrite** an existing one |
| `audit_recompute` | re-estimate after a batch is in |
| `audit_scorecard` | the card above, as rows and as prose |
| `audit_rescan` | re-score stored outputs with another verifier |
| `audit_drift` | the chart above, one point per verifier version |

Three details that are decisions rather than plumbing:

**A missing file is an error, not an empty store.** `AuditStore` treats an absent
path as a store about to be written, which is right for a run and wrong for a
question about one: reading a typo as "no records yet" is how a caller ends up
telling a user their verifier is unbiased.

**`version=None` means the busiest version, and the reply always names which one
it picked.** A store can hold several, and answering about the wrong one silently
is the failure this package exists to prevent, committed by its own reporting.

**`audit_rescan` resolves a `module:attribute` reference, which runs whatever it
imports.** It is bounded by the same allowlist the spec system uses — the
`agentdescent` package and nothing else — so widening it is a decision the person
operating the server makes, never one a calling model can make for them by
naming a module.
