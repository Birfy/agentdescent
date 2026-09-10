# Sparse audit -- Phase 0 report (2026-09-10)

**Verdict: PROCEED.** the bias is real and larger than the gate's sampling noise (|delta|/sd = 5.55). A correction changes decisions.

## What was measured

| | |
|---|---|
| workload | BIG-Bench Hard across 6 subtasks, 78 questions |
| verifier `f` (cheap, biased) | LLM judge (deepseek-v4-flash, thinking=off) |
| oracle `Y` (ground truth) | normalized exact match against the reference |
| `verifier_version` | `f55dec40cec559f7` |
| loop | 6 rounds x 3 workers, held_out_frac=0.4, tournament=False |
| sampling | i.i.d., inclusion probability 1.0 |
| records | `reports/audit_phase0_bbh_2026-09-10.jsonl` |
| wall clock | 341s |
| model calls | 147 (15847+23198 tokens) |
| seed | 0 |

## The bias

`Delta = E[f - Y]`. Positive means the judge scores answers **higher** than exact match does -- the direction that lets a loop accept changes ground truth says improved nothing.

| pool | n | tasks | `Delta_hat` | 95% CI (unit) | 95% CI (clustered) | mean `f` | mean `Y` | disagree |
|---|---|---|---|---|---|---|---|---|
| all resolved | 49 | 49 | 0.3265 | [0.2041, 0.4694] | [0.2041, 0.4694] | 0.878 | 0.551 | 0.327 |
| calibration only | 36 | 36 | 0.3333 | [0.1944, 0.5000] | [0.1944, 0.5000] | 0.861 | 0.528 | 0.333 |

Units seen by the tap: 49; audited: 49; resolved and analysed: 49.

The estimator is the Hajek (inclusion-probability-weighted) mean with a percentile bootstrap. It is **not** the PPI estimator Phase 3 needs: with the labels this cheap there is no unlabelled mass to borrow strength from, and Phase 0 only has to decide whether the bias exists and matters.

**Read the clustered interval.** A run scores the same task again for every artifact version, so the audited units are not independent draws and the unit bootstrap is too narrow. Resampling tasks is the honest one; the gap between the two columns is the size of that dependence.

## Does it move a decision?

The acceptance gate reads a Beta posterior over 31 held-out tasks, whose own sd at the observed rate is **0.0589**.

- `|Delta_hat| / gate sd` = 5.55
- audit-limited (`SE(Delta)^2 > var_p`): **True** -- when true, buying more in-loop evaluation cannot improve the criterion and the budget belongs on oracle labels instead.

## Where the residual came from

The residual is mostly a property of the *shape* of the answer, so a workload drawn across shapes has to be reported across them -- otherwise the headline number is an average of two different phenomena.

| subtask | n | `Delta` | `sigma` | disagree |
|---|---|---|---|---|
| `salient_translation_error_detection` | 10 | +0.9000 | 0.3162 | 0.900 |
| `date_understanding` | 10 | +0.7000 | 0.4830 | 0.700 |
| `causal_judgement` | 6 | +0.0000 | 0.0000 | 0.000 |
| `object_counting` | 7 | +0.0000 | 0.0000 | 0.000 |
| `sports_understanding` | 8 | +0.0000 | 0.0000 | 0.000 |
| `word_sorting` | 8 | +0.0000 | 0.0000 | 0.000 |

### Generous, or not discriminating?

Exact match refuses `(B) Numerical Values` against a gold of `(B)`, and a judge accepting that is the benign story this experiment was built to measure. So: forgive every formatting difference the judge is *told* to forgive -- compare the option labels alone -- and ask whether it still says yes where that says no.

- labelled-answer units: **20**
- correct by the lenient label oracle: **10**
- the judge called right: **17**
- **rubber-stamped** (judge said right, lenient oracle says wrong): **7/20 = 35%**

A judge that is merely *generous* scores near zero here. This one has stopped discriminating on label-shaped answers -- a different failure from the one `Delta` describes, and indistinguishable from it in `Delta`.

## The fixed cheap subset (issue #179 §1.2)

`ThreeLayerVerifier._subset` draws `cheap_eval_tasks` held-out items **once** and reuses them, and the acceptance measurement includes them. Ranking can therefore overfit a fixed sample the gate then reads. If it does, artifacts score higher inside that subset than outside it.

Mean `score(inside cheap subset) - score(outside)` over 1 artifacts: **-0.3519**.

Ranking was **off** (`fusion_tournament=False`, the default), so nothing selected on the cheap layer at all. Whatever gap appears here is workload variation -- which is what makes it the baseline the `--tournament` arm has to be read against.

Resolution: 4 tasks inside against 27 outside, over 1 artifacts. That can see a large contamination and cannot resolve a small one; read a null here as "no evidence of", not "none".

## Final artifact

```
{}
```

held-out reward (as the loop measured it, i.e. through `f`): 0.806

## Reproduce

```bash
python3 scripts/audit_phase0.py --workload bbh --tasks 80 --rounds 6 --workers 3 --seed 0 --model deepseek-v4-flash
```
