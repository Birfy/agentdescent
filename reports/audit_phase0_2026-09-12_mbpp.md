# Sparse audit -- Phase 0 report (2026-09-12)

**Verdict: PROCEED.** the bias is real and larger than the gate's sampling noise (|delta|/sd = 1.68). A correction changes decisions.

## What was measured

| | |
|---|---|
| workload | MBPP, scored by executing each task's asserts, 120 questions |
| verifier `f` (cheap, biased) | LLM judge (deepseek-v4-flash, thinking=off) |
| oracle `Y` (ground truth) | the task's own asserts, executed against the candidate |
| `verifier_version` | `f55dec40cec559f7` |
| loop | 3 rounds x 2 workers, held_out_frac=0.4, tournament=False |
| sampling | i.i.d., inclusion probability 1.0 |
| records | `reports/audit_phase0_2026-09-12_mbpp.jsonl` |
| wall clock | 279s |
| model calls | 307 (34086+79390 tokens) |
| seed | 0 |

## The bias

`Delta = E[f - Y]`. Positive means the judge scores answers **higher** than exact match does -- the direction that lets a loop accept changes ground truth says improved nothing.

| pool | n | tasks | `Delta_hat` | 95% CI (unit) | 95% CI (clustered) | mean `f` | mean `Y` | disagree |
|---|---|---|---|---|---|---|---|---|
| all resolved | 102 | 54 | 0.0882 | [0.0098, 0.1667] | [0.0096, 0.1748] | 0.843 | 0.755 | 0.186 |
| calibration only | 52 | 27 | 0.1154 | [0.0000, 0.2308] | [0.0370, 0.2222] | 0.846 | 0.731 | 0.192 |

Units seen by the tap: 102; audited: 102; resolved and analysed: 102.

The estimator is the Hajek (inclusion-probability-weighted) mean with a percentile bootstrap. It is **not** the PPI estimator Phase 3 needs: with the labels this cheap there is no unlabelled mass to borrow strength from, and Phase 0 only has to decide whether the bias exists and matters.

**Read the clustered interval.** A run scores the same task again for every artifact version, so the audited units are not independent draws and the unit bootstrap is too narrow. Resampling tasks is the honest one; the gap between the two columns is the size of that dependence.

## Does it move a decision?

The acceptance gate reads a Beta posterior over 48 held-out tasks, whose own sd at the observed rate is **0.0525**.

- `|Delta_hat| / gate sd` = 1.68
- audit-limited (`SE(Delta)^2 > var_p`): **False** -- when true, buying more in-loop evaluation cannot improve the criterion and the budget belongs on oracle labels instead.

## The fixed cheap subset (issue #179 §1.2)

`ThreeLayerVerifier._subset` draws `cheap_eval_tasks` held-out items **once** and reuses them, and the acceptance measurement includes them. Ranking can therefore overfit a fixed sample the gate then reads. If it does, artifacts score higher inside that subset than outside it.

Mean `score(inside cheap subset) - score(outside)` over 2 artifacts: **-0.1023**.

Ranking was **off** (`fusion_tournament=False`, the default), so nothing selected on the cheap layer at all. Whatever gap appears here is workload variation -- which is what makes it the baseline the `--tournament` arm has to be read against.

Resolution: 4 tasks inside against 44 outside, over 2 artifacts. That can see a large contamination and cannot resolve a small one; read a null here as "no evidence of", not "none".

## Final artifact

```
{
  "rd8ab657ce1": "Output the required code implementation, not the problem statement."
}
```

held-out reward (as the loop measured it, i.e. through `f`): 0.875

## Reproduce

```bash
python3 scripts/audit_phase0.py --workload mbpp --tasks 120 --rounds 3 --workers 2 --seed 0 --model deepseek-v4-flash
```
