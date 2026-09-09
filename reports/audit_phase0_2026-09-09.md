# Sparse audit -- Phase 0 report (2026-09-09)

**Verdict: PROCEED.** the bias is real and larger than the gate's sampling noise (|delta|/sd = 2.70). A correction changes decisions.

## What was measured

| | |
|---|---|
| workload | HotpotQA validation, 80 questions |
| verifier `f` (cheap, biased) | LLM judge (deepseek-v4-flash, thinking=off) |
| oracle `Y` (ground truth) | normalized exact match against the reference |
| `verifier_version` | `f55dec40cec559f7` |
| loop | 6 rounds x 3 workers, held_out_frac=0.4, tournament=False |
| sampling | i.i.d., inclusion probability 1.0 |
| wall clock | 1385s |
| model calls | 547 (31653+284636 tokens) |
| seed | 0 |

## The bias

`Delta = E[f - Y]`. Positive means the judge scores answers **higher** than exact match does -- the direction that lets a loop accept changes ground truth says improved nothing.

| pool | n | `Delta_hat` | 95% CI | SE | mean `f` | mean `Y` | disagree |
|---|---|---|---|---|---|---|---|
| all resolved | 178 | 0.2360 | [0.1742, 0.2978] | 0.0317 | 0.573 | 0.337 | 0.236 |
| calibration only | 130 | 0.2077 | [0.1385, 0.2769] | 0.0364 | 0.592 | 0.385 | 0.208 |

Units seen by the tap: 178; audited: 178.

The estimator is the Hajek (inclusion-probability-weighted) mean with a percentile bootstrap interval. It is **not** the PPI estimator Phase 3 needs: with the labels this cheap there is no unlabelled mass to borrow strength from, and Phase 0 only has to decide whether the bias exists and matters.

## Does it move a decision?

The acceptance gate reads a Beta posterior over 32 held-out tasks, whose own sd at the observed rate is **0.0874**.

- `|Delta_hat| / gate sd` = 2.70
- audit-limited (`SE(Delta)^2 > var_p`): **False** -- when true, buying more in-loop evaluation cannot improve the criterion and the budget belongs on oracle labels instead.

## The fixed cheap subset (issue #179 §1.2)

`ThreeLayerVerifier._subset` draws `cheap_eval_tasks` held-out items **once** and reuses them, and the acceptance measurement includes them. Ranking can therefore overfit a fixed sample the gate then reads. If it does, artifacts score higher inside that subset than outside it.

Mean `score(inside cheap subset) - score(outside)` over 5 artifacts: **-0.2929**.

With `fusion_tournament=False` (the default) nothing ranks on the cheap layer at all, so a gap here is workload variation rather than selection pressure. Re-run with `--tournament` to see what ranking adds.

## Final artifact

```
{
  "r5c86b1c946": "Output the exact proper name of the requested entity rather than a generic term."
}
```

held-out reward (as the loop measured it, i.e. through `f`): 0.594

## Reproduce

```bash
python3 scripts/audit_phase0.py --tasks 80 --rounds 6 --workers 3 --seed 0 --model deepseek-v4-flash
```
