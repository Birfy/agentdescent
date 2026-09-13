# Sparse audit -- Phase 0 report (2026-09-09)

> **This run's store is not kept.** Nothing recomputes a claim from it: the
> numbers below are self-contained, and no script, test or document reads the
> file. Every other Phase 0 store *is* kept, because some committed claim can
> only be recomputed from it.

**Verdict: PROCEED.** the bias is real and larger than the gate's sampling noise (|delta|/sd = 1.97). A correction changes decisions.

## What was measured

| | |
|---|---|
| workload | HotpotQA validation, 80 questions |
| verifier `f` (cheap, biased) | LLM judge (deepseek-v4-flash, thinking=off) |
| oracle `Y` (ground truth) | normalized exact match against the reference |
| `verifier_version` | `f55dec40cec559f7` |
| loop | 6 rounds x 3 workers, held_out_frac=0.4, tournament=True |
| sampling | i.i.d., inclusion probability 1.0 |
| records | `reports/audit_phase0_2026-09-09_tournament.jsonl` |
| wall clock | 1857s |
| model calls | 537 (28236+256119 tokens) |
| seed | 0 |

## The bias

`Delta = E[f - Y]`. Positive means the judge scores answers **higher** than exact match does -- the direction that lets a loop accept changes ground truth says improved nothing.

| pool | n | tasks | `Delta_hat` | 95% CI (unit) | 95% CI (clustered) | mean `f` | mean `Y` | disagree |
|---|---|---|---|---|---|---|---|---|
| all resolved | 178 | 50 | 0.1742 | [0.1236, 0.2303] | [0.0934, 0.2697] | 0.534 | 0.360 | 0.174 |
| calibration only | 126 | 41 | 0.1905 | [0.1270, 0.2619] | [0.0873, 0.3111] | 0.595 | 0.405 | 0.190 |

Units seen by the tap: 178; audited: 178.

The estimator is the Hajek (inclusion-probability-weighted) mean with a percentile bootstrap. It is **not** the PPI estimator Phase 3 needs: with the labels this cheap there is no unlabelled mass to borrow strength from, and Phase 0 only has to decide whether the bias exists and matters.

**Read the clustered interval.** A run scores the same task again for every artifact version, so the audited units are not independent draws and the unit bootstrap is too narrow. Resampling tasks is the honest one; the gap between the two columns is the size of that dependence.

## Does it move a decision?

The acceptance gate reads a Beta posterior over 32 held-out tasks, whose own sd at the observed rate is **0.0882**.

- `|Delta_hat| / gate sd` = 1.97
- audit-limited (`SE(Delta)^2 > var_p`): **False** -- when true, buying more in-loop evaluation cannot improve the criterion and the budget belongs on oracle labels instead.

## The fixed cheap subset (issue #179 §1.2)

`ThreeLayerVerifier._subset` draws `cheap_eval_tasks` held-out items **once** and reuses them, and the acceptance measurement includes them. Ranking can therefore overfit a fixed sample the gate then reads. If it does, artifacts score higher inside that subset than outside it.

Mean `score(inside cheap subset) - score(outside)` over 5 artifacts: **-0.1786**.

Ranking **was** on for this run, so the cheap layer really did choose which candidate went forward. A *positive* gap would be the contamination: candidates picked for scoring well on those few tasks, then measured again on a set that contains them. A negative one says the subset happens to hold harder tasks and selection did not overcome that.

Resolution: 4 tasks inside against 28 outside, over 5 artifacts. That can see a large contamination and cannot resolve a small one; read a null here as "no evidence of", not "none".

## Final artifact

```
{}
```

held-out reward (as the loop measured it, i.e. through `f`): 0.656

## Reproduce

```bash
python3 scripts/audit_phase0.py --tasks 80 --rounds 6 --workers 3 --seed 0 --model deepseek-v4-flash
```
