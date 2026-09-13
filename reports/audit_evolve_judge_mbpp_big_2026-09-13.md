# Evolving the judge's rubric -- mbpp (2026-09-13)

**The rubric clears the bar.**

| | |
|---|---|
| records | `reports/audit_phase0_mbpp_big_2026-09-13.jsonl` |
| P(new error mode) | 0.0000 (worth training at 0.25) |
| largest error mode | **15** x `no-function` (worth training at 5) |
| training units | 76 (48 right + 48 wrong, from 48/71) |
| held-out units | 87 of 87 calibration labels |
| dropped for task overlap | 0 |
| model calls | 462 |
| wall clock | 463.8s |

## The rubric it found

```
# Grading rules
- The candidate must be of the same syntactic form as the reference (e.g., a function definition if the reference is a function definition).
- Require that the candidate is a valid implementation satisfying the problem specification, not merely a literal output.
```

## On the held-out calibration labels

`sigma` is the target. `delta` is reported and never scored: a mean goes to zero when errors cancel.

| | before | after |
|---|---|---|
| `sigma` | 0.4747 | **0.2549** |
| `delta` | +0.2759 | +0.0690 |
| false negatives | 3.2% | 0.0% |

Fixed 21, broke 1, moved 22 units against a noise floor of **10** -- the flips a re-run of the *starting* rubric produced on the same outputs.

## Scorecard

# Verifier scorecard -- `f55dec40cec559f7+rubric`

| metric | value | previous | change | verdict |
|---|---|---|---|---|
| sigma | 0.2549 | 0.4747 | -0.2198 | better (can block) |
| false-negative rate | 0.0000 | 0.0323 | -0.0323 | better (can block) |
| delta_hat | 0.0690 | -- | -- | -- |
| disagreement | 0.0690 | 0.2989 | -0.2299 | better |
| ordering agreement | 1.0000 | 1.0000 | +0.0000 | unchanged |
| gain_factor | nan | -- | -- | -- |
| labels | 87.0000 | 87.0000 | +0.0000 | -- |

**sigma** -- The spread of `f - Y`, and the row to read first. It is what the acceptance gate's variance is built from, and unlike the bias it cannot be improved by making errors in both directions.

**false-negative rate** -- Correct answers the verifier marks down. Bounded at 5% by policy, not by measurement.

**delta_hat** -- Reported, not targeted. A mean error goes to zero when errors cancel: on the audit that motivated this card, a correct-looking rule cut it 74% while making the verifier worse. It is what the calibrator subtracts and what a drift monitor watches -- not what a change to the verifier should be judged by.

**disagreement** -- How often the two scorers differ at all, in either direction.

**ordering agreement** -- Artifact pairs the verifier orders the way ground truth does (1 of 1). The only property the acceptance gate actually uses, and the one no other row here carries -- a verifier can be badly biased and order perfectly, or nearly unbiased and pick the wrong winner. **Not blocking**: these artifacts are a lineage from one run rather than independent draws, and a handful of pairs is not something to gate on. Read a reversal as a reason to go and look.

**gain_factor** -- Approaching 1 means the verifier carries no usable signal about the truth, at which point the answer is a different verifier rather than a bigger audit.

## Nothing blocks

Every row that has a target moved the right way, or there was nothing to compare against.

---

# Ordering -- 2 artifacts, 87 units

- artifact pairs ordered the same way: **1/1** (100.0%)
- the verifier's favourite is also the truth's
- unit-level Kendall tau-b: 0.8649 (1550 concordant, 0 discordant)
    - every error runs the same way, so **no unit pair can be discordant** and tau says nothing about ordering here. A verifier that answered 1.0 to everything would score the same.

| artifact | n | mean f | mean Y |
|---|---|---|---|
| `bab6bec25105a4c1` | 44 | 0.773 | 0.705 |
| `cfb66be8073f1b14` | 43 | 0.070 | 0.000 |

These artifacts come from one run -- a lineage, not independent draws -- and 1 pairs is not a sample to put an interval around. Read a flip as a reason to look at it, not as a rate.

---

# Rescan -- 87 stored outputs, 2 artifacts

- agreement with the previous verifier: **74.7%**
- mean shift: -0.2069  (sd of the per-unit change 0.4610)
- residual sd against ground truth: 0.4747 -> 0.2549
- artifact pairs whose ordering reverses: **0/1** (0.0%)
