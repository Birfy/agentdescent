# Evolving the judge's rubric -- mbpp (2026-09-13)

**Not shown to help.** 5.3% of correct answers are marked down, above the 5% policy bound; 100% of compared artifact pairs reverse order under the new verifier (alarm at 10%); the recorded history was scored by an instrument that no longer exists and the affected version chain has to be marked

| | |
|---|---|
| records | `reports/audit_phase0_2026-09-12_mbpp.jsonl` |
| P(new error mode) | 0.0200 (worth training at 0.25) |
| largest error mode | **6** x `looks-like-the-reference` (worth training at 5) |
| training units | 22 (11 right + 11 wrong, from 39/11) |
| held-out units | 52 of 52 calibration labels |
| dropped for task overlap | 0 |
| model calls | 344 |
| wall clock | 582.6s |

## The rubric it found

```
# Grading rules
- Reject candidate answers that are merely the expected output when the task requires a function definition or code.
- Reject any candidate that does not contain a function definition with the name required by the problem.
- Verify that the candidate's solution produces the same output as the reference for all possible inputs, not just the provided example.
- Reject candidate answers that are not valid Python code.
- Grade the candidate based on whether it meets the problem's requirements, not on whether it exactly matches the provided reference.
- Reject any candidate answer that is empty or contains only whitespace.
- Ignore parameter names when comparing candidate functions to the reference.
```

## On the held-out calibration labels

`sigma` is the target. `delta` is reported and never scored: a mean goes to zero when errors cancel.

| | before | after |
|---|---|---|
| `sigma` | 0.4272 | **0.3125** |
| `delta` | +0.1154 | +0.0192 |
| false negatives | 5.3% | 5.3% |

Fixed 5, broke 0, moved 5 units against a noise floor of **2** -- the flips a re-run of the *starting* rubric produced on the same outputs.

## Scorecard

# Verifier scorecard -- `f55dec40cec559f7+rubric`

| metric | value | previous | change | verdict |
|---|---|---|---|---|
| sigma | 0.3125 | 0.4272 | -0.1147 | better (can block) |
| false-negative rate | 0.0526 | 0.0526 | +0.0000 | unchanged **(blocking)** |
| delta_hat | 0.0192 | -- | -- | -- |
| disagreement | 0.0962 | 0.1923 | -0.0962 | better |
| ordering agreement | 1.0000 | 0.0000 | +1.0000 | better |
| gain_factor | nan | -- | -- | -- |
| labels | 52.0000 | 52.0000 | +0.0000 | -- |

**sigma** -- The spread of `f - Y`, and the row to read first. It is what the acceptance gate's variance is built from, and unlike the bias it cannot be improved by making errors in both directions.

**false-negative rate** -- Correct answers the verifier marks down. Bounded at 5% by policy, not by measurement.

**delta_hat** -- Reported, not targeted. A mean error goes to zero when errors cancel: on the audit that motivated this card, a correct-looking rule cut it 74% while making the verifier worse. It is what the calibrator subtracts and what a drift monitor watches -- not what a change to the verifier should be judged by.

**disagreement** -- How often the two scorers differ at all, in either direction.

**ordering agreement** -- Artifact pairs the verifier orders the way ground truth does (1 of 1). The only property the acceptance gate actually uses, and the one no other row here carries -- a verifier can be badly biased and order perfectly, or nearly unbiased and pick the wrong winner. **Not blocking**: these artifacts are a lineage from one run rather than independent draws, and a handful of pairs is not something to gate on. Read a reversal as a reason to go and look.

**gain_factor** -- Approaching 1 means the verifier carries no usable signal about the truth, at which point the answer is a different verifier rather than a bigger audit.

## Do not ship

- 5.3% of correct answers are marked down, above the 5% policy bound
- 100% of compared artifact pairs reverse order under the new verifier (alarm at 10%); the recorded history was scored by an instrument that no longer exists and the affected version chain has to be marked

---

# Ordering -- 2 artifacts, 52 units

- artifact pairs ordered the same way: **1/1** (100.0%)
- the verifier's favourite is also the truth's
- unit-level Kendall tau-b: 0.7509 (396 concordant, 6 discordant)

| artifact | n | mean f | mean Y |
|---|---|---|---|
| `3e485934a8e78781` | 25 | 0.800 | 0.840 |
| `bab6bec25105a4c1` | 27 | 0.704 | 0.630 |

These artifacts come from one run -- a lineage, not independent draws -- and 1 pairs is not a sample to put an interval around. Read a flip as a reason to look at it, not as a rate.

---

# Rescan -- 52 stored outputs, 2 artifacts

- agreement with the previous verifier: **90.4%**
- mean shift: -0.0962  (sd of the per-unit change 0.2977)
- residual sd against ground truth: 0.4272 -> 0.3125
- artifact pairs whose ordering reverses: **1/1** (100.0%) **(above the alarm)**
    - `3e485934a8e7` vs `bab6bec25105`: +0.012 -> -0.096

The recorded history was scored by an instrument that no longer exists. Mark the affected version chain explicitly; do not let the old numbers stand unlabelled.
