# Verifier diagnosis -- `f55dec40cec559f7`

`reports/audit_phase0_2026-09-09.jsonl`, 177 resolved pairs over 49 tasks

| kind | over | under | total | sigma if fixed |
|---|---|---|---|---|
| spec_gap | 12 | 0 | 12 | 0.3104 |
| ambiguous (floor) | 9 | 0 | 9 | 0.3309 |
| unclassified | 10 | 0 | 10 | 0.3243 |

31 disagreements in 177 pairs (17.5%); delta +0.1751, sigma 0.3812.

**Floor: sigma cannot go below 0.2203** against this oracle -- that is what remains once every non-ambiguous disagreement is fixed. Aiming lower means changing the definition of correct, not improving the verifier.

---

## Two rules a person reaches for first, measured one at a time

Reject an answer that echoes the question. Reject one far shorter than
the reference. Both are obviously correct in isolation, and each is
scored below against **every** labelled pair rather than against the
disagreements it targets.

| rule | fixed | broke | sigma | delta | false negatives | verdict |
|---|---|---|---|---|---|---|
| A -- reject an answer that echoes the question | 12 | 11 | 0.3812 -> **0.4104** | +0.1751 -> +0.0452 | 0.0% -> 22.4% | **does not help** |
| B -- reject an answer far shorter than the reference | 10 | 0 | 0.3812 -> **0.3243** | +0.1751 -> +0.1186 | 0.0% -> 0.0% | helps |
| A + B, as anyone would ship them | 19 | 11 | 0.3812 -> **0.3615** | +0.1751 -> +0.0056 | 0.0% -> 22.4% | helps |

**Rule A cuts the bias by 74% and makes the verifier worse.** It corrects twelve
over-credits and breaks eleven correct judgements, so the mean falls because the
errors now cancel; the spread, which is what the acceptance gate's variance is
built from, goes up. Every summary that leads with `delta` ships it.

**Bundled with a rule that works, it passes.** Rule B alone is a clean win --
ten corrections, nothing broken. Together the bundle's sigma improves, so the
bundle helps, and it still contains a rule that is measurably harmful and a
false-negative rate that went from nothing to 22%.

That is the argument for measuring each rule alone rather than the change as
shipped: a bundle launders whatever is in it.

# Rescan -- 177 stored outputs, 5 artifacts

- agreement with the previous verifier: **83.1%**
- mean shift: -0.1695  (sd of the per-unit change 0.3762)
- residual sd against ground truth: 0.3812 -> 0.3615
- artifact pairs whose ordering reverses: **2/10** (20.0%) **(above the alarm)**
    - `0bf0b7ead111` vs `26ecdd4b7262`: +0.031 -> -0.062
    - `0bf0b7ead111` vs `bab6bec25105`: +0.121 -> -0.009

The recorded history was scored by an instrument that no longer exists. Mark the affected version chain explicitly; do not let the old numbers stand unlabelled.

---

# Verifier scorecard -- `f55dec40cec559f7`

| metric | value | previous | change | verdict |
|---|---|---|---|---|
| sigma | 0.3812 | -- | -- | -- (can block) |
| false-negative rate | 0.0000 | -- | -- | -- (can block) |
| delta_hat | 0.1751 | -- | -- | -- |
| disagreement | 0.1751 | -- | -- | -- |
| gain_factor | nan | -- | -- | -- |
| labels | 177.0000 | -- | -- | -- |

**sigma** -- The spread of `f - Y`, and the row to read first. It is what the acceptance gate's variance is built from, and unlike the bias it cannot be improved by making errors in both directions.

**false-negative rate** -- Correct answers the verifier marks down. Bounded at 5% by policy, not by measurement.

**delta_hat** -- Reported, not targeted. A mean error goes to zero when errors cancel: on the audit that motivated this card, a correct-looking rule cut it 74% while making the verifier worse. It is what the calibrator subtracts and what a drift monitor watches -- not what a change to the verifier should be judged by.

**disagreement** -- How often the two scorers differ at all, in either direction.

**gain_factor** -- Approaching 1 means the verifier carries no usable signal about the truth, at which point the answer is a different verifier rather than a bigger audit.

## Nothing blocks

Every row that has a target moved the right way, or there was nothing to compare against.

> Nothing to compare against, so nothing blocks. This is a baseline, not a passing grade.
