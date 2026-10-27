# Statistical evidence — confidence, effect size, contamination

A point-estimate rule ("the mean gain is positive") adopts noise: with a small
held-out set a candidate can win by one lucky task. Two numbers make an RSI
adoption decision reproducible:

```python
from agentdescent import bootstrap_ci, is_significant, contamination_overlap

ci = bootstrap_ci(before_by_task, after_by_task, n=4000, alpha=0.05, seed=0)
if is_significant(ci, min_effect=0.01):
    ...

report = contamination_overlap(train_prompts, val_prompts, n=5)
if report.overlap is None:
    ...  # no n-grams could be screened: not evidence of cleanliness
elif report.overlap > 0.2 or report.exact_duplicates:
    ...  # hard stop, inspect report.shared / report.examples
```

- `paired_gain(before, after)` uses only task keys present on both sides, in a
  stable sorted order, and refuses non-finite scores.
- `bootstrap_ci` resamples **tasks** (paired), returns mean, central interval,
  the share of resamples above zero and the usable sample count. Same seed →
  same numbers, including across `PYTHONHASHSEED` values. A single pair reports
  `prob_positive=0.5` (no evidence), not certainty.
- `is_significant(ci, min_effect=...)` requires at least two pairs and a lower
  bound above the effect bar. `alpha` defaults to the interval's own alpha;
  passing a different one raises instead of silently returning `False`.
- `contamination_overlap(train_texts, val_texts, n=5)` reports the fraction of
  validation n-grams that occur in training, plus a few shared grams.
  Normalized exact duplicates count as leaked even when the text is too short
  for word n-grams, and CJK/short text falls back to character n-grams;
  training text is indexed both ways so a short validation item inside a longer
  prompt is still compared with like grams.
  `overlap=None` means there was nothing to screen — deliberately different
  from `0.0` ("nothing matched"). `exact_duplicates` counts verbatim matches.
  A low overlap is not proof of no contamination; paraphrase and embeddings are
  not covered.

These are pure functions: they do not call a model, read a run, or mutate an
archive, so they can be used by a guard, a report or an external audit.
