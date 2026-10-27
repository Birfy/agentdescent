Added
`bootstrap_ci` / `is_significant` / `paired_gain` and `contamination_overlap`:
seeded paired-task bootstrap confidence intervals with an effect-size test, and
a contamination report between train and validation text. Paired keys are
sorted so the same seed is reproducible across processes, non-finite scores are
refused, normalized exact duplicates and CJK/short text are screened, and
"nothing to screen" is `None` rather than a clean-looking `0.0`. Pure
standard-library functions for adoption gates, reports and external audits.
