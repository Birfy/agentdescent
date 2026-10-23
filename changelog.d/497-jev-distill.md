Added
`JEVModel`, a dependency-free distilled Judge–Evaluator–Verifier student: a
linear logistic scorer over caller-supplied `(artifact, task)` features that
learns oracle labels online (`observe`), in passes (`fit`), predicts in `[0, 1]`
where a verifier is expected, and checkpoints as JSON. Intended to distil oracle labels from a disagreement-selected task loop, not to
replace the oracle; features are standardised online and the JSON checkpoint
carries the scaling state.
