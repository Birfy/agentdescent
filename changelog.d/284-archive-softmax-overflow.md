Fixed
`selection.Archive` no longer crashes at small temperatures (#284).
`performance` / `novelty` computed `exp(score / temperature)` raw, so below
about `temperature = 1/709` `select()` raised `OverflowError`. Just above
that, a few hundred near-top candidates summed to `inf` and
`random.choices` refused the total. The raw weights are still used,
bit-for-bit, whenever they are representable, so seeded draws do not move.
When they overflow, or their total is not a positive finite number, the
max-shifted softmax (the same distribution) is used instead. A `NaN`,
infinite or non-numeric `temperature` is now rejected with `ValueError` at
construction rather than failing at the first `select()`.
