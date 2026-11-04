Fixed
When the seed prompt already scores the target on held-out and the run does
not improve on it, `diagnose` now reports `seed-already-solved` (add cases the
seed gets wrong, use a stricter score, or raise `target_reward`) instead of
`solved` / "review and apply the artifact", and foreground `evolve` prints the
same note after `held-out reward: 1.000 -> 1.000` (and adds `note` to `--json`).
`docs/algorithms.md` replaces an empty code block with a tip to include cases
the seed fails (#732).
