# Candidate methods — parallel speedup

*Pending the post-restructuring rerun. This page will carry the paired
sync-vs-serial comparison: same candidate and proposal-call budgets, the only
change being `evolve(max_concurrency=workers)` against
`evolve(max_concurrency=1)`.*

## Headline

*TBD: paired end-to-end and engine-window speedups (min/median/max, n).* The
pre-restructuring run measured **1.36x E2E / 1.89x engine-window (n=33)** with
11/11 methods showing a median end-to-end win; those figures describe the
previous implementation and are retained in the
[runtime study](algo-candidate-methods.md) until the rerun replaces them.

## Per-method speedup

| Method | Sync/serial E2E | Sync/serial engine | Paired runs |
|---|---:|---:|---:|
| *TBD* | | | |

## What to look for after the rerun

- Worker-sized merge batches mean parallel workers' proposals now actually
  meet in one merge: union merges on library/memory artifacts, reflective
  merges on contested text. Whether merged unions change the speedup picture
  is an open measurement.
- Self-verify methods (Voyager, SkillWeaver) pay one extra rollout per
  candidate in every mode; the comparison stays equal-budget.
