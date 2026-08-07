# The runtime matrix — async behaviour

*Pending the post-restructuring rerun. This page will carry the
async-vs-sync comparison: `async_evolve` (barrier-free, completion-order merge
sweeps) against synchronous parallel `evolve`, at equal candidate budgets.*

## Headline

*TBD: async-vs-sync end-to-end, engine-window, and time-to-quality ratios,
with per-mode target-reach rates printed next to every TTQ figure.* The TTQ
ratio drops pairs where either side missed the target, so the reach rates are
the denominator context a reader needs before quoting any headline — the
summary JSON now carries `per_mode_target_reach` for exactly this purpose.

## Per-method TTQ

| Method | Serial TTQ (reached) | Sync TTQ (reached) | Async TTQ (reached) | Async/sync |
|---|---:|---:|---:|---:|
| *TBD* | | | | |

## What to look for after the rerun

- Async is equal-*candidate*, not equal-*work*: the barrier-free loop performs
  more rollouts and merge sweeps for the same candidate budget. The cost table
  must sit next to any TTQ claim.
- Stale-candidate rates and reflective-merge behaviour under completion-order
  sweeps are now measured per run (`observed_fusion_calls`, stale counters).
