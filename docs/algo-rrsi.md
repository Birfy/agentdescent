# RRSI — regularized harness mechanisms (microport increments)

This is an **offline mechanism microport** of selected pieces from Google
Research's *Regularized Recursive Self-Improvement of Agent Harnesses* (RRSI).
It is not a full port and does not claim to reproduce the paper's benchmark
results.

## Sources and pin

- Paper: [arXiv:2609.24972v3](https://arxiv.org/abs/2609.24972v3), revised
  2026-10-04.
- Released implementation: [`google-research/rrsi` at
  `be50316e1db05914068a973f322770ef08ed7ba1`](https://github.com/google-research/rrsi/tree/be50316e1db05914068a973f322770ef08ed7ba1).
- Adapted sources: `rrsi/selection.py` (with component novelty semantics from
  `rrsi/components.py`) and `rrsi/schedule.py`.

## What is preserved so far

### Round selection (`examples/rrsi/selection.py`)

A pure function layer for a round whose screened candidates were all measured
from the same incumbent. It applies the upstream historical-best floor
(`S_candidate >= S_best - delta`), both branches of the cost rule,
structural-component novelty, domain guards, and highest-score winner selection
(stable first candidate on ties). With no admissible candidate the caller
retains the incumbent. `update_best_score` keeps the historical best monotone.

The offline regression fixtures reject non-finite, out-of-range `[0, 1]` scores,
negative/non-finite token costs, non-finite historical bests and arithmetic
overflow. A candidate carrying a critic/smoke veto remains rejected even if a
measurement is also attached, and rejected or unevaluated candidates receive no
invented measurement in their decision record. These are defensive validation
choices around the upstream rule, not upstream parity claims.

### Edit-budget schedule (`examples/rrsi/schedule.py`)

The paper's cosine anneal on the L0 edit budget:

\[
b_t = \lceil b_{\min} + (b_{\max}-b_{\min})\cdot\tfrac12\bigl(1+\cos(\pi t / T)\bigr)\rceil,
\quad t = 0..T-1
\]

`edit_budget` / `budget_table` preserve the upstream float guard
(`round(v, 9)` before `ceil`), the `T <= 0 → b_max` degenerate case, and
clamping of `t` into `[0, T]`. Defensive validation rejects non-positive or
inverted `b_min`/`b_max` and non-integer inputs; that is an AgentDescent
boundary, not an upstream claim.

## Parity and intentional differences

| Behavior | Pinned upstream | This microport |
|---|---|---|
| Historical score floor | `S' >= S* - delta` | Preserved |
| Large-gain cost branch | `dS > delta`, require `dC <= beta0 + beta1*dS` | Preserved |
| Within-band branch | require `w_s*dS - w_c*dC + w_n*novelty > 0` | Preserved |
| Missing/zero token estimate | `dC = 0` if either side is falsy | Preserved; this can admit without telemetry |
| Invalid scores and costs | Not guarded by `selection.py` itself | Rejected before decision arithmetic |
| Critic-rejected candidate with an evaluation attached | Caller normally supplies no evaluation | Veto is authoritative and measurement omitted from decision |
| Round candidates | Evaluate from same base, choose admissible argmax | Caller contract documented; this module does not orchestrate evaluation |
| Cosine edit budget | `edit_budget` / `budget_table` | Preserved, plus defensive bound checks |
| Invalid `b_min`/`b_max` | Not guarded by `schedule.py` itself | Rejected before anneal arithmetic |

Neither mechanism is wired into `evolve()`, `aggregator_factory`, a proposal
policy, or a resumable run record yet. There is no RRSI CLI, harness-directory
domain, leakage critic, per-edit history, trial aggregation, pruning loop, or
evaluation/accounting adapter. AgentDescent's default evolution behavior is
unchanged. Real-model and benchmark measurements are **unmeasured** here;
upstream reported values are not AgentDescent results.

## Local verification

```bash
pytest -q tests/test_rrsi_selection.py tests/test_rrsi_schedule.py
```

Selection goldens cover the historical floor, both cost branches, domain veto,
ties, no winner, missing cost telemetry, rejected/missing candidates and invalid
numeric inputs. Schedule fixtures cover endpoints, monotonic anneal, the float
guard, clamping, degenerate horizons and invalid bounds. Completing the full
RRSI integration (issue #221) still needs pre-evaluation leakage screening,
proposal/history state, and the `evolve()`/CLI fixture.
