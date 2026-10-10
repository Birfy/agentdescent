# RRSI — regularized harness selection (first increment)

This is an **offline mechanism microport** of the round-selection rule from
Google Research's *Regularized Recursive Self-Improvement of Agent Harnesses*
(RRSI). It is not a full port and does not claim to reproduce the paper's
benchmark results.

## Sources and pin

- Paper: [arXiv:2609.24972v3](https://arxiv.org/abs/2609.24972v3), revised
  2026-10-04.
- Released implementation: [`google-research/rrsi` at
  `be50316e1db05914068a973f322770ef08ed7ba1`](https://github.com/google-research/rrsi/tree/be50316e1db05914068a973f322770ef08ed7ba1).
- Adapted source: `rrsi/selection.py`; component novelty semantics from
  `rrsi/components.py`.

## What this increment preserves

`examples/rrsi/selection.py` is a pure function layer for a round whose screened
candidates were all measured from the same incumbent. It applies the upstream
historical-best floor (`S_candidate >= S_best - delta`), both branches of the
cost rule, structural-component novelty, domain guards, and highest-score
winner selection (stable first candidate on ties). With no admissible candidate
the caller retains the incumbent. `update_best_score` keeps the historical best
monotone.

The offline regression fixtures reject non-finite, out-of-range `[0, 1]` scores,
negative/non-finite token costs, non-finite historical bests and arithmetic
overflow. A candidate carrying a critic/smoke veto remains rejected even if a
measurement is also attached, and rejected or unevaluated candidates receive no
invented measurement in their decision record. These are defensive validation
choices around the upstream rule, not upstream parity claims.

## Parity and intentional differences

| Behavior | Pinned upstream | This increment |
|---|---|---|
| Historical score floor | `S' >= S* - delta` | Preserved |
| Large-gain cost branch | `dS > delta`, require `dC <= beta0 + beta1*dS` | Preserved |
| Within-band branch | require `w_s*dS - w_c*dC + w_n*novelty > 0` | Preserved |
| Missing/zero token estimate | `dC = 0` if either side is falsy | Preserved; this can admit without telemetry |
| Invalid scores and costs | Not guarded by `selection.py` itself | Rejected before decision arithmetic |
| Critic-rejected candidate with an evaluation attached | Caller normally supplies no evaluation | Veto is authoritative and measurement omitted from decision |
| Round candidates | Evaluate from same base, choose admissible argmax | Caller contract documented; this module does not orchestrate evaluation |

The mechanism is not wired into `evolve()`, `aggregator_factory`, a proposal
policy, or a resumable run record in this increment. There is no RRSI CLI,
harness-directory domain, leakage critic, edit-budget schedule, per-edit
history, trial aggregation, pruning loop, or evaluation/accounting adapter yet.
AgentDescent's default evolution behavior is unchanged. Real-model and benchmark
measurements are **unmeasured** here; upstream reported values are not
AgentDescent results.

## Local verification

Run the deterministic fixture with:

```bash
pytest -q tests/test_rrsi_selection.py
```

The golden cases cover the historical floor, both cost branches, domain veto,
ties, no winner, missing cost telemetry, rejected/missing candidates and invalid
numeric inputs. Completing the full RRSI integration requires a separate
increment for pre-evaluation leakage screening, proposal/history state, and the
`evolve()`/CLI fixture described in issue #221.
