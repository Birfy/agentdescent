# RRSI — regularized harness selection (bounded offline increments)

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

## What these increments preserve

`examples/rrsi/selection.py` is a pure function layer for a round whose screened
candidates were all measured from the same incumbent. It applies the upstream
historical-best floor (`S_candidate >= S_best - delta`), both branches of the
cost rule, structural-component novelty, domain guards, and highest-score
winner selection (stable first candidate on ties). With no admissible candidate
the caller retains the incumbent. `update_best_score` keeps the historical best
monotone.

`examples/rrsi/round.py` adds an offline **single-round runner** around that
selector. A caller supplies proposal, screening and evaluation callbacks. The
proposal callback receives one read-only snapshot, version and content digest
of the incumbent; variants naming another version or digest are rejected. All
candidate files and components are snapshotted before any screening or
evaluation callback runs. The screening callback runs before evaluation, and
screened-in survivors are evaluated and selected together. The callback is only
a pre-evaluation seam: this module does not implement Google's leakage checks or
a model critic. Proposal, screen, evaluation, measurement and guard errors have
explicit outcomes, without fabricated measurements.

Each round returns a JSON-serializable `RoundRecord` containing the base digest,
input selection parameters, candidate artifact digests, outcomes, measurements,
decisions and best-score update. This is a record for one round, not a resume
store or full run history. A typical caller retains the incumbent when
`RoundResult.winner` is `None`.

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
| Round candidates | Evaluate from same base, choose admissible argmax | `run_round` enforces one read-only base snapshot/version, screens before evaluation, and selects among all survivors |

The mechanism is not wired into `evolve()`, `aggregator_factory`, or a proposal
policy. There is no RRSI CLI, harness-directory domain, implemented leakage
critic, edit-budget schedule, measured per-edit history, multi-round resume,
trial aggregation, pruning loop, or production evaluation/accounting adapter
yet. The caller owns each callback and any durable storage. AgentDescent's
default evolution behavior is unchanged. Fixtures use deterministic local
callbacks; real-model and benchmark measurements are **unmeasured** here, and
upstream reported values are not AgentDescent results.

## Local verification

Run the deterministic fixtures with:

```bash
pytest -q tests/test_rrsi_selection.py tests/test_rrsi_round.py
```

The golden cases cover the historical floor, both cost branches, domain veto,
ties, no winner, missing cost telemetry, rejected/missing candidates and invalid
numeric inputs. Round fixtures also prove screening precedes evaluation, all
candidates share a read-only base snapshot, callback errors are recorded, score
and cost remain paired with the selected candidate, and records round-trip via
JSON. Completing the full RRSI integration requires separate increments for an
actual leakage critic, proposal/history state, resume semantics, and the
`evolve()`/CLI fixture described in issue #221.
