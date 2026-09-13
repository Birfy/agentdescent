# Audit experiments

Every number quoted in `docs/audit.md`, the `agentdescent.audit` docstrings and
the CHANGELOG comes from one of these. Each `.md` is the written report; the
`.jsonl` beside it is the audit store it was computed from, so any claim can be
recomputed without re-running the model.

Run against `deepseek-v4-flash` through an Anthropic-compatible endpoint.
~3,140 model calls, ~81 minutes in total.

## Does the verifier's bias exist and matter? (Phase 0)

`Delta` is how much higher the judge scores answers than ground truth does.
Verdict is the script's own kill test.

| report | workload | judging task | solver right | judge wrong | verdict |
|---|---|---|---|---|---|
| [2026-09-09](audit_phase0_2026-09-09.md) | HotpotQA | is this paraphrase the same answer? | 27.7% | 17.5% | PROCEED |
| [2026-09-09 (tournament)](audit_phase0_2026-09-09_tournament.md) | HotpotQA | the `--tournament` arm of the same run | — | — | — |
| [2026-09-10](audit_phase0_bbh_2026-09-10.md) | BBH | does content implying (B) count as "(B)"? | 55.1% | 32.7% | PROCEED |
| [2026-09-12](audit_phase0_2026-09-12_gsm8k.md) | GSM8K | is this number that number? | 98.3% | **0.0%** | STOP |
| [2026-09-12](audit_phase0_2026-09-12_gsm_hard.md) | GSM-Hard | is this number that number? | 79.6% | 3.7% | STOP |
| [2026-09-12](audit_phase0_2026-09-12_mbpp.md) | MBPP | does this code do what that code does? | 75.5% | 18.6% | PROCEED |
| [2026-09-13](audit_phase0_mbpp_big_2026-09-13.md) | MBPP (250 tasks) | as above, larger sample | 38.3% | 30.1% | PROCEED |

[`judge_error_table.py`](judge_error_table.py) recomputes the two right-hand
columns offline from the stores.

## What the verifier gets wrong, and whether a fix helps

| report | what it is |
|---|---|
| [Verifier diagnosis](verifier_diagnosis_f55dec40cec559f7.md) | the HotpotQA errors sorted by what it would take to fix them, the `sigma` floor, two hand-written rules measured one at a time, and a search over rule combinations |
| [Judge repair](audit_judge_repair_2026-09-12.md) | one hand-written clause against a control arm, on both workloads. The control — the **unchanged** prompt, re-run — is the noise floor every other row is read against |

## Can ground truth evolve the judge?

| report | pool | `sigma` | rules | outcome |
|---|---|---|---|---|
| [small pool](audit_evolve_judge_mbpp_2026-09-13.md) | 50 labels, loop gate 9 units | 0.4272 → 0.3125 | 7 | refused by the scorecard |
| [larger pool](audit_evolve_judge_mbpp_big_2026-09-13.md) | 119 labels, loop gate 30 units | 0.4747 → **0.2549** | **2** | **nothing blocks** |

Fewer rules and more improvement, because the first run's gate could not see
below 1/9 and was selecting on noise.

## The finding the workloads were collected to test

[**Making the solver fail does not make the judge
fail**](judge_error_needs_a_hard_judging_task_2026-09-12.md) — GSM-Hard and MBPP
put the solver at 79.6% and 75.5%, near enough the same difficulty, and the
judge's error rate differs five-fold. Judge error comes from the *judging* task
being ambiguous, not from the solving task being hard.

## Reproducing

```bash
python -m scripts.audit_phase0 --workload mbpp --tasks 250 --rounds 3 --workers 2
python -m scripts.audit_diagnose --records reports/audit_phase0_2026-09-09.jsonl
python -m scripts.audit_evolve_judge --records <a store> --workload mbpp
```

Every script takes `--dry-run`, which swaps an offline stand-in for the model
and exercises the whole path without a key.
