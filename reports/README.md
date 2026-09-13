# Audit experiments

The measurements behind `agentdescent.audit` — every number quoted in
`docs/audit.md`, the module docstrings and the CHANGELOG.

**Only the core results are kept here.** The full set — seven Phase 0 reports,
two judge-rubric runs, the repair experiment, and every audit store they were
computed from — is preserved in this branch's history at commit **`c86e216`**,
where all 23 files existed together:

```bash
git show c86e216 --stat -- reports/                      # what is there
git show c86e216:reports/audit_phase0_mbpp_big_2026-09-13.md
git checkout c86e216 -- reports/                         # restore all of it
```

> If this branch is **squash-merged**, that history does not reach `main` and
> those files are gone with it. Merge with history, or re-run the scripts below.

Run against `deepseek-v4-flash` through an Anthropic-compatible endpoint:
~3,140 model calls, ~81 minutes.

## Does the verifier's bias exist and matter?

`Delta` is how much higher the judge scores answers than ground truth does.
Verdict is the Phase 0 kill test's own.

| workload | judging task | solver right | judge wrong | verdict |
|---|---|---|---|---|
| HotpotQA | is this paraphrase the same answer? | 27.7% | 17.5% | PROCEED |
| BBH | does content implying (B) count as "(B)"? | 55.1% | 32.7% | PROCEED |
| GSM8K | is this number that number? | 98.3% | **0.0%** | STOP |
| GSM-Hard | is this number that number? | 79.6% | 3.7% | STOP |
| MBPP | does this code do what that code does? | 75.5% | **18.6%** | PROCEED |

**The last two rows are the finding.** GSM-Hard and MBPP put the solver at near
enough the same difficulty and the judge's error rate differs five-fold. Judge
error comes from the *judging* task being ambiguous, not from the solving task
being hard — so a reward that is unit tests, exact match or numeric comparison
does not need this package at all.

On GSM-Hard the judge made **zero** errors in 54 units: both disagreements were
the *oracle* failing to parse a correct answer (`14053029 2/3` is exactly the
reference; it read `3`).

## Can ground truth repair the judge?

Run on MBPP, twice, and the pair is the result rather than either half:

| | small pool | larger pool |
|---|---|---|
| the loop's own gate | 9 units (resolution 1/9) | **30** |
| reward over ten rounds | 0.778, never moved | 0.800 → **0.967** |
| rules accepted | 7 | **2** |
| `sigma` on held-out | 0.4272 → 0.3125 | 0.4747 → **0.2549** |
| false negatives | 5.3% → 5.3% | 3.2% → **0.0%** |
| scorecard | refused | **nothing blocks** |

Fewer rules and more improvement: a gate that cannot see below 1/9 was selecting
on noise. The two rules it kept:

> - The candidate must be of the same syntactic form as the reference (e.g., a
>   function definition if the reference is a function definition).
> - Require that the candidate is a valid implementation satisfying the problem
>   specification, not merely a literal output.

## What is still here, and why

| file | why it is kept |
|---|---|
| [`verifier_diagnosis_f55dec40cec559f7.md`](verifier_diagnosis_f55dec40cec559f7.md) | the errors sorted by what it would take to fix them, the `sigma` floor, two hand-written rules measured one at a time, and a search over combinations. Every number in `diagnose.py`'s and `propose.py`'s docstrings comes from it, and regenerating it byte-identical is this branch's regression check |
| `audit_phase0_2026-09-09.jsonl` | `scripts/audit_diagnose.py`'s default input; two tests pin numbers against it |
| `audit_phase0_bbh_2026-09-10.jsonl` | the second workload `scripts/audit_judge_repair.py` runs on |

## Reproducing

```bash
python -m scripts.audit_phase0 --workload mbpp --tasks 250 --rounds 3 --workers 2
python -m scripts.audit_diagnose --records reports/audit_phase0_2026-09-09.jsonl
python -m scripts.audit_evolve_judge --records <a store> --workload mbpp
```

Every script takes `--dry-run`, which swaps an offline stand-in for the model and
exercises the whole path without a key.
