# Self-evolution bench — measure every rung, end to end

*Script:* [`bench/self_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/bench/self_evolution.py)

Each rung of the meta stack has its own module and its own tests; nothing
showed that they **compose**. This bench runs the whole chain on one
deterministic, offline arithmetic domain and reports every rung's measured
number — the evidence that the chain works end to end, no model key:

```bash
python -m bench.self_evolution          # human-readable per rung
python -m bench.self_evolution --json   # machine-readable report
```

It exits `0` only when every rung ran and produced a measurement.

## The stages

| Stage | What runs | What it reports |
| --- | --- | --- |
| **seed** | two `continuous_meta_run` sessions (the second seeds from the first) | verdicts, the adopted rule's parameter, the archive count |
| **population** | `meta_population_run` diversifies from the archive's rules | islands, what was promoted, the archive count |
| **exam** | `exam_archive` re-judges adoptions with an independent oracle | passed / flagged / demoted |
| **portfolio** | `layers_from_archive` + `portfolio_report` reallocate the next window | the layers and their new budgets |
| **harness** | two algorithm ports compete; the archive is the leaderboard | each port's best reward, the routed winner, why |
| **chain** | `chain_audit` traces the newest adoption's evidence rings | each ring's status and whether the adoption verifies |

## Determinism and trust

The domain is a tiny capability-capped arithmetic task set: every run is
offline (no model calls) and deterministic (fixed seeds), so the numbers are
stable across machines — the temporary directories differ, the measurements do
not. Small held-out warnings are silenced because the bench *wants* a tiny,
fast domain.

A rung that raises is recorded as a **failed stage** (`ok: false`, with the
error) and the bench continues, so a broken rung is the measurement rather than
a crash. `BenchReport.all_ok` is true only when every stage is `ok`.

## Extending it

The bench is written against the modules on `main`. New rungs can be added as
stages once they land — a joint multi-slot evolution stage and a `self_improve`
control-loop stage are the natural next two, so the bench keeps pace with the
stack it measures.
