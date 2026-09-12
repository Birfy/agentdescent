# Making the solver fail does not make the judge fail (2026-09-12)

Rung 5 -- evolving the judge's rubric against ground truth -- was blocked on a
saturated improvement pool: across HotpotQA and BBH, `P(the next label shows an
error mode nobody has seen)` had fallen to **0.0169**. The stated fix was a new
workload. Two were tried. Neither worked, and the reason is worth more than the
rung was.

## The assumption that failed

**A harder *solving* task was treated as a route to more judge errors.** It is
not. Judge error comes from the *judging* task being ambiguous, and those are
independent.

| workload | judging task | solver right | judge-vs-oracle disagreement |
|---|---|---|---|
| HotpotQA | is this paraphrase the same answer? | 27.7% | **17.5%** |
| BBH | does content implying (B) count as "(B)"? | 55.1% | **32.7%** |
| GSM8K | is this number that number? | **98.3%** | 0.0% |
| GSM-Hard | is this number that number? | 79.6% | **3.7%** |

Every cell is recomputed from the committed records by
`reports/judge_error_table.py`; the BBH row read 67.3% in a first draft, which
is that workload's judge *agreement* rate rather than its solver accuracy.

GSM8K failed for the obvious reason: the solver got 98.3% right, so there was
almost nothing to be wrong about. GSM-Hard fixed *that* -- the solver drops to
79.6% -- and the disagreement rate barely moved. The judge is simply right about
numeric equality, however hard the arithmetic that produced the number was.

## The two disagreements, in full

Fifty-four resolved pairs, two disagreements, and **both are the oracle's fault**:

| the answer | the reference | who was right |
|---|---|---|
| `14053029 2/3` | `14053029.666666666` | the **judge**. The answer is exact; the oracle reads the last number and got `3` |
| `98,826 hours, 37 minutes, and 35 seconds` | `5929597.583333333` (minutes) | the **judge**. A unit conversion; the oracle got `35` |

So on GSM-Hard the judge made **zero errors in 54 units**. `Delta = +0.037` with
a clustered 95% CI of `[0.0000, 0.0926]`, and the verdict is STOP.

That has a consequence for the diagnosis, not only for the headline: a mixed
number contains a `/`, so `14053029 2/3` was being classified
`wrong-number-with-working` -- an oracle bug counted as a judge slip, which
would have sent the improvement pool's budget after it. `equivalent-fraction` is
now its own mode. The unit conversion is left unclassified rather than guessed
at.

## What this means for rung 5

`P(new)` on GSM-Hard's improvement pool is **0.0455**, against a gate of 0.25.
The gate still refuses, and adding a fifth workload of the same kind would not
change that, because these workloads do not produce judge errors to learn from.

**The workload rung 5 needs is one where grading is hard, not one where solving
is hard.** Long-form answers, multi-part answers, answers whose correctness is a
judgement call -- code review, summarisation, an agent trajectory. That is also
the setting the audit package exists for: when `reward` is a fact about the
output, none of this is needed.

Two smaller findings from the same runs, recorded so they are not re-derived:

* **Constraining the solver is not a substitute for a harder problem.** Capping
  GSM8K's solver at 96 tokens does drop accuracy to 67%, but the errors are
  *truncations* -- the output stops mid-derivation with no answer in it, so the
  oracle reads whatever number the sentence was cut after. That is a broken
  solver, and every mode it generates is an artefact of the cap.

* **Phase 0 had no deadline.** The first GSM-Hard run deadlocked: zero CPU for
  15 minutes, three worker threads blocked on one lock with `abstime=0x0`, no
  open sockets, holding 58 resolved pairs it would never report. `evolve`'s own
  docstring names it -- `round_timeout=None` "waits forever ... but a single
  hung rollout then stalls the run, because the aggregator is a barrier".
  `--round-timeout` now defaults to 600s. The deadlock itself is unfixed.

## Reproduce

```bash
python -m scripts.audit_phase0 --workload gsm8k    --tasks 120 --rounds 4
python -m scripts.audit_phase0 --workload gsm_hard --tasks 120 --rounds 3 --workers 2
```
