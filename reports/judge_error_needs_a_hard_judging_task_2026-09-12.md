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
| MBPP | does this code do what that code does? | 75.5% | **18.6%** |

**The last two rows are the argument.** GSM-Hard and MBPP put the solver at
79.6% and 75.5% -- near enough the same difficulty -- and the judge's error rate
differs by **five times**. The solver is not the variable. Whether grading is a
judgement call is.

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

## The workload that works

MBPP: the oracle runs the task's own asserts, the judge is shown the reference
solution and has to decide whether a differently-written candidate does the same
thing. It cannot execute anything, two correct solutions look nothing alike, and
a subtly wrong one looks exactly like a right one.

**Verdict PROCEED** -- the first since BBH. `Delta = +0.0882`, clustered 95% CI
`[0.0096, 0.1748]`, which excludes zero; `|Delta| / gate sd = 1.68`, so a
correction changes decisions. 307 model calls, 279s.

And the errors are the ones the setup predicts. Of nineteen disagreements,
**twelve are `looks-like-the-reference`**: the judge said yes to code shaped like
the answer that does not pass the asserts. Six are `does-not-parse` and one is
`no-function` -- a judge saying yes to something that is not a program, which is
a different bug and is kept in a different bucket for that reason.

## What this means for rung 5

MBPP supplies the judge errors. It does **not** clear the gate: `P(new)` on its
improvement pool is **0.02**, against a threshold of 0.25.

That refusal is now worth doubting, and the doubt is about the gate rather than
the pool. Good--Turing estimates the chance the next label shows a *species* not
yet seen, and `code_error_mode` has six species. After nineteen disagreements
covering three of them, "no new kinds left" is true and is **not** the question
rung 5 needs answered. Twelve examples of one mode is a thing a rubric clause
can be written against; a mode function with a bounded range cannot express that,
and every mode function here has a bounded range except HotpotQA's, which only
escapes it through an `other:<task>` catch-all that makes each unclassified
error its own species.

So the gate as specified tests *variety* and rung 5 needs *sufficiency*. Both
matter -- a pool that keeps finding new kinds is one you should keep labelling --
but refusing on variety alone rejects the case this whole line of work was
trying to reach.

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
