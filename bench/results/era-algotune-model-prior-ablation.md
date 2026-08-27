# A model prior in `P(s,a)` works; refusing regressions does not

Two mechanisms, added in `04c9a79`, tested 2×2 across three seeds on
`polynomial_real` — 45 rollouts, 3 workers, deepseek-v4-flash, everything else
at the aligned-upstream defaults.

* **`--prior-exponent 2`** — the mutation prompt asks for a `PROMISE: <n>` line
  rating the approach 1–10 *after tuning*, and that rating becomes the `P(s,a)`
  slot in PUCT: `p^2 / Σp^2` in place of ERA's uniform `1/N`. Unrated candidates
  take the mean of the rated ones, so a missing line neither helps nor hurts.
* **`--repair-regressions`** — when a candidate is valid but slower than its
  parent, keep drawing instead of accepting it.

| arm | geo mean | median | min | max | calls | wall |
|---|---:|---:|---:|---:|---:|---:|
| base | 5.005x | 1.008x | 0.983x | 126.506x | 101 | 1109s |
| prior | **201.372x** | **192.321x** | **44.121x** | 962.345x | 109 | 1179s |
| repair | 6.999x | 1.204x | 1.016x | 280.332x | 107 | 1132s |
| both | 19.543x | 79.400x | 1.008x | 93.251x | 101 | 1268s |

The prior arm is the only one whose *worst* seed (44.121x) beats the base arm's
median. It costs nothing: 109 model calls against 101, 1179s against 1109s.

But three seeds an arm is three seeds an arm, and the base arm itself spans
0.983x to 126.506x on this task. A rank test over the six prior-vs-base runs
gives **p=0.10** — suggestive, and nothing more.

Worse, the arm-level comparison is confounded. `base` ran at the default
`c_puct=1.0`; the other three ran at 2.5, because a squared prior needs a wide
enough exploration term to bite on. So "prior vs base" is really "prior at 2.5
vs uniform at 1.0", and there is no uniform-prior-at-2.5 control in this design.
The nearest thing is `both` against `repair` — same `c_puct`, same repair loop,
prior the only difference — which goes 19.5x against 7.0x, in the same direction
and just as underpowered. The missing control is queued.

The evidence that does not depend on any of this is one level down.

## The rating is predictive, and it steers

250 nodes across the six rated runs carry a `PROMISE`. Split them at 8:

| | reached ≥2x | didn't |
|---|---:|---:|
| promise ≥ 8 | 60 | 77 |
| promise ≤ 6 | 3 | 90 |

Fisher one-sided **p = 2.2e-13**. As a filter the rating has **92.3% recall**
(60 of the 65 nodes that ever beat 2x announced themselves beforehand) at
**43.8% precision** against a 26.0% base rate. Spearman against log speedup over
the 185 valid rated nodes is **0.53**.

Against *validity* it is worth nothing: Spearman 0.046. The model can tell you
how fast an idea would be if it worked; it cannot tell you whether its own code
runs. That is the right division of labour for a search prior — the sandbox
already measures correctness, and wastes its budget guessing at speed.

Precision of 44% is the point, not a flaw. A prior that separated perfectly
would just be a lookup table for "did I write numba"; the median promise-8 node
still lands at 0.985x. What the rating buys is the *left* column: only 3 of 93
low-promise nodes were worth expanding, and the search now knows that before
paying for the subtree.

It does change where the budget goes:

| promise | nodes | mean visits | ever expanded |
|---|---:|---:|---:|
| 1–6 | 93 | 2.27 | 13 |
| 7–9 | 87 | 10.56 | 60 |
| 10 | 70 | 6.64 | 50 |

Spearman(promise, visits) = 0.49. Mean tree depth rises from 8.0 in the base arm
to 12.3 with the prior — the search stops re-drawing shallow siblings and
follows a line.

## Regressions are load-bearing

Tracing each run's winner back to the root:

| run | best | steps | steps that made things worse |
|---|---:|---:|---:|
| prior-s0 | 192.32x | 11 | 4 |
| prior-s2 | 962.35x | 7 | 0 |
| repair-s2 | 280.33x | 14 | 3 |
| base-s1 | 126.51x | 10 | 3 |
| both-s1 | 93.25x | 8 | 3 |

**9 of the 12 winning lineages pass through at least one step that made things
worse.** `prior-s0`'s 192x descends from a node measured at **0.01x** — a
hundred times slower than the reference. `both-s1`'s 93x descends from two nodes
that produced no valid solution at all.

That is exactly what `--repair-regressions` throws away. Its cost is visible in
the counters: `gave_up` rises from ~5 slots per run to ~24, because the loop
burns its four attempts refusing slower children and comes back with nothing.
Valid nodes in the tree fall from 41 to 33. And `both` (19.5x) is *worse* than
`prior` alone (201.4x) — the repair loop eats what the prior buys.

Keep `--prior-exponent`. Leave `--repair-regressions` off, which is its default.

## What this does not show

One task, on the task whose run-to-run spread is the widest we have measured.
The node-level result (n=250, p=2e-13) is solid and is about the rating itself;
the arm-level result (n=3, p=0.10, and `c_puct` confounded with the prior) is a
hypothesis.

Whether a model prior helps on tasks where *every* direction is mediocre — where
the useful signal would be "none of these are worth 10" rather than "this one
is" — is untested. The eight OpenEvolve tasks, where the aligned run reached a
harmonic mean of 1.443x, are exactly that population.
