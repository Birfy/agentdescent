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

The arm-level comparison was also confounded: `base` ran at the default
`c_puct=1.0` and the other three at 2.5, because a squared prior needs a wide
enough exploration term to bite on. Three seeds of uniform-prior-at-2.5 settle
it:

| arm | seeds | geo mean |
|---|---|---:|
| base — c_puct 1.0, uniform | 0.983 / 1.008 / 126.506 | 5.005x |
| **cpuct — c_puct 2.5, uniform** | **1.004 / 1.006 / 1.007** | **1.006x** |
| prior — c_puct 2.5, prior² | 44.121 / 192.321 / 962.345 | 201.373x |

Widening the exploration term on its own buys nothing — all three control seeds
sit flat at 1.00x, and against `base` the permutation test gives p=0.80. Against
that control every prior seed beats every control seed, **p=0.050**, the floor a
3-vs-3 rank test can reach. What wins is the prior, not the constant.

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

And on other tasks it mostly does nothing. Five of the eight OpenEvolve pairs
are in (base and prior, one seed each, same settings):

| task | base | prior | ratio |
|---|---:|---:|---:|
| convolve2d_full_fill | 103.983x | 101.918x | 0.98x |
| eigenvectors_complex | 1.008x | 1.007x | 1.00x |
| fft_cmplx_scipy_fftpack | 5.020x | 5.019x | 1.00x |
| psd_cone_projection | 3.873x | 3.995x | 1.03x |
| polynomial_real | 1.012x | 540.172x | 534x |

Four of the five move by less than 3%. The prior is not a general accelerator:
it does not raise a ceiling already reached (convolve2d at 104x) and it does not
rescue a task where nothing works (eigenvectors_complex, 1.0x either way). It
pays exactly where it was built to pay — a large win present in the draw
distribution that plain ranking abandons, which so far is one task in five.

## And it can aim the budget at a wall

`least_squares` is the first task where the prior clearly *hurt*, and the
mechanism is the one the node-level result already predicted.

| | base | prior |
|---|---:|---:|
| valid nodes | 20/46 | **3/46** |
| draws | 60 | 85 |
| failed draws | 22 | 56 |
| best on the scoring shards | 5.172x | 1.102x |
| held-out | rejected | rejected |

The ratings on that task, against validity:

| promise | nodes | valid |
|---|---:|---:|
| 3–4 | 2 | 0 |
| 6 | 6 | 1 |
| 7 | 15 | 1 |
| 8 | 17 | **0** |
| 10 | 3 | **0** |

Seventeen nodes rated 8 and three rated 10, and not one of the twenty produced a
valid solution. The two valid rated nodes were the diffident ones. `is_solution`
rejected 26 of the failures outright.

This is Spearman(promise, validity) = 0.046 cashed out. The rating carries no
correctness information, so on a task where the fast direction is also the wrong
direction, aiming the exploration term by promise aims it at the wall — 56 failed
draws against 20, and a tree with three valid nodes in it. A prior over `P(s,a)`
can only be as good as the thing the model is confident about, and the model is
confident about speed.

Both arms lose the task anyway: the held-out set rejects each arm's winner. But
the base arm at least found a 5.172x program to have rejected.

Two pairs remain to run.
