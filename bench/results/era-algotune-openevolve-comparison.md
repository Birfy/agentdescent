# Against OpenEvolve's eight AlgoTune tasks

OpenEvolve's `examples/algotune` covers **eight** of AlgoTune's 154 tasks. This
started as a head-to-head on those eight and ended as something narrower: their
scores are measured at problem sizes of their own choosing, so most of the
comparison does not stand. What the model prior changed on this port's side is
unaffected and is the part worth reading.

## Read this first: six of the eight are not the same problem

OpenEvolve's per-task `config.yaml` sets `algotune.data_size`, the `n` handed to
`generate_problem`. AlgoTune calibrates that `n` per task so the reference takes
about 100ms. OpenEvolve does not use AlgoTune's value:

| task | AlgoTune's n | their n | ratio | their score |
|---|---:|---:|---:|---:|
| polynomial_real | 396 | 500 | 1x | 321.01x |
| convolve2d_full_fill | 6 | 5 | 1x | 256.15x |
| affine_transform_2d | 1123 | 100 | 11x | 3.22x |
| psd_cone_projection | 349 | 35 | 10x | 1.94x |
| eigenvectors_complex | 463 | 25 | 19x | 1.48x |
| fft_cmplx_scipy_fftpack | 1860 | 95 | 20x | 2.20x |
| lu_factorization | 1104 | 25 | 44x | 1.19x |
| fft_convolution | 542069 | 125 | **4337x** | 1.38x |

The two tasks they run at AlgoTune's size are exactly the two where they score in
the hundreds. The six they shrink by 10x to 4337x are exactly the six where they
score 1.19x to 3.22x — at those sizes the reference runs in microseconds and
fixed Python overhead dominates, so there is no asymptotic win left to find.

**So `1.984x` and this port's `1.443x` / `2.195x` are not measurements of the
same thing, and the aggregate comparison below is retired.** Everything on this
side runs at AlgoTune's calibrated sizes, which is what makes it comparable to
upstream's own leaderboard; their aggregate is not comparable to either.

Two consequences worth stating, because both correct claims made earlier in this
file:

* **`psd_cone_projection` is the control.** Both systems evolved the same program
  — `eigh`, `np.maximum(..., out=...)`, `(eigvecs * eigvals) @ eigvecs.T`,
  return the array. They report 1.94x at n=35; this port measures 3.995x at
  n=349. Identical code, a factor of two in reported speedup, purely from size.
* **The `lu_factorization` serialisation win was never available to them.** At
  n=25 the three `.tolist()` calls move 1875 floats; at n=1104 they move 3.65
  million and are 68% of the reference's runtime. Their solution keeps the
  `.tolist()` and scores 1.19x, but that is not a miss — at their size there was
  nothing there. The earlier reading, that they searched the same ground and
  missed it, was wrong.

What survives is the per-task comparison on the two size-matched tasks:

| task | n (ours / theirs) | this port | OpenEvolve |
|---|---|---:|---:|
| polynomial_real | 396 / 500 | **540.172x** | 321.01x |
| convolve2d_full_fill | 6 / 5 | 101.918x | **256.15x** |

One each. `polynomial_real` is the stronger of the two for this port: their `n`
is larger, so their number is not helped by a smaller problem, and they reached
321.01x after being told "JAX — JIT compilation ... can provide 100x+ speedups"
while this port names no technique. Their `convolve2d_full_fill` wins on a
`float32` conversion — their prompt also names dtype as a thing to tune.

The rest of this file is kept for the per-task detail and for the base-vs-prior
comparison, which is internal to this port and unaffected.

## Their headline is 1.984x, and it is not the mean of their own table

`6ce6753` "corrected" `ff784f5` by recomputing OpenEvolve's harmonic mean from
their per-task numbers as 2.267x. That correction was wrong and is withdrawn.
Their README says, verbatim:

> **Best AlgoTune Score: 1.984x** (harmonic mean across 8 successful tasks)

The harmonic mean of the eight per-task numbers they publish really is 2.267x,
but the two are different quantities. Their report is a four-phase journey —
generic hints 1.381x, then *manual human analysis* of what the optimisations
should be, then those spoon-fed as specific hints 1.886x, then a final
generic-hint configuration at 1.984x. The per-task "Result: Nx speedup" entries
are the best found anywhere across that journey; 1.984x is the score of the last
configuration. Comparing against the per-task list would be comparing one run of
ours to their best-of-four-phases.

**1.984x is the number to compare against.** The original `ff784f5` figure was
right for the wrong reason and the arithmetic behind the correction was applied
to the wrong quantity.

## The aligned run

Upstream-faithful settings, uniform `1/N` prior, deepseek-v4-flash on AlgoTuner's
own system message with no technique named:

| task | aligned | OpenEvolve | ratio |
|---|---:|---:|---:|
| convolve2d_full_fill | 111.915x | 256.15x | 0.44x |
| psd_cone_projection | 4.749x | 1.94x | **2.45x** |
| fft_cmplx_scipy_fftpack | 4.426x | 2.20x | **2.01x** |
| eigenvectors_complex | 1.017x | 1.48x | 0.69x |
| affine_transform_2d | 1.004x | 3.22x | 0.31x |
| polynomial_real | 0.996x | 321.01x | 0.00x |
| fft_convolution | 0.968x | 1.38x | 0.70x |
| lu_factorization | 0.925x | 1.19x | 0.78x |
| **harmonic mean** | **1.443x** | **1.984x** (published) | |

Ahead on two of eight. Their number is a Gemini Flash 2.5/Pro ensemble against
this port's single deepseek-v4-flash, so the benchmark and the quantity match
but the budget and the model do not.

## The prompts are not the same, and it is worth two of their wins

Their README calls it "The Golden Rule: Libraries YES, Implementation Details
NO", and the effective hints it lists are:

> • **JAX** — JIT compilation for numerical computations that can provide 100x+
> speedups […]
> • Lower-order interpolation: Try order=0,1,2,3 — lower orders can provide
> dramatic speedups

Those two hints name the winning technique for their two tasks that most need
one: `polynomial_real` at 321.01x is JAX JIT, and `affine_transform_2d` at 3.22x
is the interpolation order. The aligned run uses AlgoTuner's own system message,
which names no technique at all — a deliberate fidelity choice, and the reason
those are two of our four worst results.

That makes `polynomial_real` at **540.172x with the model prior and no hint** the
more interesting number of the two: the same task, a larger speedup, and the
direction found rather than supplied.

## What the prior changed

All eight re-run base-against-prior at one seed each, same settings apart from
`--c-puct 2.5 --prior-exponent 2` on the prior arm.

| task | aligned | oe2-base | oe2-prior | OpenEvolve |
|---|---:|---:|---:|---:|
| polynomial_real | 0.996x | 1.012x | **540.172x** | 321.01x |
| convolve2d_full_fill | 111.915x | 103.983x | 101.918x | 256.15x |
| affine_transform_2d | 1.004x | 0.978x | 0.994x | 3.22x |
| fft_cmplx_scipy_fftpack | 4.426x | 5.020x | **5.019x** | 2.20x |
| psd_cone_projection | 4.749x | 3.873x | **3.995x** | 1.94x |
| eigenvectors_complex | 1.017x | 1.008x | 1.007x | 1.48x |
| fft_convolution | 0.968x | 0.983x | 1.041x | 1.38x |
| lu_factorization | 0.925x | 0.936x | **4.464x** | 1.19x |
| **harmonic mean** | **1.443x** | **1.440x** | **2.195x** | **1.984x** |

The base arm lands at 1.440x against the aligned run's 1.443x, which is the
sanity check this comparison needed: the two runs share no rollouts and agree to
0.2% in aggregate, even though individual tasks move by a factor of two. Against
that, the prior arm at 2.195x is ahead on five tasks, level on two and behind on
one, and ahead of OpenEvolve's published 1.984x.

## Three of the four wins are real and one is the reference's serialisation

Read per task, not by the aggregate.

* **`polynomial_real`, 540.172x — algorithmic.** A numba-JIT'd Durand-Kerner
  iteration replacing `np.roots`'s companion-matrix eigenvalue solve. Found with
  no technique named in the prompt, against OpenEvolve's 321.01x found after
  being told "JAX — JIT compilation ... can provide 100x+ speedups".
* **`psd_cone_projection`, 3.995x — algorithmic.** The reference calls
  `np.linalg.eig` on a matrix it knows to be symmetric, materialises
  `np.diag(eigvals)`, and does two full matmuls. The winner uses `eigh` and
  `(eigvecs * eigvals) @ eigvecs.T`.
* **`fft_cmplx_scipy_fftpack`, 5.019x — a library substitution.** JAX's JIT'd
  `fftn` for `scipy.fftpack.fftn`. The reference returns its array directly, so
  there is no serialisation to skip here.
* **`lu_factorization`, 4.464x — not algorithmic.** The winner returns numpy
  arrays where the reference returns `P.tolist(), L.tolist(), U.tolist()` —
  three 1104x1104 matrices boxed into Python lists. Timed single-threaded, as
  the sandbox runs it:

  | | ms |
  |---|---:|
  | `lu(A)` alone | 36.78 |
  | the three `.tolist()` calls | 132.52 |
  | reference, both | 183.19 |
  | winner, arrays returned | 29.97 |

  68% of the reference's time is the serialisation. `is_solution` calls
  `np.asarray` on what it is given, so this is legal and upstream's own solvers
  exploit it routinely — but it is not a better factorisation.

**The margin over OpenEvolve rests on that one.** Put `lu_factorization` back at
the aligned run's 0.925x and the prior arm scores **1.777x**, below their 1.984x.

What the prior can still claim there is the search, not the trick. The base arm
reached for the same task and wrote almost the same program:

```python
P, L, U = lu(A, check_finite=False)
return {'LU': {'P': P.tolist(), 'L': L.tolist(), 'U': U.tolist()}}   # 0.936x
```

It found the flag, which is worth 1.05x single-threaded, and kept the
serialisation, which is worth 6x. The prior arm dropped the `.tolist()`. Same
task, same budget, same model — one arm stopped at the surface of the reference
and the other did not.

## Still behind, and why

Four tasks still lose to OpenEvolve, and the diagnoses are not all the same:

* `affine_transform_2d` (0.994x against 3.22x) — the task is gated by
  `is_solution`, which rejected 100 of the prior arm's 113 draws. Their 3.22x
  came from being told to try interpolation orders 0-3; we name no technique.
* `fft_convolution` (1.041x against 1.38x) — n=542069. Validity here jumped
  from 1 valid node of 46 in the aligned run to 21-23 in both new arms, on a
  configuration that diffs clean, which is worth a controlled check of its own.
* `eigenvectors_complex` (1.007x against 1.48x) — nothing found by either arm.
* `convolve2d_full_fill` (101.918x against 256.15x) — the one place we are
  simply beaten on the same idea, both arms landing near 100x.

Not one of the eight winners uses a compiler except `polynomial_real`, and that
is the task set rather than the search: the other seven references are a single
call into compiled code with no interpreted loop for numba to bite on.
`era-algotune-when-a-compiler-pays.md` measures that across upstream's own 2595
solvers.

## Without the serialisation win

It appears once. Three of the eight references serialise their answer, and only
one of them was exploited:

| reference | returns | our winner |
|---|---|---|
| `polynomial_real` | `computed_roots.tolist()` | also `.tolist()` — 396 roots, nothing taken |
| `eigenvectors_complex` | a list of lists | 1.007x, nothing found either way |
| `lu_factorization` | three 1104x1104 `.tolist()` | returns the arrays — **this is the one** |
| the other five | raw arrays and dicts | no serialisation to skip |

So `polynomial_real`'s 540.172x pays exactly the same conversion the reference
pays, and the five tasks whose references return raw objects offer nothing to
exploit. Removing it from the comparison two ways:

| | this port | OpenEvolve |
|---|---:|---:|
| **A.** eight tasks, `lu_factorization` put back at what a non-trick solver got (0.925–0.936x) | 1.777–1.782x | 1.984x published |
| **B.** seven tasks, `lu_factorization` dropped on both sides | 2.046x | 2.193x* |

\* their published 1.984x with `lu_factorization`'s 1.19x contribution removed
from the harmonic sum. Dropping their weakest task raises their score too, which
is why B is the fairer of the two.

**Without it, this port is 7–10% behind OpenEvolve, not 10.6% ahead.** The two
methods agree: −6.7% on B, −10.4% on A.

The prior is still most of what moved the port, though. On the seven-task set it
takes the base arm from 1.559x to 2.046x, **+31.2%**, on a prompt that names no
technique against theirs that names JAX and interpolation order. What it does
not do is close the gap.

## Against their prompt, tuned and untuned

OpenEvolve's report is staged, and each stage has a score, so their own numbers
say what prompt engineering was worth to them:

| | their prompt | score |
|---|---|---:|
| phase 1 | basic library mentions, no implementation details | 1.381x |
| phase 3 | explicit implementation hints from their manual analysis | 1.886x |
| phase 4 | final — names JAX "100x+ speedups" and interpolation orders 0–3 | **1.984x** |

Tuning the prompt bought them **+43.7%**, 1.381x to 1.984x.

Every number on this port's side uses AlgoTuner's own system message and names no
technique at any point — the equivalent of their phase 1 throughout:

| | this port | score |
|---|---|---:|
| aligned run | uniform `1/N` prior | 1.443x |
| base arm | same, independent rollouts | 1.440x |
| prior arm | the model's own rating in `P(s,a)` | **2.195x** |
| prior arm | with `lu_factorization` discounted to 0.925x | 1.777x |

The prior bought **+52.5%** raw, **+23.4%** discounted.

| | vs their untuned 1.381x | vs their tuned 1.984x |
|---|---:|---:|
| base, no prior | +4.2% | −27.4% |
| prior, discounted | +28.7% | −10.4% |
| prior, raw | +58.9% | +10.6% |

Two readings, and both are fair:

* **The prior is worth about what their prompt tuning was worth.** +52.5% against
  +43.7%, from two mechanisms that cost differently — theirs took a phase of
  manual human analysis of what the optimisations should be, and the JAX hint
  that came out of it is what carries their `polynomial_real`. The prior costs
  one line in the prompt and 8 extra model calls a run.
* **Discount the serialisation trick and it is not.** +23.4% against +43.7%, and
  1.777x still sits below their 1.984x. A prompt that names the winning
  technique is a strong intervention, and on this task set it is still ahead of
  what steering the search buys.

The comparison is not clean on their side either: phase 4 changed evaluator
configuration as well as the prompt, so their +43.7% is not prompt alone.

## The honest summary

Three numbers, and they say different things:

* **1.443x** — the port on upstream-faithful settings, uniform prior. Reproduced
  independently at 1.440x.
* **2.195x** — the same with a model prior in `P(s,a)`. Ahead of OpenEvolve's
  published 1.984x, but the margin is one serialisation trick.
* **1.777x** — that figure with `lu_factorization` discounted to what the
  aligned run got. This is the number to quote if the question is whether the
  search finds better *algorithms* than theirs. It does not, yet.

Set against that: their prompt names JAX and interpolation order, ours names no
technique, and `polynomial_real` at 540.172x unhinted against their 321.01x
hinted is the single result that most favours this port.
