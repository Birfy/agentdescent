# Against OpenEvolve's eight AlgoTune tasks

OpenEvolve's `examples/algotune` covers **eight** of AlgoTune's 154 tasks. This
is where the port stands on the same eight, and what the model prior changed.

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
