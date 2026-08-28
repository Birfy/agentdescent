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

Five of the eight have been re-run base-against-prior at one seed each
(`--c-puct 2.5 --prior-exponent 2`); `fft_convolution` and `lu_factorization`
are queued and `affine_transform_2d` is running.

| task | oe2-base | oe2-prior | OpenEvolve |
|---|---:|---:|---:|
| convolve2d_full_fill | 103.983x | 101.918x | 256.15x |
| eigenvectors_complex | 1.008x | 1.007x | 1.48x |
| fft_cmplx_scipy_fftpack | 5.020x | 5.019x | 2.20x |
| psd_cone_projection | 3.873x | 3.995x | 1.94x |
| polynomial_real | 1.012x | **540.172x** | 321.01x |
| **harmonic mean (these 5)** | **2.043x** | **3.437x** | 3.025x* |

\* their per-task numbers restricted to these five, which as above is a
best-of-journey figure and not their published score. There is no five-task
subset of their headline to compare against, so treat this column as indicative
only; the eight-task comparison against 1.984x is the real one, and it needs the
three runs still outstanding.

On these five the prior arm is ahead — and the entire margin is one task. Four of
the five move by under 3%; `polynomial_real` goes from 1.012x to 540.172x, and
because a harmonic mean is dominated by its smallest terms, converting a single
1.0x into a large number is worth more than everything else in the table
combined. Read the per-task column, not the aggregate: the prior turned one loss
into a win and left four results where they were.

`polynomial_real` at 540.172x is also, as far as we can tell, the best published
result on that task — against OpenEvolve's 321.01x (JAX) and upstream's best of
138.5x (glm-4.5, numba).

## Still behind, and why

Three of the four remaining losses have a diagnosis that is not the search:

* `affine_transform_2d` — 82 of 100 draws rejected by `is_solution`.
* `fft_convolution` — n=542069, 127 of 130 draws failed, 1 valid node of 46.
* `convolve2d_full_fill` — 111.9x against 256.15x is the one place we are simply
  beaten on the same idea.

Not one of the eight winners uses a compiler, and that is the task set rather
than the search: all eight references are a single call into compiled code with
no interpreted loop for numba to bite on. `era-algotune-when-a-compiler-pays.md`
measures that across upstream's own 2595 solvers.
