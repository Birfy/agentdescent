# Against OpenEvolve's eight AlgoTune tasks

OpenEvolve publishes eight AlgoTune examples with speedups. This is where the
port stands on the same eight, and what the model prior changed.

## Correction to `ff784f5`

That commit reported "harmonic mean 1.443x against their 1.984x". **Our number
was over eight tasks and theirs over seven.** 1.984x is the harmonic mean of
OpenEvolve's eight minus one — it was computed for `13fc2e3`, when only seven of
ours had finished, and was not recomputed when the eighth landed. Over the same
eight, **OpenEvolve's harmonic mean is 2.267x**. The gap was wider than reported,
not narrower.

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
| **harmonic mean** | **1.443x** | **2.267x** | |

Ahead on two of eight. Their number is a Gemini Flash 2.5/Pro ensemble against
this port's single deepseek-v4-flash, so the benchmark and the quantity match
but the budget and the model do not.

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
| **harmonic mean (these 5)** | **2.043x** | **3.437x** | **3.025x** |

On these five the prior arm passes OpenEvolve — and the entire margin is one
task. Four of the five move by under 3%; `polynomial_real` goes from 1.012x to
540.172x, and because a harmonic mean is dominated by its smallest terms,
converting a single 1.0x into a large number is worth more than everything else
in the table combined. Read the per-task column, not the aggregate: the prior
turned one loss into a win and left four results where they were.

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
