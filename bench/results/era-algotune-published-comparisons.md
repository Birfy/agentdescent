# Other published AlgoTune numbers, and what compares

> **Thread policy, and a correction that was itself wrong.** The sandbox used to
> pin OpenMP/OpenBLAS/MKL/NumExpr/vecLib to one thread but not
> `NUMBA_NUM_THREADS`, so an `@njit(parallel=True)` candidate ran on every core
> against a one-core reference. That asymmetry was real. The first fix pinned
> numba too — which was wrong the other way, because writing a parallel
> implementation *is* an optimisation and upstream sets no thread policy at all.
> Neither side is pinned now, and `RLIMIT_CPU` scales with the core count so a
> parallel candidate is not killed for using what it was given.
>
> Re-measured on an idle box under that policy, every recorded number stands:
> 962.345x measures 992.3x, 280.332x measures 298.1x, 192.321x measures 190.5x,
> and the non-parallel 540.172x measures 538.3x. The reason the asymmetry cost so
> little is that the reference does not parallelise: `np.roots` at n=396 is a
> LAPACK eigenvalue solve whose QR iteration is sequential, timing 116.72 ms
> unpinned against 117.18 ms pinned. Nothing in this file is withdrawn.


Published as a page: https://claude.ai/code/artifact/2fe924b7-09e0-432e-8fc4-58d5128dfa24

## AlphaEvolve and MetaEvolve (arXiv:2607.21971)

Qwen3-14B, 50 rounds, on exactly the eight tasks OpenEvolve publishes. Both papers'
per-task tables reconcile with their headline scores to three decimals -- AlphaEvolve
1.3921 against a published 1.392, MetaEvolve 2.0449 against 2.045. OpenEvolve's does not:
its table gives 2.267x against a published 1.984x.

| task | aligned run | our best | AlphaEvolve | MetaEvolve | OpenEvolve |
|---|---:|---:|---:|---:|---:|
| `convolve2d_full_fill` | 111.915 | 111.915 | 291.338 | 78.128 | 256.15 |
| `affine_transform_2d` | 1.004 | 1.004 | 1.072 | 6.945 | 3.22 |
| `polynomial_real` | 0.996 | 962.345 | 1.014 | 2.457 | 321.01 |
| `psd_cone_projection` | 4.749 | 4.749 | 1.795 | 1.914 | 1.94 |
| `fft_cmplx_scipy_fftpack` | 4.426 | 5.020 | 1.228 | 1.558 | 2.20 |
| `eigenvectors_complex` | 1.017 | 1.017 | 1.432 | 1.474 | 1.48 |
| `fft_convolution` | 0.968 | 1.041 | 1.015 | 1.346 | 1.38 |
| `lu_factorization` | 0.925 | 4.464 | 1.300 | 1.311 | 1.19 |
| **harmonic mean** | **1.443** | **2.232** | **1.392** | **2.045** | 1.984\* |

\* their published headline; the mean of their own table is 2.267x.

Best configuration wins 5 of 8 against each of AlphaEvolve and MetaEvolve. Neither paper
states its problem sizes, so the comparability OpenEvolve fails cannot be confirmed for them
either -- but their per-task values are at least consistent with AlgoTune's distribution at the
calibrated sizes (polynomial_real at 1.014x and 2.457x against upstream's 1.009x median,
rather than OpenEvolve's 321x).

## EvoMem (arXiv:2608.10795)

Gemini 3 Flash, 30 generations of 8 mutants. Reports AlgoTune speedups averaging 8.58x with a
maximum of 16.33x, evaluated on Power Control and Kalman filter -- both of which this port ran.
It publishes aggregate statistics rather than per-task numbers, so this is a bound:

| task | this port | EvoMem | upstream best | upstream median |
|---|---:|---:|---:|---:|
| `power_control` | 297.579 | <= 16.33 reported max | 838.784 | 299.653 |
| `kalman_filter` | 22.681 | <= 16.33 reported max | 85.801 | 12.848 |

## The benchmark's own leaderboard (algotune.io, all 154 tasks)

GPT-5.2 2.05x, Gemini 3.1 Pro 2.02x, GPT-5.4 1.85x, Gemini 3 Pro 1.83x, Claude Opus 4.5 1.77x,
o4-mini 1.72x, DeepSeek R1 1.70x, GPT-5 1.67x, down to GPT-5 Pro (medium) at 1.31x. The paper's
own baseline agent, AlgoTuner on o4-mini-high, scores 1.76x and improves 62.6% of the 154 tasks.
Not directly comparable -- this port has run 29 tasks, not 154 -- but it sets the field's scale.

## Checked and not comparable

* **CodeEvolve** (arXiv:2510.14150) -- evaluates on the AlphaEvolve benchmark suite, not AlgoTune.
* **ThetaEvolve** (arXiv:2511.23473) -- circle packing and mathematical inequalities, not AlgoTune.
* **Epoch AI's AlgoTune page** -- describes the methodology, publishes no scores of its own.
