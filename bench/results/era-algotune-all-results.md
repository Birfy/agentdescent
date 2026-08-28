# Every AlgoTune result from this port

> **Correction (numba was unpinned).** Numbers from a candidate compiled
> `@njit(parallel=True)` were inflated: the sandbox pinned OpenMP, OpenBLAS, MKL,
> NumExpr and vecLib to one thread but not `NUMBA_NUM_THREADS`, so such a
> candidate ran on four cores against a one-core reference. Three
> `polynomial_real` winners were affected and are re-measured single-threaded:
> 962.345x -> 342.7x, 280.332x -> 84.8x, 192.321x -> 113.2x. The best
> `polynomial_real` result is now **540.172x**, the Durand-Kerner run, which uses
> no parallelism and re-measures at 540.7x. `ode_stiff_vanderpol` is unaffected
> (identical timing at one and four threads). Fixed in `_era_support.py`, with a
> test.


Generated from the run files in this directory, against AlgoTune's own
`reports/agent_summary.json` (18 models) and OpenEvolve's `examples/algotune`.

Published as a page: https://claude.ai/code/artifact/2fe924b7-09e0-432e-8fc4-58d5128dfa24

## Twelve full-budget tasks (45 iterations, deepseek-v4-flash)

| task | n | aligned run | with prior | our best | upstream best | upstream median | OpenEvolve |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ode_stiff_vanderpol` | 2 | 3005.310 | -- | ** 3005.310** | 2062.527 (o4-mini) | 35.971 | -- |
| `polynomial_real` | 396 | 0.996 | 540.172 | ** 540.172** | 138.469 (glm-4.5) | 1.009 | 321.01 |
| `power_control` | 98 | 297.579 | -- | 297.579 | 838.784 (gemini-3.1-pro-preview) | 299.653 | -- |
| `convolve2d_full_fill` | 6 | 111.915 | 101.918 | 111.915 | 205.513 (claude-sonnet-4-5-20250929) | 144.938 | 256.15 |
| `kalman_filter` | 23 | 22.681 | -- | 22.681 | 85.801 (gemini-3.1-pro-preview) | 12.848 | -- |
| `fft_cmplx_scipy_fftpack` | 1,860 | 4.426 | 5.019 | ** 5.020** | 4.464 (claude-opus-4.6) | 2.324 | 2.20 |
| `psd_cone_projection` | 349 | 4.749 | 3.995 | 4.749 | 12.023 (gemini-3-pro-preview) | 8.728 | 1.94 |
| `lu_factorization` | 1,104 | 0.925 | 4.464 | 4.464 | 35.313 (claude-opus-4.5) | 1.004 | 1.19 |
| `fft_convolution` | 542,069 | 0.968 | 1.041 | ** 1.041** | 1.021 (claude-opus-4.5) | 0.421 | 1.38 |
| `eigenvectors_complex` | 463 | 1.017 | 1.007 | 1.017 | 1.039 (o4-mini) | 1.019 | 1.48 |
| `affine_transform_2d` | 1,123 | 1.004 | -- | 1.004 | 1.015 (gpt-5.2) | 0.220 | 3.22 |
| `lp_centering` | 215 | 0.999 | -- | 1.001 | 1.039 (gemini-3.1-pro-preview) | 1.000 | -- |

Harmonic mean over the twelve: **1.821x** for one aligned run each, **2.591x** for the
best configuration each. On the same twelve tasks that places 6th of 20 and 1st of 20
respectively -- but the second is best-of-N against upstream's one run per model, so only
the first is a placement.

## Sixteen short probes (9 iterations, glm-5.2 -- a fifth of the budget, not comparable)

| task | our best | upstream best | upstream median |
|---|---:|---:|---:|
| `wasserstein_dist` | 8.360 | 314.263 (gemini-3.1-pro-preview) | 9.688 |
| `psd_cone_projection` | 4.342 | 12.023 (gemini-3-pro-preview) | 8.728 |
| `eigenvalues_real` | 2.217 | 2.523 (gpt-5) | 2.452 |
| `ode_lorenz96_nonchaotic` | 2.206 | 11.292 (gemini-3.1-pro-preview) | 1.775 |
| `graph_laplacian` | 1.460 | 74.913 (gemini-3.1-pro-preview) | 0.611 |
| `cholesky_factorization` | 1.346 | 1.362 (claude-opus-4.6) | 1.112 |
| `convex_hull` | 1.205 | 10.812 (claude-opus-4.5) | 1.004 |
| `unit_simplex_projection` | 1.137 | 6.500 (gpt-5.2) | 1.103 |
| `matrix_multiplication` | 1.117 | 1.677 (gemini-3.1-pro-preview) | 0.689 |
| `eigenvectors_real` | 1.098 | 1.164 (gpt-5.2) | 1.014 |
| `ode_stiff_vanderpol` | 1.088 | 2062.527 (o4-mini) | 35.971 |
| `fft_convolution` | 1.063 | 1.021 (claude-opus-4.5) | 0.421 |
| `svd` | 1.038 | 1.617 (o4-mini) | 1.019 |
| `matrix_exponential` | 1.022 | 2.127 (gemini-3.1-pro-preview) | 0.594 |
| `correlate_1d` | 1.018 | 3.199 (gemini-3.1-pro-preview) | 1.035 |
| `sparse_lowest_eigenvalues_posdef` | 1.011 | 2.734 (qwen3-coder) | 1.739 |
| `toeplitz_solver` | 1.008 | 1.021 (gpt-5.4) | 1.001 |
| `convolve_1d` | 0.990 | 1.743 (glm-4.5) | 1.008 |
| `qr_factorization` | 0.976 | 9.566 (gpt-5.2) | 1.173 |
| `lu_factorization` | 0.847 | 35.313 (claude-opus-4.5) | 1.004 |
