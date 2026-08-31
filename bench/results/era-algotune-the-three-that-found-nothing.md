# The three tasks this port does not improve, and what is actually there

At 99 rollouts and two seeds, three of the eight comparison tasks come back at
roughly 1.0x on both seeds:

(The column previously headed `AlphaEvolve` is relabelled: it is not DeepMind's
AlphaEvolve, which has never been run on AlgoTune and is not publicly available, but
MetaEvolve's own untrained baseline -- OpenEvolve's search on Qwen3-14B without its RL.
See `era-algotune-published-comparisons.md`.)

| task | 99, seed 0 | 99, seed 1 | Qwen3-14B, no RL | MetaEvolve (RL) | OpenEvolve |
|---|---:|---:|---:|---:|---:|
| `affine_transform_2d` | 0.992 | 0.978 | 1.072 | 6.945 | 3.22 |
| `eigenvectors_complex` | 1.039 | 1.023 | 1.432 | 1.474 | 1.48 |
| `fft_convolution` | 1.003 | 1.014 | 1.015 | 1.346 | 1.38 |

A harmonic mean is set by its smallest terms, so these three are what the
headline is a statement about. This file asks, for each, where the reference's
time actually goes and what the ceiling is at AlgoTune's calibrated `n`. Every
number below is measured on the same 4-core box the searches ran on, min of 6-9
runs, reference and candidate timed **interleaved** so machine load hits both.

## `eigenvectors_complex` (n=463): closed, and the published numbers are not at this size

The reference sorts eigenpairs in a Python loop, allocating and normalising each
vector separately. That looks like an easy win, and it is not, because it is not
where the time is:

| | ms | share |
|---|---:|---:|
| `np.linalg.eig` alone | 111.71 | **93%** |
| everything else the reference does | 11.5 | 7% |
| reference, total | 123.20 | |

So the ceiling is `123.20 / 111.71 = 1.103x` **if the sort, the normalisation and
the serialisation were all free** -- and they cannot be: `is_solution` rejects an
ndarray (`Solution is not a list of length n`), so `.tolist()` on 463 complex
rows is mandatory. The best valid program measures **1.053x**.

Nor is the eigensolver replaceable:

| | ms | implied ceiling |
|---|---:|---:|
| `numpy.linalg.eig` | 111.71 | 1.103x |
| `scipy.linalg.eig(check_finite=False)` | 111.89 | 1.101x |
| `scipy.linalg.eig(overwrite_a=True)` | 112.60 | 1.094x |
| LAPACK `geev` called directly | 162.59 | 0.758x |

Two consequences.

**This port is done here.** 1.039x against a 1.053x best valid program is 99% of
what exists. The search is not failing on this task; there is nothing to find.

**The three published numbers are not measured at n=463.** 1.432x, 1.474x and
1.48x are all *above the 1.103x physical ceiling* at the calibrated size. They
are not wrong -- they are measured somewhere else. The size curve says where:

| n | reference | `eig` alone | best valid | ceiling |
|---|---:|---:|---:|---:|
| 20 | 0.200 ms | 0.106 ms | 1.710x | 1.889x |
| 50 | 1.144 ms | 0.812 ms | **1.354x** | **1.410x** |
| 100 | 6.142 ms | 5.575 ms | 1.069x | 1.102x |
| 200 | 20.776 ms | 18.929 ms | 1.057x | 1.098x |
| **463** | 120.430 ms | 109.215 ms | **1.053x** | **1.103x** |

All three land in the n~50 band. `era-algotune-openevolve-comparison.md`
demonstrated this size mismatch for OpenEvolve from its published
`algotune.data_size`. For the two rows of arXiv:2607.21971, which state no sizes, the
conclusion has to come from the task instead -- and the first version of this
argument was not strong enough to carry it, because a ratio measured on one
machine can move on another. Two checks close that gap.

**More cores do not raise the ceiling.** `np.linalg.eig` barely threads: dgeev's
QR iteration is sequential, and only the Hessenberg reduction reaches BLAS3.
Same matrix, same box, `OMP_NUM_THREADS` varied:

| threads | reference | `eig` | share | Python tail | ceiling |
|---:|---:|---:|---:|---:|---:|
| 1 | 125.58 ms | 116.81 ms | 93% | 8.36 ms | 1.075x |
| 2 | 127.04 ms | 115.02 ms | 91% | 8.29 ms | 1.105x |
| 4 | 122.76 ms | 110.54 ms | 90% | 8.30 ms | 1.110x |

Four times the cores buys the eigensolver 5.7%. To reach a 1.43x ceiling the
Python tail would have to be 5.7x larger *relative to* the eigensolve -- 0.43 of
it rather than the 0.075 measured here -- and no core count does that.

**AlgoTune's own calibration closes the rest.** The protocol picks `n` so the
reference takes ~100 ms *on the machine you run it on*. A faster LAPACK
therefore does not raise the ceiling; it raises `n`. And the eigensolve's share
of the reference is already at its plateau by n=100:

| n | `eig` share of reference | ceiling |
|---:|---:|---:|
| 20 | 53% | 1.889x |
| 50 | 71% | 1.410x |
| 100 | 91% | 1.102x |
| 200 | 91% | 1.098x |
| 463 | 91% | 1.103x |

So on *any* machine that follows the calibration, the ~100 ms point lands at an
`n` where the ceiling is about 1.10x. Reporting 1.43x-1.48x requires running at
an `n` below ~100, which is well under the calibrated size rather than a
property of the hardware.

## `fft_convolution` (n=542069, mode `same`): about 1.10x, and it is not in the FFT

| | ms | share |
|---|---:|---:|
| `np.array()` on the two Python-list inputs | 56.14 | **31%** |
| `signal.fftconvolve` | 125.22 | 69% |
| reference, total | 181.26 | |

Nearly a third of the reference is turning two lists of ~950k floats into
arrays, which any solver also has to pay -- but not at the same price:

| conversion | ms |
|---|---:|
| `np.array(xs)` (what the reference does) | 56.14 |
| `np.array(xs, dtype=np.float64)` | 42.11 |
| `np.fromiter(xs, np.float64, len(xs))` | **36.19** |

Combined with `signal.oaconvolve` for the transform itself, interleaved against
the reference: **1.105x, valid**. Threading the FFT does not help -- `scipy.fft`
with `workers=-1` on 4 cores measured 1.31x against a *pre-converted* input,
which collapses once the conversion is paid.

So there is roughly 10% here, against the 1.003x this port finds. Real, worth
taking, and an order of magnitude short of MetaEvolve's 1.346x -- which, like
`eigenvectors_complex`, points at a different problem size rather than a better
program.

## `affine_transform_2d` (1123x1123, order 3, mode constant): the one with real room

Here the reference's overhead genuinely is small -- a plain
`affine_transform(image, matrix, order=3, mode='constant')` measures 1.066x, so
94% of the task is the transform itself:

| | ms | share |
|---|---:|---:|
| `spline_filter` (the order-3 prefilter) | 34.90 | 31% |
| the map, `prefilter=False` | 71.16 | 63% |
| reference, total | 112.75 | |

`scipy.ndimage`'s cubic map is single-threaded C. A `@njit(parallel=True)` map
over the prefiltered coefficients, on the same 4 cores, times the whole solve at
**37.72 ms against the reference's 102.07 ms -- 2.706x**.

**That prototype is numerically wrong** (max absolute error 290.7; its B-spline
weights, not its boundary handling, are the bug) so 2.706x is an indication of
where the ceiling is, not a result. What it establishes is that the 63% of this
task which is the map does parallelise, which is the only one of these three
tasks where a large win is available at the calibrated size. Getting the weights
right is ordinary work.

## Summary

| task | this port | measured ceiling | what is there |
|---|---:|---:|---|
| `eigenvectors_complex` | 1.039x | **1.053x** valid | nothing -- 93% is one LAPACK call |
| `fft_convolution` | 1.003x | **1.105x** valid | `np.fromiter` + `oaconvolve` |
| `affine_transform_2d` | 0.992x | **~2.7x** indicated | parallelise the cubic spline map |

Two of the three are closed by the benchmark's own arithmetic. The third is
open, worth roughly a factor of 2.7 on 4 cores, and is where any further effort
on this comparison should go.
