# When a compiler pays on AlgoTune, and why none of these eight winners used one

Seven of OpenEvolve's eight tasks finished on the aligned flow and **not one
winning program uses numba, cython or jax**. Every one is plain numpy or scipy.
That looked like a failure of the search until the references were read.

## The eight references have nothing to compile

| task | n | Python loops | where the time goes |
|---|---:|---:|---|
| polynomial_real | 396 | 0 | `np.roots` |
| convolve2d_full_fill | 6 | 0 | `signal.convolve2d` |
| affine_transform_2d | 1123 | 0 | `ndimage.affine_transform` |
| fft_cmplx_scipy_fftpack | 1860 | 0 | `fftpack.fftn` |
| psd_cone_projection | 349 | 0 | `np.linalg.eig` |
| eigenvectors_complex | 463 | 1 (a sort) | `np.linalg.eig` |
| fft_convolution | 542069 | 0 | `signal.fftconvolve` |
| lu_factorization | 1104 | 0 | `scipy.linalg.lu` |

The time is already inside LAPACK, FFTW and ndimage. `@njit` on a wrapper around
`np.linalg.eig` buys nothing, and numba cannot compile most of these calls at
all.

## Where it does pay, on the tasks this port has run

| task | shape of the reference | winner |
|---|---|---|
| `ode_stiff_vanderpol` | hands `vdp` to `solve_ivp` — a Python call per step | numba, **2785x** |
| `ode_lorenz96_nonchaotic` | hands `lorenz96` to `solve_ivp` | numba direction |
| `power_control` | a cvxpy model, replaced by a hand-written fixed point | numba, **297x** |

## The rule, checked against upstream's 2595 published solvers

Classify all 154 references by whether they hand one of their own functions (or
a lambda) to a library routine, then count how often upstream's own solutions
for that task reached for numba, cython or jax:

| reference shape | tasks | solvers using a compiler |
|---|---:|---:|
| **hands a Python callback to a library** | 24 | **36.8%** |
| a plain library call | 130 | **17.0%** |

The extremes are cleaner than the means. Of the eight tasks where upstream
reached for a compiler most, six are `callback=True` and all six are ODE or PDE
problems with a Python right-hand side — ode_brusselator 88.2%,
ode_nbodyproblem 82.4%, pde_burgers1d, ode_seirs, ode_hodgkinhuxley 70.6%. Of
the six where it reached least, all six are `callback=False`, at 0.0%.

The two exceptions at the top make the rule more precise rather than less.
`wasserstein_dist` and `graph_global_efficiency` are both plain library calls
and both at 70.6% — because the *replacement* introduces the loop: the 1-D
Wasserstein closed form is a sort and a cumulative sum, and graph efficiency
becomes a BFS you write yourself.

So:

> A compiler pays when there is an interpreted loop — either already in the
> reference, because a library routine calls back into Python per step, or in
> the algorithm you replace it with. Against a single call into compiled code it
> buys nothing.

## What this says about the eight-task comparison

All eight of OpenEvolve's tasks are `callback=False` plain library calls. The
set is drawn from the half of the benchmark where compiling is least likely to
be the lever, so seven winners without a compiler is the shape of the tasks, not
a gap in the search.

OpenEvolve's 321.01x on `polynomial_real` fits the same frame from the other
side. JAX there is not compiling a loop; it is substituting XLA's eigenvalue
implementation for LAPACK's. That is the *change the library* lever, not the
*compile the loop* one, and it is the only place in this set where a
compiler-shaped package won anything.
