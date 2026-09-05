# The model prior on AlgoTune, against three published evolutionary systems

Eight AlgoTune tasks, one run each, `--prior-exponent 2 --c-puct 2.5`, 45 rollouts,
deepseek-v4-flash on AlgoTuner's own system message with no technique named. The
speedups are over each task's own reference implementation, on a held-back problem
set, with reference and candidate timed in the same sandbox.

| task | n | this port | AlphaEvolve | MetaEvolve | OpenEvolve |
|---|---:|---:|---:|---:|---:|
| `polynomial_real` | 396 | **540.172** | 1.014 | 2.457 | 321.01 |
| `convolve2d_full_fill` | 6 | 101.918 | 291.338 | 78.128 | 256.15 |
| `fft_cmplx_scipy_fftpack` | 1,860 | **5.019** | 1.228 | 1.558 | 2.20 |
| `lu_factorization` | 1,104 | **4.464** | 1.300 | 1.311 | 1.19 |
| `psd_cone_projection` | 349 | **3.995** | 1.795 | 1.914 | 1.94 |
| `fft_convolution` | 542,069 | 1.041 | 1.015 | 1.346 | 1.38 |
| `eigenvectors_complex` | 463 | 1.007 | 1.432 | 1.474 | 1.48 |
| `affine_transform_2d` | 1,123 | 0.994 | 1.072 | 6.945 | 3.22 |
| **harmonic mean** | | **2.195** | **1.392** | **2.045** | 1.984\* |

AlphaEvolve and MetaEvolve are from [arXiv:2607.21971](https://arxiv.org/abs/2607.21971)
— Qwen3-14B, 50 rounds, the same eight tasks. Their per-task tables reconcile with
their published scores: 1.3921 against 1.392 and 2.0449 against 2.045.

\* OpenEvolve's published headline. It does not reconcile with their own per-task
table, which averages 2.267x, because those entries are the best found across a
four-phase journey while the headline is the final configuration's score.

Head to head this port wins 5 of 8 against each of AlphaEvolve and MetaEvolve.

## Three things that make the aggregate less than it looks

**OpenEvolve is not measuring the same problems.** Their per-task `config.yaml` sets
`algotune.data_size`, the `n` handed to `generate_problem`. AlgoTune calibrates that
per task so the reference takes about 100 ms; OpenEvolve does not use it.

| task | AlgoTune's n | OpenEvolve's n | ratio |
|---|---:|---:|---:|
| `polynomial_real` | 396 | 500 | 1x |
| `convolve2d_full_fill` | 6 | 5 | 1x |
| `affine_transform_2d` | 1,123 | 100 | 11x |
| `fft_cmplx_scipy_fftpack` | 1,860 | 95 | 20x |
| `psd_cone_projection` | 349 | 35 | 10x |
| `eigenvectors_complex` | 463 | 25 | 19x |
| `fft_convolution` | 542,069 | 125 | 4337x |
| `lu_factorization` | 1,104 | 25 | 44x |

The two they run at AlgoTune's size are the two where they score in the hundreds; the
six they shrink by 10x to 4337x are the six scoring 1.19x–3.22x. Their column is here
for reference and should not be aggregated against the others. Neither AlphaEvolve's
nor MetaEvolve's paper states its problem sizes, so the same cannot be ruled out for
them — though their per-task values sit inside AlgoTune's own distribution at the
calibrated sizes, where OpenEvolve's do not.

**One of the wins is the reference's serialisation, not a faster algorithm.**
`lu_factorization` at 4.464x returns numpy arrays where the reference returns three
1104×1104 matrices through `.tolist()`. Timed single-threaded, that conversion is
132.5 ms of the reference's 183.2 ms against 36.8 ms for the factorisation itself.
`is_solution` calls `np.asarray` on what it is given, so it is legal and upstream's own
solvers do it — but it is not a better factorisation. Discounting it to what a
non-trick solver measured (0.936x) puts this port at **1.782x**; dropping the task on
both sides puts it at 2.046x against MetaEvolve's 2.223x, i.e. behind.

The other large results are not of that kind. `polynomial_real`'s 540.172x is a
numba-JIT'd Aberth iteration replacing `np.roots`'s companion-matrix eigenvalue solve;
`psd_cone_projection`'s 3.995x replaces `np.linalg.eig` on a matrix known to be
symmetric, and a materialised `np.diag`, with `eigh` and a broadcast. Both were
re-checked against upstream's own `is_solution` on 40 fresh seeds outside anything used
in search: 40/40 accepted.

**One run per task.** Run-to-run spread on `polynomial_real` alone has been measured
from 0.983x to 962x across seeds at these settings, and a harmonic mean over eight
single runs inherits all of it. Treat the aggregate as one sample, not an estimate.

## The prompts are not the same either

OpenEvolve's shipped `system_message` is AlgoTuner's, plus a block naming JAX, Numba,
scipy/BLAS and vectorisation. On `polynomial_real` alone that block is enlarged to add
"that can provide 100x+ speedups" and two lines on `jax.numpy` as a drop-in NumPy
replacement — on precisely the task where JAX won them 321.01x. This port names no
technique anywhere, which is what makes `polynomial_real` the most informative single
cell in the table: 540.172x unhinted against 321.01x hinted, and MetaEvolve's 2.457x.

AlphaEvolve's and MetaEvolve's prompts are not published. Their paper says both use
the evolutionary search implemented in OpenEvolve, which would inherit the hinted
prompts, but their `polynomial_real` results (1.014x and 2.457x) are not what that
hint produced for OpenEvolve, so it cannot be assumed.

## Reproducing

```
python3 -m examples.era.era_algotune --tasks polynomial_real \
    --iterations 45 --workers 3 --shards 6 --test-shards 3 --problems 2 \
    --repeats 3 --c-puct 2.5 --prior-exponent 2 --max-tokens 8000
```

The eight result files beside this one carry the full trees, per-instance timings and
the winning program for each task.
