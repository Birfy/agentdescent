# `polynomial_real`: the winning direction appears in 1 draw in 4, and dies every time

The run finished this task at **0.996x** with 42 of 46 nodes valid. Upstream's
field is bimodal with nothing in between — glm-4.5 138.5x, deepseek-reasoner
134.7x, o4-mini 73.7x, claude-sonnet-4.5 70.5x, all by numba; then gpt-5.2
1.03x, gpt-5.4 1.01x, Gemini 3/3.1 Pro, GPT-5 and Qwen3 Coder at about 1.0x.
OpenEvolve's published example reports 321.01x here, via JAX.

## Four knobs, none of which move it

Draws that leave the reference's companion-matrix/eigenvalue framing — anything
reaching for numba, jax, cython or an iterative root-finder:

| variable | arm | left the framing |
|---|---|---:|
| prompt | `bare` — upstream's own package list | 1/8, 0/10 |
| prompt | a gloss on each library | 0/8 |
| prompt | one sentence inviting their use | 1/10 |
| model | `deepseek-v4-flash` | 0/10 |
| model | `glm-5.2` | 0/10 |
| parent | the reference, `np.roots(c)` | 0/8 |
| parent | the companion matrix written out | 0/8 |
| max_tokens | 8000, clean interleaved batch | 3/25 |
| max_tokens | 16000, same batch | 1/25 |

`max_tokens` was the one that looked real: pooled against earlier probes it gave
1/64 against 5/20, Fisher p=0.0025. The clean same-batch test reverses the sign
and returns p=0.61, and the median program is *shorter* at 16000 (772 chars
against 866) — the budget was never binding. That pooled p-value came from
mixing heterogeneous arms into one, which is the same mistake that produced an
earlier "4/10 breakthrough" that did not replicate.

**Thinking is not on this list because it cannot be measured here.** It works on
this endpoint but takes over 300s a call on the real prompt against 12–21s
without, and 8 of 8 attempts timed out. Unmeasurable at this latency, not
measured and useless.

## Why: the direction is common, and every first attempt is worse

Forty draws, classified and then scored:

```
10 left the framing    6 valid, 0 above 1.05x, best 0.939x
                       0.262x  0.475x  0.525x  0.567x  0.611x  0.939x
                       -inf: is_solution rejected · SIGXCPU · gate · eigvals ValueError
29 stayed              5 valid, 0 above 1.05x, best 1.002x
```

The model proposes iterative root-finding in **one draw in four**. It is not
failing to have the idea. Every first attempt is **slower than LAPACK** — a
first-draft Newton in pure Python is 2–4x slower than a companion-matrix
eigensolve — so the gate rejects all of them, and the gate is *right*: they are
worse programs.

Upstream's 138.5x version is what that idea becomes after tuning: closed forms
below degree 3, `np.roots` through degree 4, and above that exactly ten Newton
iterations from Cauchy-bounded starts, Horner for value and derivative, the
inner loop under `njit`. None of that is reachable from a node that was never
accepted.

## What this is

A structural property of flat tree search on this task, not a knob:

> the path to the win runs through a node that looks like a regression, and
> the acceptance gate exists to reject regressions.

FUTS never selects a rejected node again, so the 25% of draws that find the
direction each die alone. AlgoTuner's ~100-turn loop crosses this because it can
keep editing the same slow program without needing each edit to be an
improvement — `edit`, `eval`, `edit` again — and OpenEvolve's island model keeps
a population rather than a single accepted lineage.

It also explains the shape of the whole eight-task set. Where the win is a
substitution a numerical programmer recognises on sight — convolution in Fourier
space (111.9x here), a cheaper transform (4.43x), a projection written directly
instead of through a convex solver (4.75x) — the *first* attempt is already
faster, so it is accepted and the tree builds on it. Where the win has to be
tuned into existence, the first attempt is slower and the search never starts.

## What it is not

Not evidence that the prompt alignment is wrong, that `deepseek-v4-flash` is too
weak, or that 8000 output tokens is too few. All three were tested and none of
them moved anything.
