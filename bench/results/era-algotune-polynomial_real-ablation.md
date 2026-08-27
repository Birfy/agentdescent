# `polynomial_real`: why the search never leaves 1.0x

The run finished this task at **0.996x** with 42 of 46 nodes valid — a healthy
tree that never found a direction. Upstream's field on the same task is bimodal
with nothing in between:

| model | speedup | how |
|---|---:|---|
| glm-4.5 | 138.5x | numba |
| deepseek-reasoner | 134.7x | numba |
| o4-mini | 73.7x | numba |
| claude-sonnet-4.5 | 70.5x | — |
| gpt-5.2 | 1.03x | — |
| gpt-5.4 | 1.01x | — |
| Gemini 3 / 3.1 Pro, GPT-5, Qwen3 Coder | ~1.0x | — |

OpenEvolve's published `examples/algotune` reports **321.01x** here, via JAX.

## What was varied, and what it bought

Each cell counts draws that reached for a compiler (numba / jax / cython),
against `polynomial_real`'s own root unless stated otherwise.

| variable | arm | compiled draws |
|---|---|---:|
| prompt | `bare` — upstream's own package list | 1/8, then 0/10 |
| prompt | a gloss on each library | 0/8 |
| prompt | one sentence inviting their use | 1/10 |
| model | `deepseek-v4-flash` | 0/10 |
| model | `glm-5.2` | 0/10 |
| parent | the reference, `np.roots(c)` | 0/8 |
| parent | the companion matrix, written out in Python | 0/8 |

**One compiled draw in roughly 54.** Three prompts, two models and two parents
move nothing, and the one that did appear failed to compile.

## Why: the win is not an acceleration

Reading upstream's 138.5x solver settles it. It does not compile the eigenvalue
solve. It *replaces the algorithm*:

* degree 0, 1, 2 — closed forms;
* degree ≤ 4 — still `np.roots`;
* **degree ≥ 5 — abandons the companion-matrix eigenvalue method entirely** for
  Newton–Raphson: a Cauchy bound `1 + max|p[1:]|` for scale, starting points
  spread as `bound · cos(2πk/n)`, Horner for value and derivative, **ten
  iterations**, the inner loop under `@jit(nopython=True)`.

That is a mathematical substitution, not a speedup applied to what was there.
Every parent the search can offer — the one-line `np.roots`, or a companion
matrix written out — sits *inside* the eigenvalue framing, and every profile
points at `eigvals`. Nothing in the prompt, the model or the parent supplies
"stop computing eigenvalues and iterate instead".

This is also why the field is bimodal rather than graded: there is no partial
credit for half of that idea. AlgoTuner's ~100-turn loop can try, measure,
revert and try again until it lands; one rewrite per node has no such budget.

Worth recording separately: those ten Newton iterations from `cos`-distributed
starts carry no convergence guarantee. The solver is correct because it lands
inside the checker's `1e-6` relative tolerance on the instances drawn, not
because the method is sound for every polynomial the generator can produce.

## What this does not say

It is not evidence that the prompt alignment is wrong, and it is not evidence
that `deepseek-v4-flash` is too weak. Both were tested here and neither moved.
It says the gap on this task is a missing *idea*, and that the three knobs this
port exposes do not supply ideas.
