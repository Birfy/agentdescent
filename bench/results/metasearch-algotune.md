# Evolving the selection rule on AlgoTune: what it costs, and why it cannot be measured yet

Stage 1b of the meta-evolution plan (`docs/design-meta-evolution.md` §4), run
live for the first time. The port works end to end. **The experiment does not**,
and this page is the quantified reason, so the next attempt starts from the
noise problem rather than from `pip install numpy`.

Produced by [`bench/metasearch_algotune.py`](../metasearch_algotune.py).

## The port runs

One inner search on `psd_cone_projection`, 4 expansions: **36 seconds, 4 real
model calls, and a 5.6x speedup** over the task's own reference (baseline
1.003x), timed in a Bubblewrap sandbox. Nothing about the machinery is missing.

Everything the design doc listed as blocking turned out to be setup:

| | |
|---|---|
| sandbox | `apt-get install bubblewrap`; `sandbox_backend()` then reports Bubblewrap |
| interpreter | `numpy`, `scipy` |
| task fetch | all 147 tasks `prepare_suite()` without error |
| usable tasks | **123 / 147** after also installing `cvxpy networkx numba mpmath pot scikit-learn cython` |

The 24 that still fail need solvers this environment does not have: 13 want
`ortools`, 3 `pysat`, 2 `sympy`, 2 `faiss`, 1 `hdbscan`; two more (`channel_capacity`,
`lp_centering`) fail their own `is_solution` check with a `SolverError`. Install
those and the pool grows further -- the ceiling is not 123.

### Per-task cost, all 147 profiled

One sandboxed evaluation of each task's *own reference* is the atom every
expansion pays twice, so it orders tasks by what a search on them will cost:

| | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| seconds per reference eval | 0.6 | 1.6 | 2.3 | 3.0 | 17.7 |

Cheapest 16 average 1.08s, cheapest 48 average 1.50s. **Whole searches vary far
more than that**: `psd_cone_projection` at 12 expansions takes 43-87s,
`lu_factorization` takes **404s** -- and on `lu` the search never beats
`scipy.linalg.lu` (0.94x), so it is expensive *and* unlearnable. Any run at this
scale has to select tasks by cost, which is what `task_cost.json` is for.

## The experiment does not

### The reward cannot see the rule

Three rules on `psd_cone_projection`, 12 expansions, and then the seed rule
against **itself**:

| rule | AUC | best speedup |
|---|---:|---:|
| seed (flat-PUCT `c = 1`) | 0.8546 | 5.713x |
| greedy (`rank` only) | 0.8504 | 5.582x |
| worst-first (deliberately bad) | 0.8479 | **5.735x** |
| **seed again, same task, same seed** | **0.8491** | |

| | |
|---|---:|
| between-rule spread | **0.0066** |
| **seed rule against itself** | **-0.0055** |

The sanity check passes -- `worst-first` does come last -- but 84% of the spread
is noise. Note also that the deliberately bad rule found the *fastest program of
the three*: at this budget the rules differ in how fast they arrive, not where.

And that was the **good** task. The scaled run's own determinism check, three
runs of the seed rule on `rbf_interpolation`:

```
rewards  0.5680  0.5104  0.6178      spread 0.1074   sd 0.0537
```

**20x worse.** A 0.51-vs-0.62 gap is not timing jitter; it is the inner search
finding *programs of different quality* on different runs.

### Why a completion cache cannot fix it

On the GSM port, `cached_completion` made inner runs exact (paired gains of
identical rules went to 0.000). It cannot here, and the mechanism is in the
prompt builder: `mutation_prompt` assembles three blocks out of **measured
timings** --

* `_eval_block` — the speedup summary
* `_timing_report`
* `_profile_block` — "25 most expensive lines, milliseconds"

so timing jitter changes the prompt text, the cache key is the prompt, the cache
misses, and the model samples a different program, which is timed differently
again. Confirmed in the run: the repeat wrote *new* cache entries rather than
hitting old ones.

This is not a configuration mistake to be fixed. **The measured timing is the
feedback the inner search runs on.** Remove it and the search is blind; keep it
and the search is stochastic. The meta-reward is therefore an expectation over
that stochasticity, and it has to be estimated by sampling.

### How much sampling: an order of magnitude out of reach

With per-task sd 0.054 against a 0.007 signal, resolving the effect at 2 SE
needs roughly **240 paired samples**. At ~75s per inner search and two rules
that is ~10 hours for *one* validation -- and every gate in the outer loop needs
the same resolution.

### The gate eats the budget

The scaled run -- 24 train + 24 validate tasks (6x the script's defaults on both
sides), 5 rounds, 2 workers, 8 expansions -- stopped after three sweeps:

| sweep | held-out | outcome | rollouts | elapsed |
|---|---:|---|---:|---:|
| 0 | 0.658 | committed 1 | 2 | 2117s |
| 1 | 0.658 | nothing (`reasons={}`) | 2 | 2157s |
| 2 | 0.658 | oracle-rejected 1 | 4 | 3961s |

**4 rollouts in 66 minutes, out of 52 inner searches** -- about 92% of the
wall-clock went to evaluation, because every candidate proposal is scored on the
whole held-out set and each of those is a full ERA search.

`held_out` is pinned at 0.658 to three decimals across all three sweeps, which
looked like a broken measurement and is not: `Runtime.eval_one` memoises on
`cache_key(artifact._signature(), task.id, env_fingerprint)`, and the artifact
changed only once (sweep 0 committed; sweep 1 produced nothing; sweep 2 was
rejected). Sweeps 1 and 2 re-scored the same rule and hit the cache.

That memoisation has a consequence worth carrying to any noisy domain: **it
freezes one noisy draw per (artifact, task) for the whole run.** A gate
comparison is then one frozen sample against one fresh sample, so the paired
difference carries `sd x sqrt(2)` ~ 0.076 per task, or **SE ~ 0.024 over the 10
held-out tasks -- 3.4x the 0.007 signal**. Sweep 0's commit and sweep 2's
rejection are both coin flips. The gate is not misbehaving; it is refusing to
distinguish things that this evidence does not distinguish.

The run was stopped there rather than spending three more hours to produce a
number inside its own error bar.

## What would have to change

Not tuning. Two structural facts have to move first:

1. **The selection rule has almost no leverage at this budget.** 8 expansions
   builds a 13-node tree; the variance is dominated by which program the model
   writes, not by which parent was chosen. A selection rule needs a *large* tree
   to matter -- and at ~10s per expansion, a 100+ expansion inner search is
   ~20 minutes, times hundreds of them.
2. **The reward is a sampled expectation, not a function.** Any honest gate here
   needs many samples per comparison, and each sample is a whole ERA search.

Three levers, cheapest first, none of them free:

* **`--test-shards`.** The inner ERA run's own held-out set is **2 shards**
  (it warns about this once per inner run). That is a direct noise source. It
  adds no searches, but it does add evaluation inside each one.
* **Select tasks by stability, not cost.** `psd` 0.0055 vs `rbf` 0.107 is a 20x
  range; this pool is likely a few measurable tasks and many unusable ones.
  Profiling the noise floor per task is the same shape as `task_cost.json`.
* **`algotune_domain(profile=False)`** removes one of the three timing-derived
  blocks. It stabilises the prompt but changes what the model sees, so it is a
  method change to report, not a free fix.

## Reproducing

```bash
apt-get install -y bubblewrap
pip install numpy scipy cvxpy networkx numba mpmath pot scikit-learn cython

python -m bench.metasearch_algotune --dry-run          # says which sandbox it found
python -m bench.metasearch_algotune --provider openai --model deepseek-v4-flash \
    --thinking disabled --determinism-check 3 \
    --rounds 5 --workers 2 --iterations 8 --problems 1 --candidate-timeout 30 \
    --completion-cache .cache/metasearch-algotune --yes
```

`--determinism-check` is the one to run first. It reports the noise floor before
the outer loop starts, and on this domain that number decides whether anything
the run goes on to report is evidence.
