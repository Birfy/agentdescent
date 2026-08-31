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

## MetaEvolve and its baseline (arXiv:2607.21971)

> **The `AlphaEvolve` column in the tables below is not DeepMind's AlphaEvolve, and
> earlier revisions of this file were wrong to imply it was.** DeepMind's AlphaEvolve
> has never been run on AlgoTune: it was evaluated on ~50 mathematical problems and on
> Google's own infrastructure kernels, it predates the AlgoTune paper, and it is not
> publicly available, so nobody outside Google can run it on anything. The row named
> `AlphaEvolve` here is **MetaEvolve's own untrained baseline** -- the paper's words:
> "we use the same evolutionary search algorithm implemented in OpenEvolve; the only
> difference lies in the backbone LLM used to generate solutions", all "implemented on
> Qwen3-14B for fair comparison". It is relabelled `Qwen3-14B, no RL` below. Nothing in
> this file is a comparison against DeepMind's system.

The paper is *Teaching LLMs to Self-Evolve: Cultivating Core Meta-Skills with
Reinforcement Learning*, and its two rows are a controlled one-variable ablation:

| | search | model | training | eight-task score |
|---|---|---|---|---:|
| `Qwen3-14B, no RL` | OpenEvolve's | Qwen3-14B | none | 1.392x |
| `MetaEvolve` | OpenEvolve's | Qwen3-14B | RL on synthesised evolution trajectories | 2.045x |

**MetaEvolve trains the model**, which this port does not: RL with verifiable rewards
from test execution, on evolution trajectories (a program, its fitness, and the history
of prior attempts) synthesised from `PRIME-RL/Eurus-2-RL-Data` -- TACO, APPS, Codeforces
and CodeContests. AlgoTune is held out and named as the out-of-distribution evaluation,
"entirely outside the training domain", so this is not train-on-test. 50 rounds of
self-evolution at inference.

Both rows' per-task tables reconcile with their headline scores to three decimals --
1.3921 against a published 1.392, 2.0449 against 2.045. OpenEvolve's does not: its table
gives 2.267x against a published 1.984x.

**All three published columns are the same codebase.** Section 3.1.3 of the paper,
verbatim: *"For AlphaEvolve and MetaEvolve, we use the same evolutionary search
algorithm implemented in OpenEvolve. The only difference lies in the backbone LLM
used to generate solutions at each evolution step."* That covers **both** of its
rows -- MetaEvolve has no inference-time search of its own; its contribution is the
training, and 1.392x -> 2.045x is the backbone changing under a fixed search.

And the paper states no measurement methodology for AlgoTune at all. Section 3.1.1
gives one sentence -- *"we adopt 8 tasks from the AlgoTune benchmark ... spanning
linear algebra, signal processing, and scientific computing"* -- with nothing on how
problems were generated, at what size, or how the speedup was timed. It is not only
`n` that is unstated.

Meanwhile OpenEvolve's AlgoTune harness takes the problem size from a config field
rather than from AlgoTune's calibration:

```yaml
  data_size: 35          # AlgoTune's calibrated n for psd_cone_projection is 349
```
```python
problem = task_instance.generate_problem(n=data_size, random_seed=trial)
```

Neither paper states its sizes. Three of the eight tasks agree with OpenEvolve's
published numbers to within 2.5% -- `psd_cone_projection` 1.914 against 1.94,
`eigenvectors_complex` 1.474 against 1.48, `fft_convolution` 1.346 against 1.38 -- which
is what one would expect if the sizes came along with the harness. That is an inference,
not a demonstration; the demonstration is the ceiling argument further down, which shows
all three columns exceed what is physically possible on `eigenvectors_complex` at n=463.

| task | aligned run | our best | Qwen3-14B, no RL | MetaEvolve (RL) | OpenEvolve |
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

Best configuration wins 5 of 8 against each of the baseline and MetaEvolve. Neither paper
states its problem sizes, so the comparability OpenEvolve fails cannot be confirmed for them
either -- but their per-task values are at least consistent with AlgoTune's distribution at the
calibrated sizes (polynomial_real at 1.014x and 2.457x against upstream's 1.009x median,
rather than OpenEvolve's 321x).

### The reproducible arm: 99 rollouts, one configuration, two seeds

The `our best` column above is a best-of across configurations, and that was its
weakness: the three systems it is set against each report one run per task, so a
best-of column is not the same measurement. It also rested on a search budget --
45 rollouts -- at which this port's own seed-to-seed spread reached 180x.

Below is a single configuration (`--iterations 99 --async --staleness full
--c-puct 2.5 --prior-exponent 2`) run twice, on the same eight tasks. Both seeds
are shown; neither is selected.

| task | 99, seed 0 | 99, seed 1 | Qwen3-14B, no RL | MetaEvolve (RL) | OpenEvolve\* |
|---|---:|---:|---:|---:|---:|
| `convolve2d_full_fill` | 115.835 | 100.721 | 291.338 | 78.128 | 256.15 |
| `affine_transform_2d` | 0.992 | 0.978 | 1.072 | 6.945 | 3.22 |
| `polynomial_real` | 544.454 | 137.005 | 1.014 | 2.457 | 321.01 |
| `psd_cone_projection` | 5.883 | 5.520 | 1.795 | 1.914 | 1.94 |
| `fft_cmplx_scipy_fftpack` | 4.759 | 4.434 | 1.228 | 1.558 | 2.20 |
| `eigenvectors_complex` | 1.039 | 1.023 | 1.432 | 1.474 | 1.48 |
| `fft_convolution` | 1.003 | 1.014 | 1.015 | 1.346 | 1.38 |
| `lu_factorization` | 7.000 | 7.139 | 1.300 | 1.311 | 1.19 |
| **harmonic mean** | **2.285** | **2.254** | **1.392** | **2.045** | 1.984 |

### AlgoTune's own metric, and a correction

An earlier revision of this file said AlgoTune's leaderboard aggregation could not
be reproduced from its per-task data. **That was wrong.** The rule it missed is
stated in AlgoTune's own description of the score: *"solutions that yield invalid
outputs or that have a speedup of under 1x [are] assigned a speedup of 1x"*, a task
with no result likewise counts 1x, and the score is the harmonic mean over every
task. Applied to `reports/agent_summary.json` it reproduces the published
leaderboard to three decimals on all nine models with a published figure:

| model | published | recomputed |
|---|---:|---:|
| gpt-5.2 | 2.05 | 2.054 |
| gemini-3.1-pro-preview | 2.02 | 2.016 |
| gpt-5.4 | 1.85 | 1.854 |
| gemini-3-pro-preview | 1.83 | 1.832 |
| claude-opus-4.5 | 1.77 | 1.766 |
| o4-mini | 1.72 | 1.716 |
| deepseek-reasoner | 1.70 | 1.702 |
| gpt-5 | 1.67 | 1.669 |
| gpt-5-pro (medium) | 1.31 | 1.307 |

Three things follow. **The clip changes what "below 1.0" means**: models this file
had shown at 0.665x-0.848x are at 1.0 under the benchmark's own rule, because those
were raw ratios AlgoTune deliberately does not score. **Nothing needs excluding**:
a missing task counts 1x, so all eighteen models rank on all eight tasks and the
"complete on all eight" filter that dropped nine of them -- including every recent
Anthropic and Google model -- was unnecessary. And **the underlying observation
survives**: those models did submit solvers slower than the reference, mostly by
adding a `.tolist()` the reference does not do; AlgoTune simply chooses not to
punish it.

Scored AlgoTune's way, on the same eight tasks, nobody excluded:

| # | | score | harness |
|---:|---|---:|---|
| 1 | **this port, seed 0** | **2.290** | ERA on AgentDescent |
| 2 | **this port, seed 1** | **2.268** | ERA on AgentDescent |
| 3 | MetaEvolve (RL) | 2.045 | OpenEvolve's search |
| 4 | OpenEvolve, phase 4 -- hints + config\* | 1.984 | OpenEvolve |
| 5 | OpenEvolve, phase 3 -- specific hints\* | 1.886 | OpenEvolve |
| ... | OpenEvolve, phase 1 -- library names\* | 1.381 | OpenEvolve |
| 5 | claude-opus-4.6 | 1.837 | AlgoTuner |
| 6 | gemini-3.1-pro-preview | 1.833 | AlgoTuner |
| 7 | claude-opus-4.5 | 1.830 | AlgoTuner |
| 8 | gpt-5.2 | 1.788 | AlgoTuner |
| ... | | | |
| 17 | Qwen3-14B, no RL | 1.392 | OpenEvolve's search |
| 23 | gpt-5-mini | 1.254 | AlgoTuner |

\* OpenEvolve's row is its own published 1.984x, the last of four phases (1.381x
generic hints, 1.886x specific hints, 1.984x final). **It is not the harmonic mean
of the eight per-task numbers this file quotes for it.** Those come from that
project's *Task-by-Task Optimization Discoveries* section, which is a log of the
best result found for each task across the whole journey -- 321.01x for
`polynomial_real` came from the JAX discovery, 3.22x for `affine_transform_2d`
from the specific-hints phase -- so its harmonic mean, 2.267x, corresponds to no
configuration anyone ran. This file computed and printed that 2.267x twice, and
withdrew it twice; it is recorded here so it does not come back a third time. The
per-task cells stay, because each was really measured, but the column is a
discoveries log and its score row is Phase 4.

### OpenEvolve, split by how much the prompt was told

Their report is four phases with different prompts and a score for each, so the
single row above understates what varies between them:

| phase | what the prompt carried | score |
|---|---|---:|
| 1 | "basic library mentions without implementation details" | 1.381x |
| 3 | detailed implementation hints written from manual discoveries; their own note: "best theoretical performance but raised overfitting concerns" | 1.886x |
| 4 (their headline) | generic hints plus per-task config tuning -- but the block still reads *"JAX ... can provide 100x+ speedups"* and *"Lower-order interpolation: Try order=0,1,2,3"* | **1.984x** |

Two things worth reading off that. Their phases are **not** ordered by hint
strength: the specific-hints arm scored *below* the generic one. And even the
1.984x prompt names JAX with a promised 100x+ and tells the model to try lower
interpolation orders -- which is most of the answer to `affine_transform_2d`,
the task where they report 3.22x against an AlgoTuner field whose best is 1.015x.

This port's arm carries neither: the AlgoTuner system message with upstream's own
bullet list of package names and no gloss on any of them (`--packages bare`). Its
own hint ablation exists but was run on a different task batch and the feature
was then removed, so **no number here compares bare against hinted on these eight
tasks**. What that ablation found, for the record: naming four techniques next to
the parent's profile was worth 3/8 draws reaching for a compiler against 0/8, and
a one-sentence invitation to use the listed packages bought nothing at all -- 0/8
against the bare list's 1/8 on `polynomial_real`.

Read the harness column before the order. The eighteen AlgoTuner rows are at the
calibrated `n`; the three above them run OpenEvolve's search and none of the three
states its sizes. Against the field that is known to be measuring the same
problems, this port's 2.290 sits above claude-opus-4.6's 1.837 -- that is the
comparison this file puts weight on.

## Everything else published on AlgoTune

Searched arXiv, GitHub, HuggingFace and Epoch AI:

| source | harness | model | what it reports |
|---|---|---|---|
| AlgoTune leaderboard | AlgoTuner | 18 models | 154 tasks at the calibrated `n` |
| MetaEvolve + baseline (arXiv:2607.21971) | OpenEvolve's search | Qwen3-14B, +/- RL | 8 tasks, sizes not stated |
| OpenEvolve | own | gemini-2.5-flash + 2.5-pro | 8 tasks, `data_size` far below calibrated |
| Dria, *Towards Open Evolutionary Agents* (HuggingFace community post, Aug 2025) | OpenEvolve, via `algotune_to_openevolve.py` | Gemini Flash 2.5 2.04x (200 iters), Gemma 3 27B 1.63x, Qwen3-Coder 480B 1.41x | 30 tasks, sizes not stated. **Co-authored by Asankhaya Sharma (codelion), OpenEvolve's own author** -- so it is the tool's author reporting on the tool, not a third-party evaluation, and it is a blog post rather than peer-reviewed |
| EvoMem (arXiv:2608.10795) | own | Gemini 3 Flash | aggregate only: 8.58x mean, 16.33x max |
| Epoch AI | none | -- | mirrors the AlgoTune leaderboard, no independent runs |

Four of the six run OpenEvolve's converter or its search, not one of those four
states a problem size, and **two of the four are OpenEvolve's own author writing
about his own tool**. Of everything published on AlgoTune, exactly one source is
both independent of OpenEvolve and known to sit at the calibrated `n`: AlgoTune's
own leaderboard. That is why the comparison this file puts weight on is the one
against those eighteen models, and not the ordering at the top of the table. Checked and *not* on AlgoTune: CodeEvolve (the AlphaEvolve
suite), ThetaEvolve (circle packing), ParEVO (PBBS/ParEval), RL4RLA, ProgramBench.

\* OpenEvolve's column is carried for continuity only. Its `algotune.data_size`
is 10x-4337x below AlgoTune's calibrated `n` on six of these eight tasks, so it
is not measuring the same problems; see `era-algotune-openevolve-comparison.md`.

**The two seeds agree to 1.4%**, and the win counts are identical: 4 of 8
against the untrained baseline on both, 5 of 8 against MetaEvolve on both. Seven of the
eight tasks reproduce within 15% -- median seed-to-seed spread 1.02x. The
exception is `polynomial_real` at 3.97x, which remains genuinely multi-modal:
the search finds a fast direction every time, but not always the same one.

That reproducibility is what the 99-rollout budget bought. It did not raise the
level -- the eight-task harmonic mean went 2.195x (45 rollouts) to 2.285x, +4.1%
-- but at 45 rollouts the same configuration returned 3.645x on one seed and
2.213x on another, a 65% swing, with `polynomial_real` at 540.172x against
2.999x and `lu_factorization` at 4.464x against 0.957x. Those two tasks were
coin flips on whether the search found the direction at all. At 99 rollouts it
finds them every time, and the aggregate stops being a draw.

Two caveats that the number does not carry on its own:

* **`lu_factorization` at 7.0x is not algorithmic.** The winner returns numpy
  arrays where the reference returns three 1104x1104 `.tolist()` calls, which is
  68% of the reference's runtime; AlgoTune's own published solver for this task
  does the same, so it is fair by the benchmark's standard, but discounting it to
  OpenEvolve's 1.19x puts the aggregate at 1.905x.
* **Three tasks find nothing.** `affine_transform_2d` (0.99x),
  `fft_convolution` (1.00x) and `eigenvectors_complex` (1.03x) are where both
  the Qwen3-14B baseline and MetaEvolve beat this port, on both seeds. A harmonic mean is
  set by its smallest terms, so these three are what the headline is actually a
  statement about.

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
