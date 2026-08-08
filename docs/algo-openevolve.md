# OpenEvolve - program evolution

This example ports OpenEvolve's function-minimization program search onto the
AgentDescent evolution engines. Python source is the evolving artifact, model
calls propose mutations, a sandboxed evaluator supplies reward, and a custom
aggregator maintains the quality-diversity archive.

The runnable port is
[`examples/openevolve/openevolve_program_evolution.py`](https://github.com/Birfy/agentdescent/blob/main/examples/openevolve/openevolve_program_evolution.py).
It supports synchronous data parallelism through `evolve()` and barrier-free
execution through `async_evolve()`.

## Algorithm mapping

| OpenEvolve mechanism | AgentDescent representation |
|---|---|
| `EpsilonGreedy` | the in-pool parent pick as a named [`SelectionPolicy`](selection.md) |
| Python program candidate | `Task` rollouts over a source-code artifact |
| Model mutation | `propose(rendered, task, output, reward)` |
| Full program replacement | `OpenEvolveStrategy.to_diff()` |
| Function-minimization evaluator | `run()` plus `reward_program()` |
| MAP-Elites islands | `OpenEvolveAggregator` and its shared archive |
| Concurrent candidate workers | `evolve(max_concurrency=...)` |
| Completion-order commits | `async_evolve(async_ratio=...)` |
| Versioned best program | AgentDescent `Ledger` dev head |
| Stale candidate handling | AgentDescent evidence cards and staleness policy |

The artifact is generated executable code, so the port declares
`blast_radius=0.6` and is classified as an L1 change. The algorithm-specific
evaluator remains the acceptance authority, as it is in OpenEvolve.

## Fidelity and boundaries

The reference is OpenEvolve commit
[`411fb59c886c18704caaffb611e17cf9e7d824d2`](https://github.com/algorithmicsuperintelligence/openevolve/tree/411fb59c886c18704caaffb611e17cf9e7d824d2),
specifically `examples/function_minimization` and the database implementation.

Preserved mechanics:

1. Python source is the genome and a model mutates a selected parent.
2. The evaluator's value, distance, reliability, and basin-multiplier formula is
   preserved from
   [`evaluator.py`](https://github.com/algorithmicsuperintelligence/openevolve/blob/411fb59c886c18704caaffb611e17cf9e7d824d2/examples/function_minimization/evaluator.py#L190-L215).
3. Parent selection mixes exploitation and exploration.
4. Each island owns a MAP-Elites grid over program length and code diversity.
5. Children remain on their target island and elites migrate in a ring.

Intentional differences:

1. The compact genome is rewritten in full instead of using SEARCH/REPLACE
   patches.
2. Feature bins use fixed length boundaries and insertion-time token-Jaccard
   diversity rather than evolving min/max scaling.
3. AgentDescent supplies workers, the ledger, evidence cards, barriers, and the
   barrier-free runtime. OpenEvolve's process controller and database therefore
   become a `Strategy` and custom `Aggregator`.
4. Candidate execution is deterministic and budgeted. An AST gate rejects
   unsafe syntax and hard-coded evaluator optima before Bubblewrap starts.
5. Bubblewrap clears the environment, removes network access, mounts the
   candidate read-only, and applies CPU, address-space, file-size, open-file,
   and process limits inside the sandbox runner.

The evaluator is Linux-specific because it requires `bwrap`. The offline test
suite skips only the sandbox execution test when Bubblewrap is unavailable; all
strategy, archive, engine, and CLI tests still run.

## Running the port

Preview without an API key, network access, or sandbox process:

```bash
python -m examples.openevolve.openevolve_program_evolution --dry-run
```

Run six mutations with GLM-5.2 through an OpenAI-compatible endpoint:

```bash
python -m examples.openevolve.openevolve_program_evolution \
  --provider glm --model glm-5.2 --iterations 6 --workers 3 --yes
```

Add `--async` for the barrier-free engine, or `--serial` for the upstream serial
algorithm. `OPENAI_API_KEY` and `OPENAI_BASE_URL` must be set for the `glm`
provider.

!!! note "`--serial` and the benchmark's `serial` row are two different baselines"
    `--serial` is the [shared port flag](self-evolution-examples.md#the-shared-command-line):
    **one worker**, so there is nothing to merge and the loop is the published
    one. The benchmark table below has a row also called `serial`, and it means
    something narrower — `evolve(max_concurrency=1)` with the full three workers,
    i.e. the same algorithm run without thread concurrency. That row isolates
    threading; the flag isolates merging. The benchmark reaches its mode directly
    through `run_agentdescent_openevolve(mode="serial", workers=3)`, so the two
    are independent and the recorded numbers below are unaffected by the flag.

## Live benchmark method

The benchmark is
[`bench/openevolve_agentdescent.py`](https://github.com/Birfy/agentdescent/blob/main/bench/openevolve_agentdescent.py),
and its compact recorded output is
[`bench/results/openevolve-agentdescent-live.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/openevolve-agentdescent-live.json).
It compares three executions of the same port:

| Mode | Engine configuration |
|---|---|
| serial | `evolve(max_concurrency=1)` |
| sync | `evolve(max_concurrency=3)` |
| async | `async_evolve(n_workers=3, async_ratio=1)` |

The recorded experiment used:

| Setting | Value |
|---|---|
| Model | GLM-5.2, OpenAI-compatible API |
| Sampling | temperature 0.7, thinking disabled |
| Repeats | 3, with evaluator seeds 0, 100, and 200 |
| Mode order | rotated each repeat to reduce order and warm-up bias |
| Mutation budget | exactly 6 reserved model calls per mode and repeat |
| Workers | 3 |
| Evaluator tasks | 12: 6 train and 6 held out |
| Independent test | 6 disjoint seeds after each run |
| Objective budget | 200 objective calls per evaluator seed |
| Quality target | fixed normalized validation reward of 0.8 |
| Replay | none; every observation is a live engine run |

The quality target was fixed before the matrix run. The three random-search
baselines measured 0.681 to 0.729, so 0.8 required a real improvement and still
left headroom. Validation reward is the mean per-seed evaluator score normalized
to `[0, 1]`; the independent test table reports OpenEvolve's aggregate combined
score, whose upper range is 1.5.

Async shutdown grace was 120 seconds. This matters because a model request that
has started cannot be cancelled: the benchmark waits for in-flight work so its
end-to-end time and token count include the full cost rather than stopping the
clock at the first useful commit.

Exact reproduction command:

```bash
python -m bench.openevolve_agentdescent --yes \
  --model glm-5.2 --repeats 3 --modes serial sync async \
  --iterations 6 --workers 3 --tasks 12 --test-trials 6 \
  --temperature 0.7 --thinking disabled --quality-target 0.8
```

## Results

All 9 runs completed. They made 54 mutation-model calls, used 138,947 tokens,
and recorded 0 API failures. Each interval below is minimum / median / maximum
across three repeats. TTQ includes only runs that reached the fixed target.

| Mode | Reached | End-to-end seconds | TTQ seconds | Final validation reward | Independent test gain |
|---|---:|---:|---:|---:|---:|
| serial | 3/3 | 153.23 / 164.88 / 205.09 | 63.52 / 66.93 / 67.58 | 0.889 / 0.919 / 1.000 | -0.049 / +0.115 / +0.777 |
| sync | 2/3 | 57.33 / 63.28 / 70.74 | 57.21 / 63.91 / 70.62 | 0.777 / 0.875 / 0.998 | -0.186 / +0.179 / +0.513 |
| async | 2/3 | 53.27 / 71.53 / 85.72 | 25.71 / 27.26 / 28.80 | 0.729 / 1.000 / 1.000 | +0.000 / +0.300 / +0.520 |

Speedups are paired by repeat and evaluator seed. A value above 1 means the
comparison mode was faster.

| Comparison | Paired runs | Speedup min / median / max |
|---|---:|---:|
| sync vs serial, end-to-end | 3 | 2.42 / 2.88 / 2.90 |
| async vs sync, end-to-end | 3 | 0.74 / 0.99 / 1.08 |
| sync vs serial, TTQ | 2 | 0.95 / 1.06 / 1.18 |
| async vs sync, TTQ | 2 | 1.99 / 2.37 / 2.75 |

The median raw end-to-end time fell from 164.88 seconds in serial mode to 63.28
seconds in sync mode. The paired result is consistent across all three seeds,
so synchronous parallelism is the strongest result in this small experiment.

Async reached the target about 2.37 times sooner than sync on the two paired runs
where both reached it. This demonstrates the intended completion-order benefit:
a useful candidate can commit without waiting for the rest of its cohort.
However, async did not reduce full return time: its paired median end-to-end
speedup was 0.99, with two slower runs and one faster run. Waiting for in-flight
requests accounts for much of that difference.

The independent test gains range from negative to strongly positive. That is a
visible small-sample overfitting warning, not a result to discard. The experiment
supports a timing claim about the runtime; it does not establish that one runtime
produces better programs than another. The evaluator itself took less than one
second at the median in every mode, so the observed timing is dominated by model
latency rather than CPU evaluation.

Finally, `n=3` is deliberately reported as a demonstration, not a production
latency estimate. Async discarded 0, 2, and 1 stale cards across its three runs,
and no worker retired. Larger repeated studies should report the same TTQ,
end-to-end, held-out test, staleness, retirement, call, token, and failure fields.
