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
   unsafe syntax and hard-coded evaluator optima before the sandbox starts.
5. Candidate isolation clears the environment, removes network access, mounts
   the candidate read-only, and applies CPU, address-space, file-size,
   open-file, and process limits inside the sandbox runner.

Isolation has two backends and picks whichever the host offers: **Bubblewrap**
(`bwrap`) on Linux, and **Seatbelt** (`sandbox-exec`) on macOS. Both deny
network access and confine writes to a scratch directory; the resource limits
come from `setrlimit` inside the runner and so are the same on either platform,
except that Darwin has no `RLIMIT_AS` and the runner reports which limits the
platform refused rather than pretending it applied them. A host with neither
backend raises rather than running model-written code unsandboxed. The offline
test suite skips only the sandbox execution test when no backend is available;
all strategy, archive, engine, and CLI tests still run.

The two backends are not equivalent, and the difference is stated rather than
glossed: Bubblewrap additionally clears the environment and mounts the root
read-only, where Seatbelt leaves reads and the environment alone. The Seatbelt
profile is therefore a defence-in-depth layer over an AST gate that has already
rejected imports and attribute access, not the only thing between a model and
the disk. What it does enforce is checked against the kernel rather than by
reading the profile back:
`test_seatbelt_actually_blocks_the_writes_the_profile_claims_to_block` runs a
process under the real profile and asserts that a write outside the scratch
directory fails, a write inside it succeeds, and `socket.create_connection`
cannot reach the network.

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

!!! note "`--serial` and the benchmark's `serial` mode are two different baselines"
    `--serial` is the [shared port flag](self-evolution-examples.md#the-shared-command-line):
    **one worker**, so there is nothing to merge and the loop is the published
    one. `bench/openevolve_agentdescent.py` also has a mode called `serial`, and it
    means something narrower — `evolve(max_concurrency=1)` with the full worker
    count, i.e. the same algorithm run without thread concurrency. That mode
    isolates threading; the flag isolates merging.

## Live benchmark method

| Setting | Value |
|---|---|
| Model | `deepseek-v4-flash`, Anthropic-shaped API |
| Sampling | temperature 0.7, **thinking disabled**, `--max-tokens 32000` |
| Mode | `async_evolve(n_workers=4, async_ratio=3)`, `--staleness full` |
| Budget | 24 rollouts, hard-capped; `--max-seconds 1800` |
| Islands | 3, migration interval 4 |
| Evaluator tasks | 8: 6 train and 2 held out |
| Independent test | 6 disjoint seeds after the run |
| Objective budget | 200 objective calls per evaluator seed |
| Isolation | Seatbelt (`sandbox-exec`), macOS |
| Replay | none; a single live engine run |

Exact reproduction command:

```bash
python -m examples.openevolve.openevolve_program_evolution --yes --seed 0 --tasks 8 --budget-rollouts 24 --workers 4 --islands 3 --migration-interval 4 --async --async-ratio 3 --staleness full --max-seconds 1800 --max-tokens 32000 --model deepseek-v4-flash
```

The recorded output is
[`bench/results/openevolve-quality-run.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/openevolve-quality-run.json).
The three-mode timing harness is still
[`bench/openevolve_agentdescent.py`](https://github.com/Birfy/agentdescent/blob/main/bench/openevolve_agentdescent.py);
this page reports quality, and the cross-algorithm speedup numbers live in
[efficiency.md](efficiency.md).

## Results — quality, on `deepseek-v4-flash`

24 rollouts, 3 islands, `--staleness full`, 4 workers, 24 mutation calls,
76,294 tokens, 0 failures, 50.8 seconds of wall clock. Every figure below is
scored on the **held-out** trial seeds, which the search never saw:

| | baseline | best found | |
|---|---:|---:|---:|
| combined score | 0.9638 | **1.4995** | ceiling is 1.5 |
| framework reward | 0.7620 | **0.9997** | |
| mean distance to the global optimum | 0.5260 | **0.00057** | 920× closer |
| mean value found (optimum: -1.5187) | -1.2892 | **-1.5187** | |
| value standard deviation | 0.1392 | **0.0** | same answer every seed |

The ceiling is 1.5 because `combined_score` is
`(0.5·value + 0.3·distance + 0.2·reliability)` times a basin multiplier that
tops out at 1.5, and the winner scores 0.9997 / 0.9994 / 1.0 on the three terms.
The evolved program is not merely better than random search; it has effectively
solved the benchmark.

The winning program appeared at iteration 15 on island 2 and its own summary
reads: *"Replaced the final random perturbation phase with a Nelder-Mead simplex
refinement step ... then adds a lighter fine-tuning pass."* 586 characters of
uniform random search became 7,848 characters of coarse grid, adaptive-step
compass search, and simplex refinement.

The archive machinery is all live: 25 programs evaluated, 22 valid, island cell
counts `[3, 2, 2]`, and **3 ring migrations**. `--staleness full` considered 24
stale cards and discarded none.

!!! danger "Two settings made this look like a search that found nothing"
    Before these were fixed the same configuration reported
    `reward 0.7639 -> 0.7639` twice, with no warning anywhere.

    **`--thinking disabled` was gated on the GLM path.** It is the default, but
    the condition also required `is_openai_compatible(args)` and a GLM model, so
    against an Anthropic-shaped endpoint it silently did nothing. Measured on
    `deepseek-v4-flash`: with reasoning on, a mutation call spends **96% of its
    output on thinking** -- 75k characters of it against a 3.2k program -- takes
    a median 234 s, and hit the 32000-token ceiling in 3 of 6 samples. With it
    off: 512–990 output tokens, 7 s, and 0 of 6 unparseable.

    **`--max-tokens` defaulted to 2048.** Upstream's `config.yaml` says 16000 and
    upstream mutates with SEARCH/REPLACE diffs; this port rewrites the whole
    genome, which needs *more* than upstream, not less. Six samples: 2048 gave no
    parseable program at all, 16000 failed 5 of 6, 32000 failed 2 of 6.

    Both failures are the same shape. The budget runs out before any visible
    content is emitted, the reply arrives **empty**, and the round records a
    candidate that simply did not improve -- which is exactly what a search that
    found nothing also looks like. `make_propose` now counts empty replies and
    warns, naming both `--thinking` and `--max-tokens` as the causes.

!!! note "This run is why the row is no longer Linux-only"
    Candidate isolation was Bubblewrap-only, so the matrix skipped this row on
    any non-Linux host (`needs="bwrap"`). The Seatbelt backend described under
    [Fidelity and boundaries](#fidelity-and-boundaries) was added to run it here,
    and the `needs` marker is gone.

