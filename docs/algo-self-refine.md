# Self-Refine — Iterative feedback refinement

> **Feedback-loop self-evolution.** One model generates, critiques its own
> answer, and refines from the critique. Runs through the shared
> [`MethodPolicy`](policies.md) runner. Example:
> [`examples/self_refine/self_refine_feedback_loop.py`](https://github.com/Birfy/agentdescent/blob/main/examples/self_refine/self_refine_feedback_loop.py).

| | |
|---|---|
| **Paper** | *Self-Refine: Iterative Refinement with Self-Feedback* — Madaan et al., 2023 ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)) |
| **Upstream code** | [madaan/self-refine@9a206d41](https://github.com/madaan/self-refine/tree/9a206d41e5d2d0c241bb441f41eeadb945afaa55) |
| **Example** | [`examples/self_refine/self_refine_feedback_loop.py`](https://github.com/Birfy/agentdescent/blob/main/examples/self_refine/self_refine_feedback_loop.py) |
| **Domain** | **GSM8K** ([`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), `main`), 64/64/64 splits |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `mechanism_microport` — [what the classes mean](port-fidelity.md) |

This port is measured in the [runtime matrix](matrix-overview.md): the mechanism
is preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

## The mechanism

One model plays generator, feedback provider, and refiner: GENERATE an answer,
FEEDBACK on it (a separate call), REFINE from the critique — iterated on the
same instance up to four times, stopping early when the feedback contains the
stop signal (the pinned gsm runner checks for the literal "it is correct").
No training of any kind.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| FEEDBACK and REFINE as separate calls | a two-call proposal (`proposal_calls_per_candidate=2`) |
| The stop signal | feedback containing "it is correct" ends the refinement — the reserved call budget becomes an upper bound |
| Same-instance iteration | rollout → proposal → held-out rerun |

## Boundaries

- Fundamental analogue: upstream refines the *answer* to one instance; this port refines the *instruction artifact*.

## Measured results — GSM8K

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge on (this method's own declaration), `deepseek-v4-flash` at
temperature 0.7, on a 192-core Linux host. Recorded in
[`bench/results/self-refine-gsm8k.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/self-refine-gsm8k.json).

| seed | test quality | validation | accepted | calls | wall |
|---|---|---|---|---|---|
| 0 | 0.609 → **0.969** | 0.531 → 0.953 | 1/80 | 1072 | 324 s |
| 1 | 0.500 → **0.984** | 0.391 → 0.922 | 3/80 | 1071 | 363 s |
| 2 | 0.547 → **0.875** | 0.562 → 0.922 | 1/80 | 1007 | 315 s |

Mean final **0.943**, mean gain **+0.391**, all three seeds moving. 17 minutes
for the set.

**Read the baseline.** It is 0.500–0.609, not 0.000, and that is the point of
the move described below: `deepseek-v4-flash` already answers half of GSM8K, so
what a method has to work with is the headroom above a real floor rather than
the whole interval.

64 items per split, drawn from GSM8K's own train and test splits — a *window* on
the benchmark, not the whole 8792 rows, which at this budget would be hours. At
sixteen a single item moved the score by 0.0625; sixty-four buys 0.016, and the
gain came out at +0.391 against +0.375 on the smaller window, which is what says
the gain is real rather than small-sample luck.

See the caveat on [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k): one
run per seed does not pin a number here either.

!!! danger "Why a benchmark, and not the fixture this repository used to write"
    These rows used to run on 48 hand-written arithmetic items graded by a rule
    chosen here — and **changed here**, mid-study, when it blocked progress. Its
    baseline was **0.000 by construction**, because the seed instruction could not
    satisfy an output convention no one had told it, so the old row read
    1.000 / 0.938 / 1.000 from a floor of zero. A number produced against a target
    its author can move is not a measurement of the method. GSM8K brings its own
    questions, answer key and grader (the integer after `####` against the last
    number the reply states); none of it is this repository's to adjust.

    Loading it is where that can quietly reverse. `load_split` **asserts** the
    published row counts — 7473 train, 1319 test — because an interrupted fetch
    once returned 944 of 1319 test rows and raised nothing. A truncated benchmark
    reads exactly like a benchmark. The splits come over `HF_ENDPOINT` as parquet
    (the run host cannot reach `huggingface.co`), with `datasets-server` as the
    fallback, and the on-disk cache is written whole and renamed.

## Run it

```bash
python -m examples.self_refine.self_refine_feedback_loop --dry-run

# one seed of the three above
python -m examples.self_refine.self_refine_feedback_loop --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --async-ratio 2 --workers 8 --budget-rollouts 80 --staleness full \
    --temperature 0.7 --max-seconds 3600
```

`--async-ratio 2` is explicit because the default here is 1, and 2 is what
this row was recorded at.

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_self_refine_upstream.py`,
`tests/test_candidate_methods.py`.
