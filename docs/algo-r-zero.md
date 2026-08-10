# R-Zero — Challenger/Solver co-evolution

**Fidelity class: `inference_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "R-Zero: Self-Evolving Reasoning LLM from Zero Data", Huang et al., 2025 ([arXiv:2508.05004](https://arxiv.org/abs/2508.05004)) |
| Upstream code (pinned) | [Chengsong-Huang/R-Zero@5699329d](https://github.com/Chengsong-Huang/R-Zero/tree/5699329d018d79535b7910abdedf5a6eebf355fd) |
| Definition | [`examples/r_zero/r_zero_challenger_solver.py`](https://github.com/Birfy/agentdescent/blob/main/examples/r_zero/r_zero_challenger_solver.py) |
| Domain | self-generated cart arithmetic; frozen evaluation carts (deduction + abduction) |

## The mechanism

Two copies of one base model co-evolve in alternating phases: the
**Challenger** is rewarded for questions at the Solver's frontier —
`min(p̂, 1−p̂)`, maximal when the Solver agrees with itself half the time —
minus a BLEU-cluster repetition penalty; the **Solver** trains on majority-vote
pseudo-labels filtered to an informative difficulty band (30–80% at the pinned
revision; the paper says 25–75%). Both are trained with GRPO.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Separate role updates | two plain-text `FieldSlots` keys with separate update calls |
| Uncertainty reward min(p̂,1−p̂) | two solver samples per generated task give an agreement rate, surfaced in the Challenger update |
| GRPO's group-relative shape | `AdvantageAcceptance` shifts the acceptance prior by group advantage |
| Frontier targeting | `DifficultyWeighted` — its 4p(1−p) weight shares peak and zeros with min(p̂,1−p̂) exactly |

## Boundaries

- Verbal role memories replace two GRPO-trained checkpoints.
- No BLEU repetition penalty; evaluation carts are frozen per seed.

## Measured results

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/r-zero-challenger-solver.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/r-zero-challenger-solver.json).

| seed | test quality | validation | accepted | invalid | calls |
|---|---|---|---|---|---|
| 0 | 0.062 → **0.188** | 0.125 → 0.438 | 2/80 | 0 | 943 |
| 1 | 0.062 → **0.375** | 0.125 → 0.375 | 3/80 | 0 | 941 |
| 2 | 0.062 → **0.312** | 0.062 → 0.312 | 1/80 | 0 | 926 |

All three seeds moved; mean gain +0.230. As on
[Absolute Zero](algo-absolute-zero.md#measured-results), read the *gain*: the
carts are generated, so the baseline is not 0.000 and the ceiling is not 1.000.

See the caveat on [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k): one
run per seed does not pin a number here either.

!!! danger "Half of every run's proposals were being thrown away, and counted as invalid"
    The first runs reported `invalid` of **41, 44 and 48 out of 80** — against 2
    to 11 for every other port. The proposals were
    `{"challenger_memory":"","solver_memory":""}`, and the update calls that
    produced them were fine: asked directly, they return 564 to 1632 characters
    of usable policy.

    `clip_text` was the cause, and it is shared by all eleven ports:

    ```python
    if not cleaned or len(cleaned) > max_len:   # max_len = 900
        return fallback                          # ""
    ```

    A function named `clip_text` that **discards** a 901-character answer and
    keeps a 900-character one. The cost lands twice: the proposal is lost, and it
    is counted as *invalid*, which reads in the metrics as the model producing
    junk. R-Zero was hit hardest because its two update prompts ask for a policy
    statement and the model writes one — four of six sampled replies ran over the
    limit.

    It truncates now, on a word boundary. `invalid` went to **0, 0, 0**.

!!! danger "The Challenger's signal had ground truth in it"
    `question_evaluate/evaluate.py` computes
    `score = max_count / len(results)` over `--num_samples` (default **9**)
    solver samples: the share agreeing with the **majority answer**. R-Zero has
    no ground truth for a question its Challenger just wrote — that is the
    premise — and rewards questions the Solver is *self-inconsistent* on.

    This port computed `p_hat = (score + agreement * score) / 2`, mixing in the
    grounded verifier's reward. `p_hat` is the majority share now, over four
    samples rather than two: two give it only 0.5 and 1.0, so the Challenger saw
    a coin flip rather than a frontier. Unparseable replies count as their own
    distinct answers rather than being dropped — a Solver that cannot state an
    answer is not one that agrees with itself, and dropping them makes an
    incoherent batch read as certain.

    `min(p, 1-p)` now peaks at a half, which is what `DifficultyWeighted`'s
    `4p(1-p)` is attached here to match — and why it is *not* attached to
    Absolute Zero, whose `1-r` is monotone.

## Run it

```bash
python -m examples.r_zero.r_zero_challenger_solver --dry-run

# the table above, one seed of the three (0, 1, 2)
python -m examples.r_zero.r_zero_challenger_solver --yes --seed 0 \
    --budget-rollouts 80 --workers 8 \
    --async --async-ratio 1 --max-seconds 3600 \
    --staleness full --temperature 0.7 --no-thinking \
    --provider claude --model deepseek-v4-flash
```

**`--async-ratio 1` is what this row ran at.** The flag was declared by the
shared parser and never passed to `run_port`, so the run took the runner's own
default of 1 while
[`bench/results/r-zero-challenger-solver.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/r-zero-challenger-solver.json)
recorded the 2 its command line had asked for. The flag is threaded now and that
file records 1 — see
[the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line)
for why the default here is 1 rather than the shared 3.

No `--reflective-merge`: the method's own `reflective` declaration set the merge,
and that declaration is a fidelity statement rather than a knob. `--max-seconds`
is the one setting the results file does not record; any value comfortably above
the row's `engine_s` leaves `--budget-rollouts` as the binding stop.

Offline tests: `tests/test_selfplay_upstream.py`.
