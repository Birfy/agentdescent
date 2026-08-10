# R-Zero — Challenger/Solver co-evolution

> **Co-evolution of two roles.** A Challenger writes questions at the Solver's
> frontier and a Solver trains on majority-vote pseudo-labels. Runs through the
> shared [`MethodPolicy`](policies.md) runner. Example:
> [`examples/r_zero/r_zero_challenger_solver.py`](https://github.com/Birfy/agentdescent/blob/main/examples/r_zero/r_zero_challenger_solver.py).

| | |
|---|---|
| **Paper** | *R-Zero: Self-Evolving Reasoning LLM from Zero Data* — Huang et al., 2025 ([arXiv:2508.05004](https://arxiv.org/abs/2508.05004)) |
| **Upstream code** | [Chengsong-Huang/R-Zero@5699329d](https://github.com/Chengsong-Huang/R-Zero/tree/5699329d018d79535b7910abdedf5a6eebf355fd) |
| **Example** | [`examples/r_zero/r_zero_challenger_solver.py`](https://github.com/Birfy/agentdescent/blob/main/examples/r_zero/r_zero_challenger_solver.py) |
| **Domain** | self-generated cart arithmetic — 16 self-play slots + 16/16 frozen evaluation carts (deduction + abduction) |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `inference_analogue` — [what the classes mean](port-fidelity.md) |

This port is measured in the [runtime matrix](matrix-overview.md): the mechanism
is preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

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

## Measured results — self-play carts

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge on (this method's own declaration), four solver samples per
generated question, `deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/r-zero-challenger-solver.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/r-zero-challenger-solver.json).

| seed | test quality | validation | accepted | invalid | calls |
|---|---|---|---|---|---|
| 0 | 0.062 → **0.188** | 0.125 → 0.438 | 2/80 | 0 | 943 |
| 1 | 0.062 → **0.375** | 0.125 → 0.375 | 3/80 | 0 | 941 |
| 2 | 0.062 → **0.312** | 0.062 → 0.312 | 1/80 | 0 | 926 |

All three seeds moved; mean gain +0.230. As on
[Absolute Zero](algo-absolute-zero.md#measured-results-self-play-carts), read the *gain*: the
carts are generated, so the baseline is not 0.000 and the ceiling is not 1.000.

See the caveat on [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k): one
run per seed does not pin a number here either.

!!! note "`p̂` carries no ground truth, by construction"
    `question_evaluate/evaluate.py` computes `max_count / len(results)` over
    `--num_samples` (default **9**) solver samples: the share agreeing with the
    **majority answer**. R-Zero has no ground truth for a question its Challenger
    just wrote — that is the premise — and rewards questions the Solver is
    *self-inconsistent* on, so the grounded verifier's reward must stay out of
    this term.

    Four samples here, not two: two give `p̂` only 0.5 and 1.0, which is a coin
    flip rather than a frontier. Unparseable replies count as their own distinct
    answers rather than being dropped — a Solver that cannot state an answer is
    not one that agrees with itself, and dropping them makes an incoherent batch
    read as certain.

    `min(p̂, 1−p̂)` peaks at a half, which is what `DifficultyWeighted`'s
    `4p(1−p)` is attached here to match — and why it is *not* attached to
    [Absolute Zero](algo-absolute-zero.md), whose `1−r̄` is monotone.

## Run it

```bash
python -m examples.r_zero.r_zero_challenger_solver --dry-run

# one seed of the three above
python -m examples.r_zero.r_zero_challenger_solver --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --workers 8 --budget-rollouts 80 --staleness full \
    --temperature 0.7 --max-seconds 3600
```

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_selfplay_upstream.py`,
`tests/test_candidate_methods.py`.
