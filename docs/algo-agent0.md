# Agent0 — Tool-integrated curriculum co-evolution

> **Curriculum/executor co-evolution.** A Curriculum agent writes tasks at the
> Executor's frontier and rewards tool use; the Executor answers them
> stop-and-go through a sandboxed calculator. Runs through the shared
> [`MethodPolicy`](policies.md) runner. Example:
> [`examples/agent0/agent0_tool_curriculum.py`](https://github.com/Birfy/agentdescent/blob/main/examples/agent0/agent0_tool_curriculum.py).

| | |
|---|---|
| **Paper** | *Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning* — 2025 ([arXiv:2511.16043](https://arxiv.org/abs/2511.16043)) |
| **Upstream code** | [aiming-lab/Agent0@f775b510](https://github.com/aiming-lab/Agent0/tree/f775b5101e62fe92976831adf4a21a38fcc0a767) |
| **Example** | [`examples/agent0/agent0_tool_curriculum.py`](https://github.com/Birfy/agentdescent/blob/main/examples/agent0/agent0_tool_curriculum.py) |
| **Domain** | self-generated cart arithmetic with a sandboxed calculator — 16 self-play slots + 16/16 frozen evaluation carts |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `inference_analogue` — [what the classes mean](port-fidelity.md) |

This port is measured in the [runtime matrix](matrix-overview.md): the mechanism
is preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

## The mechanism

A Curriculum agent and an Executor agent, both from the same base model,
co-evolve in alternating iterations. The curriculum reward combines
**uncertainty** `1−2|p̂−0.5|` (executor self-consistency near 50%), a
**tool-use bonus** `min(N_tool, C)`, a BLEU repetition penalty, and a format
gate. The executor trains with **ADPO** (ambiguity-scaled advantages) on
majority-vote pseudo-labels, rolling out multi-turn with a sandboxed Python
interpreter in stop-and-go fashion.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Stop-and-go tool rollouts | request → AST-gated calculator → continue, on both training and frozen evaluation paths |
| Uncertainty + tool-use reward | both components surfaced in the curriculum update prompt |
| Frontier curriculum | `DifficultyWeighted` — the same curve as 1−2\|p̂−0.5\| |

## Boundaries

- Verbal policy memory replaces ADPO post-training.
- One calculator tool replaces the Python interpreter; no repetition penalty.

## Measured results — self-play carts

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge on (this method's own declaration), four executor samples per
generated task, `deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/agent0-tool-curriculum.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/agent0-tool-curriculum.json).

| seed | test quality | validation | accepted | invalid | calls |
|---|---|---|---|---|---|
| 0 | 0.000 → **0.500** | 0.000 → 0.438 | 2/80 | 0 | 1616 |
| 1 | 0.125 → **0.625** | 0.000 → 0.500 | 2/80 | 0 | 1584 |
| 2 | 0.000 → **0.500** | 0.000 → 0.688 | 2/80 | 0 | 1678 |

All three seeds moved, by exactly **+0.500** each — the largest gain of the three
inference analogues ([Absolute Zero](algo-absolute-zero.md#measured-results-self-play-carts)
+0.313, [R-Zero](algo-r-zero.md#measured-results-self-play-carts) +0.230) — and
the most expensive, at ~1600 calls per seed against their ~620 and ~940. Four
executor samples times two tool turns is eight model calls per training rollout.

Read the *gain*: the carts are generated, so the baseline is not 0.000 and the
ceiling is not 1.000. See the caveat on
[PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k) on one run per seed.

!!! note "Both reward components are numbers in the prompt, not adjectives"
    `curriculum_reward.py` is
    `(min(p, 1−p) if question else −1) − penalty + calculate_tool_reward(...)`.

    **Uncertainty** is over the executor's *self-consistency* — `max_count /
    len(results)`, the same computation [R-Zero](algo-r-zero.md) uses — so the
    executor is sampled **four times** per generated task and the majority share
    is what the curriculum sees. A single rollout's grounded reward would be 0 or
    1, where `1 − 2|p − 0.5|` is zero at both ends and the signal does not exist.
    Upstream writes `min(p, 1−p)`; `1 − 2|p − 0.5|` is exactly twice it, and
    `DifficultyWeighted`'s `4p(1−p)` shares its peak and zeros with either.

    **The tool bonus** is `min(tool_call_count, 4) × 0.05`, and the update prompt
    reports `R_tool` with the call count that produced it — a prompt that says
    "with a tool-use bonus" and carries no value is not surfacing the component.

## Run it

```bash
python -m examples.agent0.agent0_tool_curriculum --dry-run

# one seed of the three above
python -m examples.agent0.agent0_tool_curriculum --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --async-ratio 2 --workers 8 --budget-rollouts 80 --staleness full \
    --temperature 0.7 --max-seconds 3600
```

`--async-ratio 2` is explicit because the default here is 1, and 2 is what
this row was recorded at.

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_selfplay_upstream.py`,
`tests/test_candidate_methods.py`.
