# Agent0 — Tool-integrated curriculum co-evolution

**Fidelity class: `inference_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning", 2025 ([arXiv:2511.16043](https://arxiv.org/abs/2511.16043)) |
| Upstream code (pinned) | [aiming-lab/Agent0@f775b510](https://github.com/aiming-lab/Agent0/tree/f775b5101e62fe92976831adf4a21a38fcc0a767) |
| Definition | [`examples/agent0/agent0_tool_curriculum.py`](https://github.com/Birfy/agentdescent/blob/main/examples/agent0/agent0_tool_curriculum.py) |
| Domain | self-generated cart arithmetic with a sandboxed calculator; frozen evaluation carts |

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

## Measured results

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/agent0-tool-curriculum.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/agent0-tool-curriculum.json).

| seed | test quality | validation | accepted | invalid | calls |
|---|---|---|---|---|---|
| 0 | 0.000 → **0.500** | 0.000 → 0.438 | 2/80 | 0 | 1616 |
| 1 | 0.125 → **0.625** | 0.000 → 0.500 | 2/80 | 0 | 1584 |
| 2 | 0.000 → **0.500** | 0.000 → 0.688 | 2/80 | 0 | 1678 |

All three seeds moved; mean gain **+0.458**, the largest of the three
inference analogues ([Absolute Zero](algo-absolute-zero.md) +0.313,
[R-Zero](algo-r-zero.md) +0.230) — and the most expensive, at ~1600 calls per
seed against their ~600 and ~940. Four executor samples times two tool turns is
eight model calls per training rollout.

Read the *gain*: the carts are generated, so the baseline is not 0.000 and the
ceiling is not 1.000. See the caveat on
[PromptBreeder](algo-promptbreeder.md#measured-results) on one run per seed.

!!! danger "Both curriculum reward components were missing"
    `curriculum_reward.py`:

    ```python
    final_score = (min(score, 1 - score) if question else -1) - penalty \
                  + calculate_tool_reward(predicts[i])
    ```

    **The uncertainty term was always zero.** `score` there is
    `max_count / len(results)` from `generate_results` — the executor's
    *self-consistency* over repeated samples, the same computation R-Zero uses.
    This port fed the term the single rollout's **grounded reward**, which is 0
    or 1, and `1 - 2|p - 0.5|` is zero at both. The curriculum signal did not
    exist on any item of any run. It samples the executor four times now and
    takes the majority share.

    Upstream's term is `min(p, 1-p)`; the port's `1 - 2|p - 0.5|` is exactly
    twice it. Same shape, and `DifficultyWeighted`'s `4p(1-p)` shares its peak
    and zeros with either.

    **The tool reward was never a number.** `calculate_tool_reward` is
    `min(tool_call_count, 4) * 0.05`. The update prompt said "with a tool-use
    bonus" and carried no value, while the notes claimed the component was
    surfaced. It now reports `R_tool` and the call count that produced it.

!!! note "One `majority_share`, two ports"
    R-Zero's `question_evaluate/evaluate.py` and Agent0's `generate_results` are
    the same computation, so it lives in `_selfplay_domain` — separate copies are
    how a fix to one leaves the other, and both had the same defect.
