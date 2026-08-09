# Reflexion — Verbal reinforcement / episodic memory

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Reflexion: Language Agents with Verbal Reinforcement Learning", Shinn et al., 2023 ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) |
| Upstream code (pinned) | [noahshinn/reflexion@218cf0ef](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0) |
| Definition | [`examples/reflexion/reflexion_episodic_memory.py`](https://github.com/Birfy/agentdescent/blob/main/examples/reflexion/reflexion_episodic_memory.py) |
| Domain | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |

## The mechanism

After a failed attempt, Reflexion converts the trajectory and the external
evaluator's signal into a **verbal reflection**, appends it to an episodic
memory, and retries the same task with that memory in context. The memory is
append-only and bounded to the last Ω entries (Ω=1–3 in the paper;
`memory[-3:]` in the pinned alfworld runs).

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Bounded append-only memory | `WindowedMemory`: commit-ordered content-addressed keys, rendered as the last 3 entries |
| Reflect on external feedback | the proposal call, fed the strict evaluator's feedback |
| Retry with memory | the engine's held-out rerun |
| Parallel reflections | appends never contradict, so they union-merge with no ranking evaluation |

## Boundaries

- Upstream retries the same failed instance; the held-out rerun is the analogue.
- The equal-budget design requests a reflection after every rollout, not only on failure.

## Measured results

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`--reflective-merge`, `deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/reflexion-episodic-memory.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/reflexion-episodic-memory.json).

| seed | test quality | validation | accepted | calls |
|---|---|---|---|---|
| 0 | 0.000 → **0.125** | 0.000 → 0.312 | 4/80 | 440 |
| 1 | 0.000 → **0.125** | 0.000 → 0.188 | 2/80 | 455 |
| 2 | 0.000 → **0.812** | 0.000 → 0.875 | 1/80 | 487 |

Mean 0.354. All three seeds moved, and the spread — two at 0.125 against one at
0.812 — is the result rather than noise around a central value.

### Against the other mechanisms

Same 48 items, same model, same 80-rollout budget:

| | mean test |
|---|---|
| [AFlow](algo-aflow.md) | 0.896 |
| **Reflexion** | 0.354 |
| [PromptBreeder](algo-promptbreeder.md) | 0.271 |

Reflexion is being asked a question its paper does not ask. Upstream's memory is
**per task instance** and its whole premise is retrying *that* instance; this
port shares one memory across the domain, so what is measured is whether verbal
reflection **transfers**. The answer here is "partly, and unreliably".

!!! danger "Without the worked examples, the reflector answered the arithmetic"
    `_generate_reflection_query` prepends `FEW_SHOT_EXAMPLES` under *"Here are
    two examples:"* — two failed trajectories each followed by its `New plan:`.
    A first version of this port matched upstream's **wording** and dropped that
    **structure**, and the run scored 0.125 / 0.000 / 0.062.

    Measured over 40 reflections with no examples:

    | | |
    |---|---|
    | reflections generated | 40 |
    | that mentioned the output convention | **1** |
    | the first four, verbatim | `925`, `264`, `465`, `$32.15` |
    | entries in the final memory | **0** |

    The query ends in `New plan:` and contains a question and its answer, so with
    nothing to imitate the likelier continuation is simply to answer it. The
    entries were then useless, nothing cleared the acceptance gate, the memory
    stayed empty, and every rollout saw the bare seed instruction. Adding the
    examples took the mean from 0.062 to 0.354.

    The first draft of those examples used `m03` and `m11` — real domain items,
    which land in held-out or test depending on the seed, so every reflection
    prompt in the run handed over two graded answers.
    `test_the_examples_do_not_hand_over_a_held_out_answer` caught it; the items
    are invented now.
