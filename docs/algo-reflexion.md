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

## Measured results — GSM8K

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/reflexion-gsm8k.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/reflexion-gsm8k.json).

| seed | test quality | validation | accepted | calls |
|---|---|---|---|---|
| 0 | 0.797 → 0.891 | 0.844 → 0.828 | **0/80** | 678 |
| 1 | 0.750 → 0.688 | 0.688 → 0.719 | **0/80** | 683 |
| 2 | 0.734 → 0.656 | 0.688 → 0.734 | **0/80** | 680 |

**Nothing was accepted, on any seed.** Eighty reflections a run, none of which
improved the held-out split enough to clear the gate. This row does not show a
method doing badly so much as a method whose premise the framework cannot host:
Reflexion's memory is per *instance* upstream and its whole move is retrying
**that** instance, and a shared memory asked to transfer across unseen questions
is a question the paper does not ask.

!!! danger "So the numbers above are the noise floor, not a result"
    With nothing accepted the artifact never changed, which makes each row's
    baseline and final **two evaluations of the same instruction**. They differ
    by +0.094, −0.062 and −0.078.

    That is what re-scoring one instruction on 64 GSM8K items at temperature 0.7
    costs, and it is the scale every gain in this study should be read against —
    [PromptBreeder](algo-promptbreeder.md)'s +0.495 and
    [AFlow](algo-aflow.md)'s +0.458 are five times it; a +0.05 gain would be
    indistinguishable from having changed nothing.

    This row is the only one that could measure it, precisely because it accepted
    nothing.

!!! warning "And its baseline is not comparable with the others'"
    `WindowedMemory.render` emits `MEMORY_HEADER` **even when the memory is
    empty**, so this port's seed artifact reads:

    > Solve the grade-school math word problem. Return only the final answer.
    > \# Plans from past attempts. You have attempted problems like this before
    > and failed; these plans say how to avoid failing the same way. Use them to
    > improve your strategy (most recent last).
    > (empty)

    Every other port starts from the first line alone. That header is itself an
    instruction to be careful, and this row's baseline is 0.73–0.80 where theirs
    are 0.38–0.61. Reflexion did not start where they started, so its *gain* is
    not theirs to compare against — only its final score is.

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
