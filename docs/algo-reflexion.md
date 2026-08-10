# Reflexion — Verbal reinforcement / episodic memory

> **Memory self-evolution.** Turn a failed trajectory into a verbal reflection,
> append it to a bounded episodic memory, and retry with that memory in context.
> Runs through the shared [`MethodPolicy`](policies.md) runner. Example:
> [`examples/reflexion/reflexion_episodic_memory.py`](https://github.com/Birfy/agentdescent/blob/main/examples/reflexion/reflexion_episodic_memory.py).

| | |
|---|---|
| **Paper** | *Reflexion: Language Agents with Verbal Reinforcement Learning* — Shinn et al., 2023 ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) |
| **Upstream code** | [noahshinn/reflexion@218cf0ef](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0) |
| **Example** | [`examples/reflexion/reflexion_episodic_memory.py`](https://github.com/Birfy/agentdescent/blob/main/examples/reflexion/reflexion_episodic_memory.py) |
| **Domain** | **GSM-Hard** ([`reasoning-machines/gsm-hard`](https://huggingface.co/datasets/reasoning-machines/gsm-hard)), 64/64/64 shuffled splits |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `mechanism_microport` — [what the classes mean](port-fidelity.md) |

This port is measured in the [runtime matrix](matrix-overview.md): the mechanism
is preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

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

## Measured results — GSM-Hard

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge on (this method's own declaration), `deepseek-v4-flash` at
temperature 0.7. Recorded in
[`bench/results/reflexion-gsmhard.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/reflexion-gsmhard.json).

| seed | test quality | validation | accepted | invalid | calls |
|---|---|---|---|---|---|
| 0 | 0.656 → **0.688** | 0.406 → 0.469 | 1/80 | 2 | 445 |
| 1 | 0.562 → **0.547** | 0.484 → 0.531 | 1/80 | 4 | 434 |
| 2 | 0.531 → **0.562** | 0.547 → 0.609 | 1/80 | 4 | 515 |

**Validation rises on every seed (+0.047 to +0.063) and test barely moves
(+0.016 mean, one seed negative).** That gap is the port's actual finding, and
it is the honest answer to the question this port asks: a reflection written
from one failure does raise the score on the split the gate judges, and
transfers to unseen questions weakly at best. Reflexion's memory is per
*instance* upstream and its whole move is retrying **that** instance, so a
shared memory asked to generalise is a question the paper does not ask.
`--per-instance` runs the faithful variant and is expected to accept nothing.

!!! note "Why GSM-Hard, and what the `invalid` column counts"
    **A memory fed by failures needs a domain that produces them.** On GSM8K
    this port accepted **0 of 80 on all three seeds**
    ([the run](https://github.com/Birfy/agentdescent/blob/main/bench/results/reflexion-gsm8k.json)
    is kept): held-out sat at 0.75–0.80, four rollouts in five succeeded and
    wrote nothing, and the failures that did occur shared no cause. GSM-Hard is
    the same 1319 questions with large numbers substituted — held-out starts at
    0.41–0.55 and the failures concentrate on one cause, arithmetic done in the
    model's head, which is a failure mode a transferable rule can address.

    That GSM8K run is also where the study's **±0.09 noise floor** comes from:
    nothing committed, so each pair is two evaluations of one unchanged
    instruction, and they differ by +0.094, −0.062 and −0.078.

    `invalid` counts `WindowedMemory`'s `is_a_plan` validator rejecting an
    entry — six alphabetic words, a **shape** floor rather than a quality bar,
    because whether a plan is any good is the held-out gate's question. It exists
    because the reflection prompt ends in `New plan:` above a failed question, so
    a bare number is a live continuation, and a bounded memory would let one
    displace a real plan.

!!! warning "Its baseline is not comparable with the other ports'"
    `WindowedMemory.render` emits `MEMORY_HEADER` **even when the memory is
    empty**, so this port's seed artifact reads:

    > Solve the grade-school math word problem. Return only the final answer.
    > \# Plans from past attempts. You have attempted problems like this before
    > and failed; these plans say how to avoid failing the same way. Use them to
    > improve your strategy (most recent last).
    > (empty)

    Every other port starts from the first line alone, and that header is itself
    an instruction to be careful. Reflexion did not start where they started, so
    its *gain* is not theirs to compare against — only its final score is.

## Run it

```bash
python -m examples.reflexion.reflexion_episodic_memory --dry-run

# one seed of the three above
python -m examples.reflexion.reflexion_episodic_memory --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --async-ratio 2 --workers 8 --budget-rollouts 80 --staleness full \
    --temperature 0.7 --max-seconds 3600
```

`--async-ratio 2` is explicit because the default here is 1, and 2 is what
this row was recorded at.

`--per-instance` runs the faithful variant. Flags:
[the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_reflexion_upstream.py`,
`tests/test_candidate_methods.py`.
